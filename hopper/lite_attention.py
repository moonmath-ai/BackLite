"""
LiteAttention: A lightweight Flash Attention 3 wrapper with sparse backward pass.

This module provides a clean interface for Flash Attention 3 with optional
INT8 quantization and mass-based sparsity for the backward pass.

Sparse Backward Pass:
=====================
The key optimization saves per-tile LSE statistics during the forward pass
and uses them to generate a block sparsity mask for the backward kernel,
skipping tiles that contribute negligible probability mass.
"""

import torch
import math
from typing import Optional, Tuple, Union

from ._internal.flash_attn_interface import flash_attn_func

# Import the C++ extension to register operators with PyTorch
import lite_attention._C  # noqa: F401
_lite_attention_ops = torch.ops.lite_attention


class LiteAttention:
    """
    A lightweight attention class that encapsulates Flash Attention 3.

    Supports optional INT8 quantization (SageAttention-style) and mass-based
    backward sparsity via ``negl_prob``.

    Args:
        use_int8 (bool): Whether to use INT8 quantization for Q and K.
        negl_prob (float): Negligible probability mass threshold for backward
            sparsity.  Values > 0 enable per-tile LSE collection in the forward
            pass and block-sparse masking in the backward pass.

    Example:
        >>> attn = LiteAttention(negl_prob=0.01)
        >>> output = attn(query, key, value)
    """

    def __init__(self, use_int8: bool = False, negl_prob: float = 0.0, **_kwargs):
        self.use_int8 = use_int8
        self.negl_prob = negl_prob

    # ------------------------------------------------------------------
    # Tile-size query
    # ------------------------------------------------------------------

    @staticmethod
    def ceil_div(x, y):
        """Ceiling division utility function."""
        return (x + y - 1) // y

    @staticmethod
    def get_MN(head_dim, dtype, v_colmajor=False):
        """
        Get the tile sizes (block dimensions) for attention computation.

        Args:
            head_dim (int): Dimension of each attention head.
            dtype (torch.dtype): Data type of the tensors.
            v_colmajor (bool): Whether value tensor is column-major.

        Returns:
            tuple[int, int]: (kBlockM, kBlockN)
        """
        is_int8 = dtype == torch.int8
        element_size = dtype.itemsize
        result = _lite_attention_ops.get_tile_size_fwd_sm90(
            head_dim, head_dim, False, False, element_size,
            v_colmajor, False, False, is_int8,
        )
        return result[0], result[1]

    # ------------------------------------------------------------------
    # INT8 quantization
    # ------------------------------------------------------------------

    def _quantize_query_key(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        SageAttention-style quantization for Q and K.

        Q: per-block quantization (kBlockM tokens share a scale per head).
        K: smooth by subtracting channel-wise mean, then per-block quantization.
        """
        if self.use_int8:
            k_mean = key.mean(dim=1).float().contiguous()
            head_dim = query.shape[-1]
            q_scale = (1.44269504089 * scale) if scale is not None else (1.44269504089 / math.sqrt(head_dim))
            q_int8, k_int8, q_descale, k_descale = _lite_attention_ops.quantize_qk(
                query, key, k_mean, False, q_scale,
            )
            return q_int8, k_int8, q_descale, k_descale
        return query, key, None, None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: Optional[float] = None,
        return_softmax_lse: bool = False,
        tile_stats: Optional[torch.Tensor] = None,
        **_kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform Flash Attention 3 computation.

        Args:
            query: (batch, seq_len, heads, head_dim)
            key:   (batch, seq_len, heads_k, head_dim)
            value: (batch, seq_len, heads_k, head_dim)
            scale: Optional softmax scale (default 1/sqrt(head_dim)).
            return_softmax_lse: Return (output, lse) tuple.
            tile_stats: Pre-allocated tile-stats tensor; if *None* and
                ``negl_prob > 0`` the tensor is allocated automatically by
                the autograd wrapper.

        Returns:
            Attention output, or (output, lse) when *return_softmax_lse* is True.
        """
        query, key, q_descale, k_descale = self._quantize_query_key(query, key, scale)

        output = flash_attn_func(
            q=query,
            k=key,
            v=value,
            softmax_scale=None if self.use_int8 else scale,
            q_descale=q_descale,
            k_descale=k_descale,
            return_softmax_lse=return_softmax_lse,
            tile_stats=tile_stats,
            negl_prob=self.negl_prob,
        )
        return output


class SeqParallelLiteAttention:
    """
    Sequence-parallel wrapper around :class:`LiteAttention`.

    Creates one :class:`LiteAttention` instance per node and routes calls
    based on ``split_idx``.

    Args:
        num_nodes (int): Number of sequence-parallel nodes.
        use_int8 (bool): Whether to use INT8 quantization.
        negl_prob (float): Negligible probability mass threshold.
    """

    def __init__(self, num_nodes: int, use_int8: bool = False, negl_prob: float = 0.0, **_kwargs):
        self.num_nodes = num_nodes
        self.lite_attention = [
            LiteAttention(use_int8=use_int8, negl_prob=negl_prob)
            for _ in range(num_nodes)
        ]

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        split_idx: int,
        scale: Optional[float] = None,
        return_softmax_lse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform attention for a specific sequence-parallel node.

        Args:
            split_idx (int): Node index (0 .. num_nodes-1).
        """
        assert split_idx < self.num_nodes, "split_idx must be less than num_nodes"
        return self.lite_attention[split_idx](
            query, key, value, scale, return_softmax_lse,
        )
