# BackLite

### [Project Page](https://moonmath-ai.github.io/BackLite/) | [MoonMath.ai](https://moonmath.ai)

**BackLite** is a lightweight [Flash Attention 3](https://github.com/Dao-AILab/flash-attention) wrapper that accelerates training via a **sparse backward pass**. By collecting per-tile LSE statistics during the forward pass, BackLite identifies attention tiles whose cumulative probability mass is negligible and skips them entirely during gradient computation.


## 📖 Overview

Attention backward passes are a dominant cost in transformer training. BackLite reduces this cost by making the backward pass **block-sparse**: tiles that contribute less than a configurable probability mass threshold (`negl_prob`) to the output are skipped when computing gradients, with no impact on the forward pass or output.

Key properties:
- **Accurate**: Only tiles with negligible probability mass (< `negl_prob`) are skipped — gradients for significant tiles are computed exactly.
- **Zero forward overhead** when `negl_prob = 0` (falls back to standard FA3).
- **Adaptive**: Sparsity is derived from actual attention statistics per sample, per head.
- **Composable**: Supports LSE output for combining partial attention results.

## 🔍 How It Works

### Sparse Backward Masking

BackLite introduces a two-phase mechanism:

**Phase 1 — Forward pass with tile statistics**: When `negl_prob > 0`, the FA3 forward kernel records the per-tile log-sum-exp (LSE) for every `(query tile, key tile)` pair alongside the standard output and full-row LSE.

**Phase 2 — Backward pass with block-sparse mask**: Before the backward kernel runs, a fused Triton kernel (`mask_from_stats_fused`) converts the tile LSEs into a block-sparse boolean mask. For each backward tile pair, it computes:

$$p_\text{tile} = \exp\!\left(\mathrm{LSE}_\text{tile} - \mathrm{LSE}_\text{row}\right)$$

If $p_\text{tile}$ is less than `negl_prob`, the tile is marked as skippable and the backward kernel bypasses it entirely — no memory reads, no FLOPs.

This approach:
- Introduces **no approximation in the forward pass**
- Introduces **negligible gradient error** for properly chosen `negl_prob` (tiles with $< 1\%$ mass contribute $< 1\%$ to the gradient)
- Generates the sparsity mask in a **single Triton pass** over the stored tile statistics

## 📊 Backward Sparsity

<!-- TODO: add accurate sparsity / gradient-error table from benchmark results -->
### TBD

## 🔧 Installation

### Requirements
- H100 / H200 GPU
- CUDA >= 12.8
- CUDA toolkit
- C++ 20
- PyTorch 2.2 and above
- `packaging` Python package (`pip install packaging`)
- `ninja` Python package (`pip install ninja`) *
- Linux

\* Make sure that `ninja` is installed and that it works correctly (e.g. `ninja --version` then `echo $?` should return exit code 0). If not (sometimes `ninja --version` then `echo $?` returns a nonzero exit code), uninstall then reinstall `ninja` (`pip uninstall -y ninja && pip install ninja`). Without `ninja`, compiling can take a very long time (2h) since it does not use multiple CPU cores. With `ninja` compiling takes 3-5 minutes on a 64-core machine using CUDA toolkit.

### Build from Source

Clone this repo and build from source:

```sh
git clone https://github.com/moonmath-ai/BackLite.git
cd BackLite/hopper
pip install .
```

If your machine has less than 96GB of RAM and lots of CPU cores, `ninja` might run too many parallel compilation jobs that could exhaust the amount of RAM. To limit the number of parallel compilation jobs, you can set the environment variable `MAX_JOBS`:

```sh
MAX_JOBS=4 pip install .
```

## 🚀 Usage

### Basic Usage (Single GPU)

```python
BackLite(negl_prob: float = 0.05)
```

**Parameters:**
- `negl_prob` (float): Negligible probability mass threshold for backward sparsity. Tiles whose cumulative attention mass is below this value are skipped during gradient computation. Set to `0.0` to use standard FA3 with no sparsity. Typical values: `0.01`–`0.1`.

```python
from back_lite import BackLite

# Standard FA3 (no backward sparsity)
attn = BackLite()
output = attn(query, key, value)

# Sparse backward pass — skip tiles contributing < 5% probability mass
attn = BackLite(negl_prob=0.05)
output = attn(query, key, value)

# With explicit softmax scale
output = attn(query, key, value, scale=1.0 / math.sqrt(head_dim))

# Forward + backward example
attn = BackLite(negl_prob=0.05)
output = attn(query, key, value)
loss = output.sum()
loss.backward()  # backward uses the sparse mask generated in the forward pass
```

**Forward signature:**
```python
attn(
    query,               # (batch, seq_len, heads, head_dim)
    key,                 # (batch, seq_len, heads_k, head_dim)
    value,               # (batch, seq_len, heads_k, head_dim)
    scale=None,          # softmax scale, default 1/sqrt(head_dim)
    return_softmax_lse=False,  # return (output, lse) instead of output
    tile_stats=None,     # pre-allocated tile-stats buffer (optional)
)
```

> [!NOTE]
> When `negl_prob > 0`, BackLite automatically allocates a `tile_stats` buffer during the forward pass to store per-tile LSE statistics. This buffer is passed to the autograd `backward` function, where a Triton kernel converts it into a block-sparse mask before the backward kernel runs. No changes to your training loop are required.

## 📝 Integration Example

<!-- TODO: Fill in full integration example -->
### TBD

## 🐛 Debugging

Set `BACK_LITE_VERBOSE=1` to enable additional logging (tile statistics, mask density, backward sparsity ratio) during training.

## 🙏 Acknowledgements

BackLite is built on top of [FlashAttention3](https://github.com/Dao-AILab/flash-attention) by Tri Dao and contributors. We thank the FlashAttention team for their foundational work on efficient attention mechanisms.

We also thank the teams behind [SparseVideoGen](https://github.com/svg-project/Sparse-VideoGen), [RadialAttention](https://github.com/mit-han-lab/radial-attention), [SageAttention](https://github.com/thu-ml/SageAttention), [Wan2.1](https://github.com/Wan-Video/Wan2.1), and [LTX-Video](https://github.com/Lightricks/LTX-Video) for their insights and benchmarking support.

## License

BackLite is build on top of FA3 which has a BSD 3-Clause license. As such the original code maintains that license and any new code for BackLite is distributed under an MIT license.

See [LICENSE-BSD](LICENSE-BSD) and [LICENSE-MIT](LICENSE-MIT) for further details.
