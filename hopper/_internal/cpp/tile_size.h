/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <tuple>

// Return {kBlockM, kBlockN, MmaPV_is_RS, IntraWGOverlap}
constexpr std::tuple<int, int, bool, bool> tile_size_fwd_sm90(
        int headdim, int headdim_v, bool is_causal, bool is_local, int element_size=2,
        bool v_colmajor=false, bool paged_kv_non_TMA=false, bool softcap=false, bool is_int8 = false) {
    const bool maybe_intra_wg_overlap = true;

    if (element_size == 2) {
        if (headdim <= 64) {
            if (headdim_v == 512) {
                return {64, 64, false, false};
            } else if (headdim_v == 256) {
                return {128, 96, true, false};
            } else {
                bool const use_blockN_128 = is_causal || is_local || paged_kv_non_TMA;
                return {192, use_blockN_128 ? 128 : 192, use_blockN_128, true};
            }
        } else if (headdim <= 96) {
            return {192, is_local || paged_kv_non_TMA ? 128 : 144, false, true && maybe_intra_wg_overlap};
        } else if (headdim <= 128) {
            return {128, 128, true, true && maybe_intra_wg_overlap};
        } else if (headdim <= 192) {
            return {128, paged_kv_non_TMA || is_local ? 96 : (headdim_v <= 128 ? 128 : 112), true, true && maybe_intra_wg_overlap};
        } else {
            return {128, is_local ? 64 : 80, true, true && maybe_intra_wg_overlap};
        }
    } else {
        if (is_int8){
            if (headdim <= 64) {
                bool const use_blockN_128 = is_causal || is_local || paged_kv_non_TMA;
                return {192, use_blockN_128 ? 128 : 192, use_blockN_128, true};
            } else if (headdim <= 96) {
                return {192, is_local || paged_kv_non_TMA ? 128 : 144, false, true && maybe_intra_wg_overlap};
            } else if (headdim <= 128) {
                return {192, 128, false, true && maybe_intra_wg_overlap};
            } else if (headdim <= 192) {
                return {128, paged_kv_non_TMA || is_local ? 96 : (headdim_v <= 128 ? 128 : 112), true, true && maybe_intra_wg_overlap};
            } else {
                return {128, is_local ? 64 : 80, true, true && maybe_intra_wg_overlap};
            }
        }else{
            if (headdim <= 64) {
                return {192, 160, true, true && maybe_intra_wg_overlap};
            } else if (headdim <= 96) {
                return {192, 128, true, true && maybe_intra_wg_overlap};
            } else if (headdim <= 128) {
                return {128, paged_kv_non_TMA ? 160 : (v_colmajor || (softcap && is_local) ? 192 : 224), true, true};
            } else if (headdim <= 192) {
                return {128, (paged_kv_non_TMA || softcap) && is_local ? 128 : 160, true, true && maybe_intra_wg_overlap};
            } else {
                return {128, is_local ? 64 : 128, true, !paged_kv_non_TMA && maybe_intra_wg_overlap};  // PagedKV uses more registers so we disabled IntraWGOverlap
            }
        }
    }
}

// Return {kBlockM, kBlockN, kNWarps, kStages, Q_in_regs}
constexpr std::tuple<int, int, int, int, bool> tile_size_fwd_sm8x(
        bool sm86_or_89, int headdim, int headdim_v, bool is_causal, bool is_local, int element_size=2,
        bool paged_kv=false, bool varlen_and_split=false,
        bool softcap=false, bool append_kv=false) {
    if (element_size == 2) {
        if (headdim <= 64) {
            return {128, varlen_and_split ? 80 : (is_local ? 96 : 112), 4, 1, false};
        } else if (headdim <= 96) {
            return {128, varlen_and_split || is_local ? 48 : 64, 4, 1, false};
        } else if (headdim <= 128) {
            bool const use_8_warps = sm86_or_89 | varlen_and_split;
            return {128, use_8_warps ? (varlen_and_split ? (is_local ? 96 : 112) : (is_local ? 96 : 128)) : (is_local ? 48 : 64), use_8_warps ? 8 : 4, 1, use_8_warps};
        } else if (headdim <= 192) {
            bool const kBlockN_64 = append_kv || is_local || varlen_and_split || paged_kv;
            return {128, kBlockN_64 ? 64 : 96, 8, sm86_or_89 ? 1 : 2, !kBlockN_64};
        } else {
            return {128, sm86_or_89 ? (append_kv ? 32 : (varlen_and_split || is_local ? 48 : 64)) : (append_kv ? 48 : (varlen_and_split || is_local ? 64 : 96)), 8, 1, sm86_or_89 && !append_kv};
        }
    } else {
        // Placeholder for now
        return {128, 64, 8, 2, false};
    }
}
