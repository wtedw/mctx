# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A JAX implementation of batched MCTS."""
import functools
from typing import Any, NamedTuple, Optional, Tuple, TypeVar

import chex
import jax
import jax.numpy as jnp

from mctx._src import action_selection
from mctx._src import base
from mctx._src import seq_halving
from mctx._src import tree as tree_lib


Tree = tree_lib.Tree
T = TypeVar("T")

def fast_gather_child(src: jnp.ndarray,
                      parent: jnp.ndarray,  # [B]  (node axis)
                      act: jnp.ndarray      # [B]  (action axis)
                     ) -> jnp.ndarray:      # [B]
  B, N, A = src.shape
  # One‑hot over both axes ⇒ mask ∈ [B, N, A]
  m_parent = jax.nn.one_hot(parent, N, dtype=src.dtype)         # [B, N]
  m_action = jax.nn.one_hot(act,    A, dtype=src.dtype)         # [B, A]
  # Outer‑product → [B, N, A]
  mask = m_parent[:, :, None] * m_action[:, None, :]

  # Batched 3‑D contraction.  Contract (N,A) of mask with (N,A) of src.
  out = jnp.einsum('bna,bna->b', mask, src)
  return out

def fast_gather_1d(src: jnp.ndarray,      # [N]  (int32 / bf16 / f32 …)
                   idx: jnp.ndarray       # [M]  int32
                  ) -> jnp.ndarray:       # [M]
    """Scatter-free gather for rank-1 tensors (no batch dim)."""
    one_hot = jax.nn.one_hot(idx, src.shape[0], dtype=src.dtype)  # [M, N]
    return jnp.einsum('mn,n->m', one_hot, src)                    # [M]

# ---------------------------------------------------------------------
# rank-2 gather:  out[m] = src[row_M[m], col_M[m]]
# ---------------------------------------------------------------------
def gather_NA(src_NA: jnp.ndarray,         # [N, A]
              row_M:  jnp.ndarray,         # [M]  int32
              col_M:  jnp.ndarray          # [M]  int32
             ) -> jnp.ndarray:             # [M]
  """
  Scatter-free gather from a rank-2 table inside a vmap body.

  • src_NA  : 2-D tensor you are indexing (children_rewards[b] etc.)
  • row_M   : parent indices   (length M, NOT batched)
  • col_M   : action  indices  (length M, NOT batched)
  • returns : value for every explorer m
  """
  N, A  = src_NA.shape
  dtype = src_NA.dtype

  rmask = jax.nn.one_hot(row_M, N, dtype=dtype)       # [M, N]
  cmask = jax.nn.one_hot(col_M, A, dtype=dtype)       # [M, A]

  # outer-product → [M, N, A]; then contract the last two axes
  # einsum lowers to a single dot_general → one HLO op
  return jnp.einsum('mn,ma,na->m', rmask, cmask, src_NA)

def fast_gather_rows(bn: jnp.ndarray,          # [B, N]
                     idx_bm: jnp.ndarray       # [B, M]  int32
                    ) -> jnp.ndarray:          # [B, M]
  mask = jax.nn.one_hot(idx_bm, bn.shape[1], dtype=bn.dtype)  # [B,M,N]
  return jnp.einsum('bmn,bn->bm', mask, bn)

# ---------------------------------------------------------------------
# Overwrite one row  – rank‑2  ([B,N] or [N])
# ---------------------------------------------------------------------
def set_row(x: jnp.ndarray,                # [B,N]  or [N]
            row_idx: jnp.ndarray,          # [B]    or scalar
            val: jnp.ndarray):             # [B]    or scalar
    """x[..., row_idx[b]] ← val[b]  (scatter‑free)."""
    mask = jax.nn.one_hot(row_idx, x.shape[-1], dtype=x.dtype)     # [..., N]

    # make val broadcast along the N‑axis
    val_exp = jnp.expand_dims(val, axis=-1)                        # [..., 1]

    if jnp.issubdtype(x.dtype, jnp.integer):
        return jnp.where(mask.astype(bool), val_exp.astype(x.dtype), x)

    return x * (1 - mask) + val_exp * mask

# v1
# def set_row_cols(x:         jnp.ndarray,   # [B, N, A]  – tensor to update
#                  row_idx:   jnp.ndarray,   # [B]        – which row per batch
#                  col_idx:   jnp.ndarray,   # [B, M]     – cols to overwrite
#                  vals:      jnp.ndarray    # [B, M]     – new values
#                 ) -> jnp.ndarray:
#   """
#   Scatter-free replacement for:
#       x[batch, row_idx[b], col_idx[b, m]] = vals[b, m]

#   Works for float tensors (children_values) and int tensors (children_visits).

#   • x         : [B, N, A]  – any dtype
#   • row_idx   : [B]        – per-batch row (e.g. ROOT_INDEX=0)
#   • col_idx   : [B, M]     – per-batch list of columns being written
#   • vals      : [B, M]     – value for each (row_idx, col_idx) pair
#   """
#   B, N, A = x.shape
#   M       = col_idx.shape[1]
#   dtype   = x.dtype

#   # 1) one-hot over columns  →  mask_c ∈ {0,1}^{B×M×A}
#   mask_c  = jax.nn.one_hot(col_idx, A, dtype=dtype)           # [B,M,A]
#   new_row = jnp.sum(mask_c * vals[..., None].astype(dtype),   # [B,M,1]
#                     axis=1)                                   # [B,A]

#   # 2) which columns are touched?
#   any_mask = jnp.minimum(jnp.sum(mask_c, axis=1), 1).astype(dtype)  # [B,A]

#   # 3) gather current row *without* scatter/gather ops
#   mask_r   = jax.nn.one_hot(row_idx, N, dtype=dtype)          # [B,N]
#   old_row  = jnp.sum(mask_r[:, :, None] * x, axis=1)          # [B,A]

#   # 4) blend: keep untouched cols, overwrite the visited ones
#   blended  = old_row * (1 - any_mask) + new_row

#   # 5) write the whole row back with the existing `set_row`
#   return set_row(x, row_idx, blended)

# v2
def set_row_cols(x:        jnp.ndarray,   # [B, N, A]
                 row_idx:  jnp.ndarray,   # [B]        row per batch (axis-1)
                 col_idx:  jnp.ndarray,   # [B, M]     columns to overwrite
                 vals:     jnp.ndarray    # [B, M]     new values
                ) -> jnp.ndarray:
  """
  Scatter-free update:
      x[b, row_idx[b], col_idx[b, m]] ← vals[b, m]

  Works for float or int dtypes with no branches of any kind.
  """
  dtype        = x.dtype
  B, N, A      = x.shape
  M            = col_idx.shape[1]

  # ------------------------------------------------------------------
  # 1) masks
  # ------------------------------------------------------------------
  row_mask     = jax.nn.one_hot(row_idx, N, dtype=dtype)        # [B,N]
  row_mask3    = row_mask[..., None]                            # [B,N,1]

  col_mask     = jax.nn.one_hot(col_idx, A, dtype=dtype)        # [B,M,A]
  col_mask_sum = jnp.sum(col_mask, axis=1)                      # [B,A]  (0/1)

  # ------------------------------------------------------------------
  # 2) build the new row we want to insert         new_row[b, a]
  # ------------------------------------------------------------------
  new_row      = jnp.sum(col_mask * vals[..., None].astype(dtype), axis=1)  # [B,A]

  # ------------------------------------------------------------------
  # 3) current contents of that row                old_row[b, a]
  # ------------------------------------------------------------------
  old_row      = jnp.sum(x * row_mask3, axis=1)                 # [B,A]

  # ------------------------------------------------------------------
  # 4) delta we need to apply at the chosen columns
  #     delta = -old_row*mask + new_row
  # ------------------------------------------------------------------
  # [ted] this will cause a neg 0
  # delta_row    = -old_row * col_mask_sum + new_row              # [B,A]
  delta_row = (new_row - old_row) * col_mask_sum   # 0 outside the mask


  # ------------------------------------------------------------------
  # 5) write back — add the delta only at the selected row
  # ------------------------------------------------------------------
  x_updated    = x + row_mask3 * delta_row[:, None, :]          # [B,N,A]
  return x_updated

# # ------------------------------------------------------------------
# # helper: scatter-free row/col update *with* validity mask
# # ------------------------------------------------------------------
# def set_row_cols_masked(x, row_idx, col_idx, vals, valid):
#   """
#   Same signature as `set_row_cols` with an extra
#       valid[b, m] ∈ {0,1}
#   Only (row_idx[b], col_idx[b,m]) with valid==1 are overwritten.
#   """
#   mask_c    = jax.nn.one_hot(col_idx, x.shape[-1], dtype=x.dtype) * valid[..., None]  # [B,M,A]
#   new_row   = (mask_c * vals[..., None]).sum(1)                               # [B,A]
#   any_mask  = mask_c.sum(1).clip(max=1.)                                      # [B,A]

#   row_mask3 = jax.nn.one_hot(row_idx, x.shape[1], dtype=x.dtype)[..., None]           # [B,N,1]
#   old_row   = (x * row_mask3).sum(1)                                          # [B,A]
#   delta     = -old_row * any_mask + new_row                                   # [B,A]

#   return x + row_mask3 * delta[:, None, :]

# attempt 2, preserves dtype of x
def set_row_cols_masked(x, row_idx, col_idx, vals, valid):
  """
  Scatter-free update of x[b,row,col] = vals[b,m] * valid[b,m]
  preserving the dtype of `x` (int or float).
  """
  B, N, A = x.shape
  dtype   = x.dtype

  # 1) column mask with validity
  col_mask = jax.nn.one_hot(col_idx, A, dtype=x.dtype)            # [B,M,A]
  col_mask = col_mask * valid[..., None]                  # still dtype

  # 2) new values to write into the row
  new_row  = (col_mask * vals[..., None].astype(dtype)).sum(1)   # [B,A]

  # 3) which columns are touched?
  any_mask = jnp.minimum(col_mask.sum(1), dtype.type(1))         # [B,A]

  # 4) current contents of the row we overwrite
  row_mask3 = jax.nn.one_hot(row_idx, N, dtype=x.dtype)[..., None]       # [B,N,1]
  old_row   = (x * row_mask3).sum(1)                             # [B,A]

  # 5) delta we inject at those columns
  delta_row = -old_row * any_mask + new_row                      # [B,A]

  updated   = x + row_mask3 * delta_row[:, None, :]              # [B,N,A]
  return updated.astype(dtype)         # <- keeps int arrays as int


# ---------------------------------------------------------------------
# Overwrite one (parent,action) cell  – rank‑3 ([B,N,A] or [N,A])
# ---------------------------------------------------------------------
def set_cell(x: jnp.ndarray,               # [B,N,A]  or [N,A]
             parent: jnp.ndarray,          # [B]      or scalar
             action: jnp.ndarray,          # [B]      or scalar
             val: jnp.ndarray):            # [B]      or scalar
    """
    x[..., parent[b], action[b]] ← val[b]
    Works with or without a batch axis.
    """
    parent_oh = jax.nn.one_hot(parent, x.shape[-2], dtype=x.dtype)  # [..., N]
    action_oh = jax.nn.one_hot(action, x.shape[-1], dtype=x.dtype)  # [..., A]

    mask = parent_oh[..., None] * action_oh[..., None, :]           # [...,N,A]
    val_exp = val[..., None, None]                                  # [...,1,1]

    if jnp.issubdtype(x.dtype, jnp.integer):
        return jnp.where(mask.astype(bool), val_exp.astype(x.dtype), x)

    return x * (1 - mask) + val_exp * mask

# --------------------------------------------------------------
#  add an entire vector to row  idx  in a  [B, N, A]  tensor
# --------------------------------------------------------------
def add_row_vec(x: jnp.ndarray,            # [B,N,A]
                row_idx: jnp.ndarray,      # [B]
                vec: jnp.ndarray):         # [B,A]
    mask = jax.nn.one_hot(row_idx, x.shape[1], dtype=x.dtype)  # [B,N]
    mask = mask[:, :, None]                                    # [B,N,1]
    return x + vec[:, None, :].astype(x.dtype) * mask

def add_row_vec_2d(x_NA: jnp.ndarray,        # [N, A]
                   row:  int | jnp.ndarray,  # scalar
                   vec_A: jnp.ndarray):      # [A]
    """x[row, a] += vec[a]   –– scatter-free, rank-2."""
    N, A = x_NA.shape
    row_mask = jax.nn.one_hot(row, N, dtype=x_NA.dtype)   # [N]
    return x_NA + row_mask[:, None] * vec_A[None, :].astype(x_NA.dtype)

def set_row_any_sparse(x: jnp.ndarray,       # [B, N, ...F]
                       row_idx: jnp.ndarray, # [B]
                       val: jnp.ndarray) -> jnp.ndarray:
    """
    Overwrite row_idx[b] with val[b] using pure addition.
    Assumes `x` is zero-initialized and written only once.
    """
    B, N = x.shape[:2]
    feat_shape = x.shape[2:]

    mask = jax.nn.one_hot(row_idx, N, dtype=x.dtype)     # [B, N]
    mask = mask.reshape((B, N) + (1,) * len(feat_shape)) # [B, N, 1, ..., 1]

    val = val.astype(x.dtype).reshape((B, 1) + feat_shape)  # [B, 1, ...F]

    return x + mask * val

def set_rows(
    x:   jnp.ndarray,    # [B, N, ...F], must be float
    idx: jnp.ndarray,    # [B, K], row indices
    val: jnp.ndarray,    # [B, K, ...F], new row‐vectors
) -> jnp.ndarray:
  B, N = x.shape[:2]
  F_shape = x.shape[2:]
  K = idx.shape[1]

  # 1) one-hot mask over (B,K,N)
  mask = jax.nn.one_hot(idx, N, dtype=x.dtype)      # [B, K, N]
  # reshape to broadcast over the feature dims
  mask = mask.reshape((B, K, N) + (1,) * len(F_shape))  # [B, K, N, ...F=1]

  # 2) expand val to [B, K, 1, ...F]
  val_exp = val.reshape((B, K, 1) + F_shape)

  # 3) scatter via multiply & sum → [B, N, ...F]
  inserted = jnp.sum(mask * val_exp, axis=1)

  # 4) sum mask over K to get counts, clamp to [0,1] → [B, N, 1…]
  count_mask = jnp.sum(mask, axis=1)                   # [B, N, 1…]
  any_mask   = jnp.minimum(count_mask, 1.0)            # still float
  # ensure same trailing dims as x
  any_mask   = any_mask.reshape((B, N) + (1,) * len(F_shape))

  # 5) blend old and new purely with arithmetic
  return x * (1 - any_mask) + inserted * any_mask

def set_rows_anydtype(
    x:   jnp.ndarray,    # [B, N, ...F]  – bool / int / float
    idx: jnp.ndarray,    # [B, K]        – rows to overwrite
    val: jnp.ndarray,    # [B, K, ...F]  – replacement rows
) -> jnp.ndarray:
  """
  Scatter-free multi-row update that:
    • works for bool, integers, floats
    • uses only multiply/add inside the kernel (good for TPUs)
    • performs *one* cast to f32 and *one* cast back.
  """
  B, N = x.shape[:2]
  F_shape = x.shape[2:]
  K = idx.shape[1]

  # --- 1. build float32 one-hot mask  m[b,k,n] ∈ {0,1} --------------------
  mask = jax.nn.one_hot(idx, N, dtype=jnp.float32)                 # [B,K,N]
  mask = mask.reshape((B, K, N) + (1,) * len(F_shape))             # [B,K,N,1…]

  # --- 2. cast inputs to f32 once -----------------------------------------
  x_f32   = x.astype(jnp.float32)
  val_f32 = val.astype(jnp.float32).reshape((B, K, 1) + F_shape)

  # --- 3. aggregate K updates & build  any_mask[b,n,…] ∈ {0,1} -----------
  inserted  = jnp.sum(mask * val_f32, axis=1)                       # [B,N,…]
  any_mask  = jnp.minimum(jnp.sum(mask, axis=1), 1.0)               # [B,N,…]

  # --- 4. blend with pure arithmetic in f32 -------------------------------
  out_f32 = x_f32 * (1.0 - any_mask) + inserted                     # [B,N,…]

  # --- 5. cast back to original dtype  ------------------------------------
  return out_f32.astype(x.dtype)

def set_rows_anydtype_masked(
    x:   jnp.ndarray,      # [B, N, …F]
    idx: jnp.ndarray,      # [B, K]      – row indices
    val: jnp.ndarray,      # [B, K, …F]  – replacement rows
    mask: jnp.ndarray,     # [B, K] bool – True ⇔ write this row
) -> jnp.ndarray:
  """
  Scatter-free multi-row update that

    • preserves dtype  (bool / int{8,16,32,64} / float{16,32,64})
    • uses only mul / add in the inner kernel (TPU-friendly)
    • ignores rows whose `mask[b,k]` is False – nothing is written there
      and the arithmetic still has *static* shapes.
  """
  B, N = x.shape[:2]
  F    = x.shape[2:]            # trailing feature dims
  K    = idx.shape[1]

  # 1) one-hot over rows  … dtype=float32 so that mul/add is legal
  oh   = jax.nn.one_hot(idx, N, dtype=jnp.float32)        # [B,K,N]
  oh   = oh * mask[..., None]                             # disable masked rows
  oh   = oh.reshape((B, K, N) + (1,) * len(F))            # [B,K,N,1…F]

  # 2) cast operands once
  x_f32   = x.astype(jnp.float32)
  val_f32 = val.astype(jnp.float32).reshape((B, K, 1) + F)

  # 3) arithmetic blend
  inserted  = jnp.sum(oh * val_f32, axis=1)               # [B,N,…]
  any_mask  = jnp.minimum(jnp.sum(oh, axis=1), 1.0)       # [B,N,…]
  out_f32   = x_f32 * (1.0 - any_mask) + inserted
  # 4) cast back   (see note below)
  if jnp.issubdtype(x.dtype, jnp.bool_):
    return out_f32.astype(bool)           # strict 0/1 → False/True
  else:
    return out_f32.astype(x.dtype)


# ### for BK kernels
# # ──────────────────────────────────────────────────────────────────────────────
# #  NEW  – helpers for arbitrary–layout live explorers
# # ──────────────────────────────────────────────────────────────────────────────
# def pick_live_cols(active_mask: jnp.ndarray, K: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
#   """
#   Parameters
#   ----------
#   active_mask : bool [B, M]     True ⇔ explorer is live this round
#   K           : int             round-specific “max_k” (compile-time constant)

#   Returns
#   -------
#   live_cols : int32 [B, K]      column indices of the live explorers
#   live_mask : bool  [B, K]      False where that batch row has <K survivors
#   """
#   order     = jnp.argsort(~active_mask, axis=1)     # sort live (1) before dead (0)
#   live_cols = order[:, :K].astype(jnp.int32)
#   live_mask = jnp.take_along_axis(active_mask, live_cols, axis=1)
#   return live_cols, live_mask


# def gather_cols(arr: jnp.ndarray, col_idx: jnp.ndarray) -> jnp.ndarray:
#   """
#   arr     : [B, M, …]
#   col_idx : [B, K]
#   returns : [B, K, …]
#   """
#   B, M = arr.shape[:2]
#   one_hot = jax.nn.one_hot(col_idx, M, dtype=arr.dtype)     # [B, K, M]
#   return jnp.einsum('bkm,bm...->bk...', one_hot, arr)


def scatter_cols(accum: jnp.ndarray,
                 col_idx: jnp.ndarray,
                 src: jnp.ndarray,
                 mask: jnp.ndarray) -> jnp.ndarray:
  """
  accum : [B, M, …]   – destination tensor
  col_idx, src, mask have shapes [B, K], [B, K, …], [B, K]
  """
  B, M = accum.shape[:2]
  one_hot = jax.nn.one_hot(col_idx, M, dtype=accum.dtype) * mask[..., None]
  delta   = jnp.einsum('bkm,bk...->bm...', one_hot, src)
  return accum + delta


### [BK]
# ──────────────────────────────────────────────────────────────────────────────
#  helpers
# ──────────────────────────────────────────────────────────────────────────────
def pick_live_cols(active_mask: jnp.ndarray, K: int) -> tuple[jnp.ndarray, jnp.ndarray]:
  """
  active_mask : bool [B, M]      – current “alive” status of every explorer
  K           : int              – how many columns we want to keep this round

  Returns
  -------
  live_cols : int32 [B, K]       – column indices (0‥M-1) to process
  live_mask : bool  [B, K]       – True  ⇔ that slot is *actually* alive
                                   False ⇔ pad column (does nothing)
  """
  B, M = active_mask.shape
  idx  = jnp.broadcast_to(jnp.arange(M, dtype=jnp.int32), (B, M))

  # active columns get small keys, inactive columns get large keys → sort
  sort_keys = jnp.where(active_mask, idx, idx + M)
  col_ord   = jnp.argsort(sort_keys, axis=1)               # [B, M]
  live_cols = col_ord[:, :K]                               # [B, K]
  live_mask = jnp.take_along_axis(active_mask, live_cols, axis=1)
  return live_cols, live_mask


def gather_cols(arr: jnp.ndarray, cols: jnp.ndarray) -> jnp.ndarray:
  """
  arr  : [..., M, …]  (gather *along axis-1*)
  cols : int32 [B, K]
  """
  nd   = arr.ndim
  while cols.ndim < nd:
    cols = cols[..., None]                                 # [B,K,1,…]
  return jnp.take_along_axis(arr, cols, axis=1)





def halve_root_actions_mask(
    round_i,
    active_explorer_mask,
    sampled_glogits,
    explorer_visit_counts,
    explorer_qvalues,
    k_alive,
    *,
    max_explorers,
    use_mixed_value: bool = False,
    value_scale: chex.Numeric = 0.1,
    maxvisit_init: chex.Numeric = 50.0,
    rescale_values: bool = True,
    epsilon: chex.Numeric = 1e-8,
    ):
  """
  next round's active mask requires looking at g + logits + completed qvalues
  active_mask: [B, M]
  action_idxs: [B, M]
  glogits: [B, M]
  qvalues; [B, M],
  k: [B],
  """
  # 2) -----------------------   scorer for **currently live** buckets ------
  # gather the M children we have buckets for
  # jax.debug.print("[halv]OG explorer_qvalues@{}: {}", round_i, explorer_qvalues)
  # jax.debug.print("[halv]OG explorer_visit_counts@{}: {}", round_i, explorer_visit_counts)
  # jax.debug.print("[halv]OG active_explorer_mask@{}: {}", round_i, active_explorer_mask)
  # jax.debug.print("[halv]sampled_glogits: {}", sampled_glogits)
  score_m = jnp.where(active_explorer_mask, sampled_glogits, -jnp.inf)
  # jax.debug.print("[halv]score_m: {}", score_m)

  def root_explorer_layer_qtransform(q1):
    """
    We don't need to complete the qvalues for unvisited actions since we
    only consider the M active explorer's qvalues (which we've already visited)
    """

    # maxvisit = jnp.max(explorer_visit_counts, axis=-1)
    maxvisit = jnp.max(explorer_visit_counts, axis=-1, keepdims=True) # [B, 1]
    visit_scale = maxvisit_init + maxvisit # [B, 1]

    alpha = value_scale * visit_scale
    ## [OG]
    if rescale_values:
        q_min  = jnp.min(q1, axis=1, keepdims=True)
        q_max  = jnp.max(q1, axis=1, keepdims=True)
        q_norm = (q1 - q_min) / jnp.maximum(q_max - q_min, epsilon)
    else:
        q_norm = q1                      # no rescaling
    # ## [BK]
    # q_norm = q1                      # no rescaling

    cq = alpha * q_norm                # completed-Q for the 16 parents
    return cq

  # If we visit and the value is negative, we should pick that over invalid action
  # This will happen when we do top_k with masked_score
  explorer_cqvalues = root_explorer_layer_qtransform(explorer_qvalues)
  masked_score = score_m + explorer_cqvalues
  # jax.debug.print("[halv]explorer_cqvalues@{}: {}", round_i, explorer_cqvalues)
  # jax.debug.print("[halv]masked_score@{}: {}", round_i, masked_score)

  def select_k_best_static(scores: jnp.ndarray,   # [B, M]
                         k_alive: jnp.ndarray,  # [B] – runtime 0‥M
                        #  M: chex.Numeric,
                        ) -> jnp.ndarray:       # [B, M] int32
    """
    Returns an int32 array of shape [B, M] where the left‐most `k_alive[b]`
    entries of each row hold the indices of the top-k actions and the rest are
    filled with `sentinel` (usually 0 or −1).

    All shapes are static – suitable for jit / pmap / while_loop.
    """
    M = max_explorers
    # 1) fixed-K top-k
    _, idx_full = jax.lax.top_k(scores, M)          # [B, M]  static
    # jax.debug.print("[halv]top_k, idx_full@{}: {}", round_i, idx_full)

    ranks = jnp.argsort(idx_full) + 1 # represents for [explorer 0's rank in sorted top_k scores, explorer 1, 2, .. M]

    # 2) build a per-row mask  mask[b, j] = 1  if j < k_alive[b]
    print("k alive shape", k_alive.shape)
    # jax.debug.print("[halv]k alive@{}: {}", round_i, k_alive)
    keep_mask   = jnp.arange(M) < k_alive  # [B, M]  bool

    # ------------------------------------------------------------------
    # 3.1) build one-hot cube:  oh[b, j, m] = 1  if  idx_full[b, j] == m
    # ------------------------------------------------------------------
    one_hot = jax.nn.one_hot(idx_full, M, dtype=bool)        # [B, M, M]

    # ------------------------------------------------------------------
    # 3.2) zero-out the columns that are beyond k_alive (keep_mask == False)
    # ------------------------------------------------------------------
    one_hot = one_hot & keep_mask[..., None]                 # [B, M, M]

    # ------------------------------------------------------------------
    # 3) OR-reduce along the “j” axis  → bool [B, M]
    #    (only one True per column can survive)
    # ------------------------------------------------------------------
    return jnp.any(one_hot, axis=0), ranks


  active_mask_next, ranks = jax.vmap(select_k_best_static)(masked_score, k_alive)
  # jax.debug.print("[halv]new active_mask@{}: {}", round_i, active_mask_next)
  # jax.debug.print("[halv]new ranks@{}: {}", round_i, ranks)
  return active_mask_next, ranks

def calc_explorer_qvalues(tree, active_mask, sampled_actions):
  # ------------------------------------------------------------------
  # 0. helpers
  # ------------------------------------------------------------------
  B, M = sampled_actions.shape
  batch_r  = jnp.arange(B)[:, None]          # [B,1]  broadcast helper
  node_idx = jnp.arange(1, M + 1)            # [M]    children of the root

  # ------------------------------------------------------------------
  # 1. edge-reward  r(b,m)  and discount γ(b,m)
  #     children_{rewards|discounts} are [B, N, A];  N=0 is the root row.
  # ------------------------------------------------------------------
  root_rewards   = tree.children_rewards  [:, tree_lib.Tree.ROOT_INDEX, :]  # [B,A]
  root_discounts = tree.children_discounts[:, tree_lib.Tree.ROOT_INDEX, :]  # [B,A]

  r = jnp.take_along_axis(root_rewards,   sampled_actions, axis=1)          # [B,M]
  γ = jnp.take_along_axis(root_discounts, sampled_actions, axis=1)          # [B,M]

  # ------------------------------------------------------------------
  # 2. leaf-value   v(b,m)  stored in node_values row (node_idx = 1…M)
  # ------------------------------------------------------------------
  ### opt1:
  v = tree.node_values[batch_r, node_idx]                                   # [B,M]
  ### opt2:
  # N = tree.node_values.shape[1]
  # # build the constant selector only **once** per compilation
  # selector = jax.nn.one_hot(
  #     jnp.arange(1, M + 1),          # rows 1…M are the root’s children
  #     N,
  #     dtype=tree.node_values.dtype   # keep dtype (bf16 / f32)
  # )                                  # shape  [M, N]

  # # (B, N) · (N, M)ᵀ → (B, M)
  # v = tree.node_values @ selector.T        # single dot_general


  # ------------------------------------------------------------------
  # 3. completed-Q for every bucket  (reward + γ·value)
  # ------------------------------------------------------------------
  explorer_qvalues = r + γ * v                                              # [B,M]

  # ------------------------------------------------------------------
  # 4. mask-out idle explorers (active_mask == -1)
  # ------------------------------------------------------------------
  explorer_qvalues = jnp.where(active_mask, explorer_qvalues, 0.0)
  return explorer_qvalues


def search2(
    params: base.Params,
    rng_key: chex.PRNGKey,
    *,
    # [BK] stuff
    round_max_k,
    round_kernels,
    # OG stuff
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    root_action_selection_fn: base.RootActionSelectionFn,
    interior_action_selection_fn: base.InteriorActionSelectionFn,
    num_simulations: int,
    max_num_considered_actions: int,
    max_depth: Optional[int] = None,
    invalid_actions: Optional[chex.Array] = None,
    extra_data: Any = None,
    # qtransform stuff
    value_scale: chex.Numeric = 0.1,
    maxvisit_init: chex.Numeric = 50.0,
    rescale_values: bool = True,
    use_mixed_value: bool = True,
    epsilon: chex.Numeric = 1e-8,
  ) -> Tree:
  """Performs a full search and returns sampled actions.

  In the shape descriptions, `B` denotes the batch dimension.

  Args:
    params: params to be forwarded to root and recurrent functions.
    rng_key: random number generator state, the key is consumed.
    root: a `(prior_logits, value, embedding)` `RootFnOutput`. The
      `prior_logits` are from a policy network. The shapes are
      `([B, num_actions], [B], [B, ...])`, respectively.
    recurrent_fn: a callable to be called on the leaf nodes and unvisited
      actions retrieved by the simulation step, which takes as args
      `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
      and the new state embedding. The `rng_key` argument is consumed.
    root_action_selection_fn: function used to select an action at the root.
    interior_action_selection_fn: function used to select an action during
      simulation.
    num_simulations: the number of simulations.
    max_depth: maximum search tree depth allowed during simulation, defined as
      the number of edges from the root to a leaf node.
    invalid_actions: a mask with invalid actions at the root. In the
      mask, invalid actions have ones, and valid actions have zeros.
      Shape `[B, num_actions]`.
    extra_data: extra data passed to `tree.extra_data`. Shape `[B, ...]`.
    loop_fn: Function used to run the simulations. It may be required to pass
      hk.fori_loop if using this function inside a Haiku module.

  Returns:
    `SearchResults` containing outcomes of the search, e.g. `visit_counts`
    `[B, num_actions]`.
  """
  M = max_num_considered_actions

  # action_selection_fn = action_selection.switching_action_selection_wrapper(
  #     root_action_selection_fn=root_action_selection_fn,
  #     interior_action_selection_fn=interior_action_selection_fn
  # )

  # Do simulation, expansion, and backward steps.
  batch_size = root.value.shape[0]
  batch_range = jnp.arange(batch_size)
  if max_depth is None:
    max_depth = num_simulations
  if invalid_actions is None:
    invalid_actions = jnp.zeros_like(root.prior_logits)

  # --- 1.1 Simulate, Seq Halving Stuff, active explorers
  # Instead of root action selection, we parallel expand and fill in the tree
  # Each root explorer keeps track of its num_sims expanded thus far
  n_active_explorers_table = seq_halving.get_num_active_explorers_table(
      max_num_considered_actions, num_simulations) # [M+1, rounds (which is = nsims)]
  print("n_active_explorers_table shape", n_active_explorers_table.shape)
  # jax.debug.print("n_active_explorers_table: {}", n_active_explorers_table)

  # ---- init: make every bucket alive on round-0 ----------------------------
  num_legal_moves = jnp.sum(~invalid_actions, axis=-1)
  num_actions_considered = jnp.minimum(max_num_considered_actions, num_legal_moves)
  print("num_actions_considered shape", num_actions_considered.shape)
  # jax.debug.print("num_actions_considered: {}", num_actions_considered)

  k0 = n_active_explorers_table[num_actions_considered, 0]        # (B,)
  # jax.debug.print("k0 is: {}", k0)
  # active_mask0 = jnp.where(
  #     jnp.arange(M)[None, :] < k0[:, None],        # bool[B,M]
  #     0,                                           # first visit target = 0
  #     -1                                           # idle
  # ).astype(jnp.int32)
  active_mask0 = jnp.arange(M)[None, :] < k0[:, None] # bool[B,M], first k0 elems out of M are True
  # jax.debug.print("active_mask0 is: {}", active_mask0)

  # Allocate all necessary storage.
  tree = instantiate_tree_from_root(root, num_simulations,
                                    root_invalid_actions=invalid_actions,
                                    extra_data=extra_data)

  ### opt2
  # tree, total_sims, sampled_glogits, sampled_actions  = init_root_children(
  tree, total_sims, sampled_glogits, sampled_actions  = init_root_children_fast(
    params,
    rng_key,
    tree,
    root,
    recurrent_fn,
    max_num_considered_actions,
    invalid_actions,
    extra_data) # equiv. to simulate, expand, backwards for first layer

  # jax.debug.print("[legal] total_sims: {}, sampled_actions: {}", total_sims, sampled_actions)
  # jax.debug.print("[legal] total_sims: {}, node_visits: {}", total_sims, tree.node_visits)
  # jax.debug.print("[legal] total_sims: {}, children_index: {}", total_sims, tree.children_index)
  # search tree, total sims expanded, round index, rng

  init_carry = (tree, active_mask0, total_sims, 1, rng_key)
  # jax.debug.print("[search0] total sims2?: {}", total_sims)
  # jax.debug.print("[search0] sampled_actions: {}", sampled_actions)
  # total_sims = jnp.full_like(total_sims, fill_value=16)

  # ---------------------------------------------------------------------------
  # Determine the end round for all batches
  # 1. look up how many explorers are needed in every round
  n_active_per_batch = jnp.take_along_axis(
    n_active_explorers_table,               # [M+1,  R]
    num_actions_considered[:, None],        # [B,1] – per-game m
    axis=0                                  # gather on first axis
  )

  # 2. first round where this game needs **zero** explorers
  # jax.debug.print("n_active_per_batch: {}", n_active_per_batch)
  first_inactive_round = jnp.sum(n_active_per_batch != 0, axis=-1) # calc "is active per round", then calc position of first inactive round
  # jax.debug.print("first_inactive_round: {}", first_inactive_round)

  # 🆕 3. shorten the “only-one-legal-move” games
  # If it's 1 legal move, first_inactive_round == num_simulations,
  # and then, we set its inactive round to the earliest time we could terminate
  # (maybe the game w/ MAX_M that termiantes early)
  min_round = jnp.min(first_inactive_round)        # scalar
  first_inactive_round = jnp.where(
      first_inactive_round == num_simulations,     # the forced-move rows
      min_round,                                   # … stop when everyone else stops
      first_inactive_round)                        # … keep original value otherwise
  # ---------------------------------------------------------------------------


  def cond_fun(loop_state):
    tree, active_mask, sims, round, _rng_key = loop_state
    # [todo] what about cases where M is greater than sim?
    # all_inactive = ~jnp.all(active_mask == False)

    # finished = (round >= first_inactive_round)
    # n_finished = jnp.sum(round >= first_inactive_round)
    # all_finished = jnp.all(finished)
    # not_all_inactive = (~all_finished)
    # jax.debug.print("[528] round: {}, finished: {}, n_finished: {}, all_finished: {}, sims: {}", round, finished, n_finished, all_finished, sims)

    not_all_inactive = ~jnp.all(round >= first_inactive_round)
    return jnp.logical_and(not_all_inactive, round < num_simulations)
    # return ~jnp.all(sims >= num_simulations)
    # return round < 2

  ### opt3: [BK]
  def body_fun(loop_state):
      tree, active_mask, sim_cnt, round_i, rng_key = loop_state
      k_this = round_max_k[round_i]                # int32 scalar (tracer)
      # jax.debug.print("[bk] round_i: {}, k_this: {}", round_i, k_this)
      return jax.lax.switch(
          k_this, round_kernels, loop_state,
          # operands forwarded to every kernel
          params,
          sampled_actions,
          sampled_glogits,
          num_actions_considered,
          n_active_explorers_table,
          # value_scale,
          # maxvisit_init,
          # rescale_values,
          # use_mixed_value,
          # epsilon,
  )

  # [todo] reenable]
  tree, _active_mask, total_sims, _round, _rng = jax.lax.while_loop(cond_fun, body_fun, init_carry)
  # jax.debug.print("[search2-fin]new_sim_count: {}", total_sims)

  # ### og
  # _, tree = loop_fn(
  #     0, num_simulations, body_fun, (rng_key, tree))

  return tree


class _SimulationState(NamedTuple):
  """The state for the simulation while loop."""
  rng_key: chex.PRNGKey
  node_index: int
  action: int
  next_node_index: int
  depth: int
  is_continuing: bool

# ---------------------------------------------------------------------------
# Roll out ONE root-child explorer (vectorised over axis-0)
# ---------------------------------------------------------------------------
# VMAP layout:
#   rng_key            – 0
#   tree               – None  (shared across the M explorers)
#   is_active          – 0     (bool, True = run, False = skip)
#   top_root_actions   – 0     (used as default value for inactive root child paths
#   start_node_index   – 0
#   depth              – 0
#   action_selection_fn--None
#   max_depth--None
@functools.partial(
    jax.vmap,
    in_axes=(0, None, 0, 0, 0, 0, None, None),
    out_axes=(0, 0, 0))
def simulate_root_child(
    rng_key: chex.PRNGKey,
    tree: Tree,
    sampled_actions: chex.Array,       # int32[M]
    is_active: chex.Array,             # bool[M]
    explorer_node_index: chex.Array,   # int32[M]  (1 … M) idx of the root child
    depth: chex.Array,                 # int32[M]  (=1)
    interior_action_selection_fn: base.InteriorActionSelectionFn,
    max_depth: int
) -> Tuple[chex.Array, chex.Array]:
  """Idle explorers return (Tree.ROOT_INDEX, original action taken during init_root_children)."""

  INACTIVE_PARENT = jnp.asarray(Tree.ROOT_INDEX, jnp.int32)
  INACTIVE_ACTION = sampled_actions
  # NO_PARENT = jnp.asarray(Tree.NO_PARENT, jnp.int32)

  # -- initial state --------------------------------------------------------
  init_state = _SimulationState(
      rng_key          = rng_key,
      node_index       = INACTIVE_PARENT,          # sentinel
      action           = INACTIVE_ACTION,          # sentinel
      next_node_index  = explorer_node_index,         # 1…M (or anything)
      depth            = depth,
      is_continuing    = is_active)

  # -- body of the MuZero roll-out -----------------------------------------
  def cond_fun(state):
    return state.is_continuing

  def body_fun(state):
    rng_key, sk          = jax.random.split(state.rng_key)

    cur_node             = state.next_node_index
    action               = interior_action_selection_fn(sk, tree, cur_node, state.depth)
    next_node_idx        = tree.children_index[cur_node, action]
    d                    = state.depth + 1
    cont                 = jnp.logical_and(d < max_depth,
                                           next_node_idx != Tree.UNVISITED)
    return _SimulationState(rng_key, cur_node, action, next_node_idx, d, cont)

  end_state = jax.lax.while_loop(cond_fun, body_fun, init_state)
  return end_state.node_index, end_state.action, end_state.next_node_index

# -- Parallel simulator --------------------------------------------------------
#
#  • active_explorer_mask[b, m] == -1  →  explorer m is idle in game b
#  • active_explorer_mask[b, m] >=  0  →  explorer m is live and should advance
#
#  We run an independent tree-traversal for *every* root-child explorer
#  (there are `top_m == M` of them).  Idle explorers just emit NO_PARENT.
#
@functools.partial(
    jax.vmap,                       # batch-vectorise over games
    in_axes=(0, 0, 0, 0, 0, 0, None, None, None),   # rng, tree, sims_done, mask are batched
    out_axes=(0, 0, 0))                    # outputs → [B, M]
def simulate2(
    rng_key: chex.PRNGKey,
    tree: Tree,
    k_starting_node_idxs: chex.Array,   # (B, K) after vmap => (K)
    sampled_actions: chex.Array,        # (B, M) after vmap => (M)
    sims_done: chex.Array,              # (B,)  – unused here but kept for API
    active_explorer_mask: chex.Array,   # (B, M) after vmap ⇒ (M) here
    # nonbatched
    interior_action_selection_fn: base.InteriorActionSelectionFn,
    max_depth: int,
    max_num_considered_actions: int
) -> Tuple[chex.Array, chex.Array, chex.Array]:
  """
  Runs one simulation for every *active* root-child explorer.

  Returns
  -------
  parent_index        : int32[ M ]   – parent node where rollout stopped, Tree.NO_PARENT for idle explorer
  action              : int32[ M ]   – action to expand from that parent, sampled_action for idle explorer
  next_node_idx       : int32[ M ]   – action to expand from that parent, explorer's node idx for idle explorer
  """

  top_m = max_num_considered_actions

  # -- 2. Boolean mask of live explorers -------------------------------------
  active            = active_explorer_mask                 # (M,)

  # -- 3. Prepare per-explorer start nodes, depths & keys --------------------
  # [bk-parent-bug]
  # explorer_node_idxs       = jnp.arange(1, top_m + 1, dtype=jnp.int32)  # node 1…M, always the same
  explorer_node_idxs       = k_starting_node_idxs

  depths                   = jnp.ones((top_m,), dtype=jnp.int32)        # depth = 1
  subkeys                  = jax.random.split(rng_key, top_m)           # (M,)

  # -- 4. Roll out every explorer in parallel -------------------------------
  #     simulate_root_child is already vmapped over axis 0.
  parent_idx, act, next_node_idx = simulate_root_child(
      subkeys,
      tree,
      sampled_actions,
      active,           # <- pass boolean mask third
      explorer_node_idxs,
      depths,
      interior_action_selection_fn,
      max_depth)



  # jax.debug.print("[parentz] simulate2:: {}", parent_idx)


  # -- 5. Mask-out idle explorers -------------------------------------------
  # Idle explorer's default values are set in simulate_root_child

  return parent_idx.astype(jnp.int32), act.astype(jnp.int32), next_node_idx.astype(jnp.int32)

def expand3(
    params: chex.Array,
    rng_key: chex.PRNGKey,
    tree: Tree[T],
    active_mask: chex.Array,      # [B, M]
    recurrent_fn: base.RecurrentFn,
    parent_idxs: chex.Array,      # [B, M]
    actions: chex.Array,          # [B, M]
    next_node_idxs: chex.Array,   # [B, M]
) -> Tree[T]:
  """Expand *all* (B×M) <parent, action> pairs in parallel.

  • `parent_idxs[b, m]`   – parent node to expand from
  • `actions[b, m]`       – action taken from that parent
  • `next_node_idxs[b, m]`– fresh node id (if UNVISITED) *or*
                            the already-existing child id

  All three tensors have exactly the same static shape `[B, M]`
  (there is **no** branch on “active vs. inactive” inside this routine).
  """
  B, M          = parent_idxs.shape
  K             = B * M                              # total leaves
  A             = tree.children_index.shape[-1]      # #actions
  N             = tree.children_index.shape[1]       # #rows (nodes)

  chex.assert_shape([parent_idxs, actions, next_node_idxs], (B,M))
  # jax.debug.print("[parentz] expand3:: {}", parent_idxs)
  # jax.debug.print("[parentz] next_node_idxs:: {}", next_node_idxs)

  # ------------------------------------------------------------------ #
  # 1. gather the embeddings of **all** parents in one shot            #
  # batch_flat looks like
  # > jnp.repeat(jnp.arange(2), 4)
  # > Array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int32)
  # ------------------------------------------------------------------ #
  batch_flat    = jnp.repeat(jnp.arange(B), M)       # [K]
  parent_flat   = parent_idxs.reshape(-1)            # [K]
  ### opt1
  # emb_flat      = jax.tree_map(
  #     lambda x: x[batch_flat, parent_flat],          # -> [K, …]
  #     tree.embeddings)

  ### opt2
  def fast_parent_gather(arr, batch_idx, row_idx):
    """
    arr        : [B, N, …]        any dtype / rank ≥ 2
    batch_idx  : [K]              batch  (0…B-1)  for K leaves
    row_idx    : [K]              row    (0…N-1)
    returns    : [K, …]           gathered rows
    """
    B, N = arr.shape[:2]

    # ❶ flatten the first two axes once
    arr_flat = arr.reshape((B * N,) + arr.shape[2:])      # [B*N, …]

    # ❷ linearise the pair (b, n) → b*N + n
    flat_idx = batch_idx * N + row_idx                    # [K]

    # ❸ 1-D take – this lowers to a simple gather
    return arr_flat.take(flat_idx, axis=0)                # [K, …]

  emb_flat = jax.tree_map(
      lambda x: fast_parent_gather(x, batch_flat, parent_flat),
      tree.embeddings)

  # ### opt3
  # def fast_parent_gather_dot(arr, batch_idx, row_idx):
  #   """
  #   Pure-arithmetic replacement for taking rows (batch,row) from `arr`.

  #     arr        : [B, N, …]
  #     batch_idx  : [K]  int32   (0 … B-1)
  #     row_idx    : [K]  int32   (0 … N-1)
  #     returns    : [K, …]       same dtype as arr
  #   """
  #   dtype = arr.dtype
  #   B, N  = arr.shape[:2]

  #   # One-hot masks  – keep them in the *same* dtype as arr (bf16/f32 → fast)
  #   bmask = jax.nn.one_hot(batch_idx, B, dtype=dtype)      # [K, B]
  #   rmask = jax.nn.one_hot(row_idx,  N, dtype=dtype)       # [K, N]

  #   # Outer-product → [K, B, N]
  #   mask  = bmask[:, :, None] * rmask[:, None, :]

  #   # Contract the (B,N) axes against arr
  #   # ──────────────────────────────────────────────────────────
  #   #   mask[k, b, n] · arr[b, n, …]  → out[k, …]
  #   # ──────────────────────────────────────────────────────────
  #   return jnp.einsum('kbn,bn...->k...', mask, arr)

  # emb_flat = jax.tree_map(
  #   lambda x: fast_parent_gather_dot(x, batch_flat, parent_flat),
  #   tree.embeddings)

  # ### [opt4]
  # def fast_parent_gather_matmul(arr: jnp.ndarray,
  #                             batch_idx: jnp.ndarray,   # [K]
  #                             row_idx: jnp.ndarray      # [K]
  #                            ) -> jnp.ndarray:
  #   """
  #   Scatter-free gather that replaces

  #       arr.reshape(B*N, …).take(batch_idx*N + row_idx, axis=0)

  #   with a dot_general against a one-hot selector.

  #   • Works for any trailing feature shape.
  #   • dtype of the selector = dtype of `arr`  → no extra cast.
  #   • Pure mul+add ⇒ fuses into one GEMM on TPU / GPU.
  #   """
  #   dtype = arr.dtype          # keep bf16/f32 throughput
  #   B, N  = arr.shape[:2]
  #   K     = batch_idx.shape[0]

  #   # 1) flatten the table along (B,N)
  #   flat  = arr.reshape((B * N,) + arr.shape[2:])        # [B*N, …]

  #   # 2) build a (K, B*N) one-hot selector S
  #   flat_idx = batch_idx * N + row_idx                   # [K]
  #   # iota     = jax.lax.broadcasted_iota(jnp.int32, (B * N,)) # (B*N,)
  #   # selector = (flat_idx[:, None] == iota[None, :]).astype(dtype)  # [K,B*N]

  #   selector = jax.nn.one_hot(flat_idx, B * N, dtype=dtype)

  #   # 3) contract selector with the 0-axis of `flat`
  #   #    result has shape   [K, …]
  #   out = jax.lax.dot_general(
  #           selector,          # lhs  [K, B*N]
  #           flat,              # rhs  [B*N, …]
  #           (((1,), (0,)),     # contract selector.col with flat.row
  #            ((), ()))         # no batched dims
  #         )
  #   return out

  # emb_flat = jax.tree_map(
  #   lambda x: fast_parent_gather_matmul(x, batch_flat, parent_flat),
  #   tree.embeddings)



  # RNG for every leaf
  rng_keys      = jax.random.split(rng_key, K)

  # model inference on all leaves
  action_flat   = actions.reshape(-1)
  step_flat, emb_new_flat = recurrent_fn(
      params, rng_keys, action_flat, emb_flat)

  # reshape back to [B, M, …] for convenience
  step      = jax.tree_map(lambda t: t.reshape(B, M, *t.shape[1:]), step_flat)
  emb_new   = jax.tree_map(lambda t: t.reshape(B, M, *t.shape[1:]), emb_new_flat)

  # conveniences
  dtype_idx = tree.children_index.dtype
  dtype_flt = tree.children_rewards.dtype   # rewards / discounts share dtype

  # # ------------------------------------------------------------------ #
  # # 2. rank-3 tables: children_{index|rewards|discounts}                #
  # # ------------------------------------------------------------------ #
  m_row  = jax.nn.one_hot(parent_idxs, N, dtype=dtype_flt)          # [B,M,N]
  m_col  = jax.nn.one_hot(actions,     A, dtype=dtype_flt)          # [B,M,A]
  valid  = active_mask.astype(dtype_flt)[..., None, None]           # [B,M,1,1]
  # (B,M,N,A) boolean mask that marks exactly the cells we want to write
  mask_3d = m_row[..., None] * m_col[..., None, :] * valid          # [B,M,N,A]

  write_mask = jnp.minimum(mask_3d.sum(1), 1)                       # [B,N,A]  0/1

  new_idx  = (mask_3d * next_node_idxs[..., None, None]).sum(1).astype(dtype_idx)
  new_rew  = (mask_3d * step.reward   [..., None, None]).sum(1).astype(dtype_flt)
  new_disc = (mask_3d * step.discount [..., None, None]).sum(1).astype(dtype_flt)

  tree = tree.replace(
      children_index     = tree.children_index * (1-write_mask.astype(dtype_idx)) + new_idx,
      children_rewards   = tree.children_rewards * (1-write_mask) + new_rew,
      children_discounts = tree.children_discounts * (1-write_mask) + new_disc,
  )

  # ------------------------------------------------------------------ #
  # 3. rank-2 tables: parents, action_from_parent                      #
  # ------------------------------------------------------------------ #
  tree = tree.replace(
      parents            = set_rows_anydtype_masked(tree.parents,
                                                    next_node_idxs,
                                                    parent_idxs,
                                                    active_mask),
      action_from_parent = set_rows_anydtype_masked(tree.action_from_parent,
                                                    next_node_idxs,
                                                    actions,
                                                    active_mask),
  )

  # ------------------------------------------------------------------ #
  # 4. node statistics & embeddings (also rank-2 / rank-≥3 updates)    #
  # ------------------------------------------------------------------ #
  tree = update_tree_node_multi(
    tree,
    next_node_idxs,
    step.prior_logits,   # [B,M,A]
    step.value,          # [B,M]
    emb_new,             # pytree [B,M,…]
    active_mask)         # bool [B,M]


  return tree

@jax.vmap                               #   ← batch axis B
def backward_batch(tree: Tree,
                   leaf_indices:  jnp.ndarray,   # [M]
                   active_mask:   jnp.ndarray,   # [M] bool
                   root_child_ids:jnp.ndarray    # [M] (1…M)
                  ) -> Tree:
  """
  • Runs `backward_explorer` on every root-child explorer (axis M).
  • Sums all returned delta_trees → unique-node guarantee means
    a straight sum is correct.
  • Applies the summed delta_tree to `tree` with pure addition.
  """

  # jax.debug.print("[backwards] leaf_indices: {}", leaf_indices)
  # jax.debug.print("[backwards] active_mask: {}", active_mask)
  # jax.debug.print("[backwards] root_child_ids: {}", root_child_ids)
  # (M, …) pytree of deltas
  delta_trees = backward_explorer(
      tree,
      leaf_indices,          # (M,)
      active_mask,           # (M,)
      root_child_ids)        # (M,)

  print("[deltatree] shape", delta_trees.node_visits.shape)

  # reduce explorer axis with sum(0)
  # delta_sum = jax.tree_util.tree_map(lambda x: x.sum(axis=0), delta_trees)
  delta_sum = delta_trees

  # add deltas — dtype-safe
  updated_tree = jax.tree_util.tree_map(
      lambda old, d: old + d.astype(old.dtype),
      tree, delta_sum)

  return updated_tree


def _zeros_like_tree(t: Tree) -> Tree:
  """Return a Tree-shaped pytree whose leaves are zeros_like the input."""
  return jax.tree_util.tree_map(jnp.zeros_like, t)

# def backward_explorer(tree: Tree,
#                       leaf_idx_vec:  jnp.ndarray,   # (M,)
#                       active_mask:   jnp.ndarray,   # (M,)  bool
#                       root_child_ids:jnp.ndarray    # (M,)  int32
#                      ) -> Tree:
#   """
#   • Handles *all* M root-child explorers in one shot (no vmap here).
#   • Returns a `delta_tree` whose leaves contain only increments / diffs.
#   • Explorers with `active_mask == False` never add anything.
#   """
#   M, N, A = (leaf_idx_vec.shape[0],
#              tree.node_values.shape[0],
#              tree.children_values.shape[-1])

#   delta_tree = _zeros_like_tree(tree)                 # start with zeros

#   # ---------- while-loop ----------------------------------------------------
#   def cond_fun(state):
#     loop_i, idx_vec, *_ = state
#     still_climbing = jnp.logical_and(idx_vec != root_child_ids, idx_vec != Tree.NO_PARENT)
#     any_climbing = jnp.any(jnp.logical_and(active_mask, still_climbing(idx_vec)))
#     jax.debug.print("[backwardz] @{} still_climbing: {}", loop_i, still_climbing)
#     return any_climbing

#   def body_fun(state):
#     loop_i, idx_vec, leaf_val_vec, dt = state

#     # parents / actions for every explorer ------------------------------
#     par_vec = tree.parents[idx_vec]              # (M,)
#     act_vec = tree.action_from_parent[idx_vec]   # (M,)

#     # -------- mask of *currently* active explorers ---------------------
#     live = jnp.logical_and(active_mask, still_climbing(idx_vec))  # (M,)

#     # -------- one Euler step of the Bellman backup ---------------------
#     reward   = tree.children_rewards  [par_vec, act_vec]
#     discount = tree.children_discounts[par_vec, act_vec]
#     leaf_val_vec = reward + discount * leaf_val_vec               # (M,)

#     old_cnt  = tree.node_visits[par_vec]
#     new_cnt  = old_cnt + 1
#     old_val  = tree.node_values[par_vec]
#     new_val  = (old_val * old_cnt + leaf_val_vec) / new_cnt       # (M,)

#     # -------- build *masked* increments -------------------------------
#     # helpers: turn scatter → add-at
#     def add_at(arr, idx, val):
#       return arr.at[idx].add(val)

#     # children_* are rank-3; build flat indices first
#     child_vis_delta = dt.children_visits
#     child_val_delta = dt.children_values
#     node_vis_delta  = dt.node_visits
#     node_val_delta  = dt.node_values

#     # only the live explorers contribute
#     par_live, act_live = par_vec[live], act_vec[live]
#     child_vis_delta = child_vis_delta.at[par_live, act_live].add(1)
#     child_val_delta = child_val_delta.at[par_live, act_live].add(
#         tree.node_values[idx_vec[live]] -
#         tree.children_values[par_live, act_live])

#     node_vis_delta  = node_vis_delta.at[par_live].add(1)
#     node_val_delta  = node_val_delta.at[par_live].add(
#         new_val[live] - old_val[live])

#     dt = dt.replace(children_visits = child_vis_delta,
#                     children_values = child_val_delta,
#                     node_visits     = node_vis_delta,
#                     node_values     = node_val_delta)

#     # next indices: live explorers move up; others stay put
#     next_idx_vec = jnp.where(live, par_vec, idx_vec)

#     # ---------- debug print once per loop ------------------------------
#     jax.debug.print("[backward] iter={}  live={}  max_idx={}",
#                     loop_i, live, idx_vec.max())

#     return (loop_i + 1, next_idx_vec, leaf_val_vec, dt)

#   # initial state -----------------------------------------------------------
#   init_state = (0,                      # loop counter
#                 leaf_idx_vec,           # (M,)
#                 tree.node_values[leaf_idx_vec],   # (M,)
#                 delta_tree)

#   *_unused, delta_tree = jax.lax.while_loop(cond_fun, body_fun, init_state)
#   return delta_tree

def backward_explorer(tree: Tree,
                      leaf_idx_vec:  jnp.ndarray,   # (M,)
                      active_mask:   jnp.ndarray,   # (M,) bool
                      root_child_ids:jnp.ndarray    # (M,) int32
                     ) -> Tree:
  """
  Walks all M root–child explorers upward in parallel and returns a
  Tree-shaped pytree whose leaves hold **increments** (deltas).
  Inactive explorers (active_mask == False) contribute 0 everywhere.
  """
  delta = _zeros_like_tree(tree)                 # start with zeros

  # ------------------------------------------------------------------ #
  # helpers                                                             #
  # ------------------------------------------------------------------ #
  def still_climbing(idx):                       # (M,) → (M,) bool
    return jnp.logical_and(idx != root_child_ids,
                           idx != Tree.NO_PARENT)

  def any_live(idx):
    return jnp.any(jnp.logical_and(active_mask, still_climbing(idx)))

  # ------------------------------------------------------------------ #
  # while-loop                                                         #
  # ------------------------------------------------------------------ #
  def cond_fun(state):
    loop_i, idx_vec, *_ = state
    is_still_climbing = jnp.logical_and(idx_vec != root_child_ids, idx_vec != Tree.NO_PARENT)
    # jax.debug.print("[backwardz3] @{} active_mask: {}", loop_i, active_mask)
    # jax.debug.print("[backwardz3] @{} is_still_climbing: {}", loop_i, is_still_climbing)
    is_any_climbing = jnp.any(jnp.logical_and(active_mask, is_still_climbing))
    # jax.debug.print("[backwardz3] @{} is_any_climbing: {}", loop_i, is_any_climbing)
    return is_any_climbing

  def body_fun(state):
    # Initially, the prev_node_val should be the recently expanded node value, or the existing leaf node value (maybe)
    #
    # leaf_val_vec is the new leaf qvalue (always the same), but backproped (discounted by -1) by the last iter
    # so it'll be leaf_val, then -leaf_val, then leaf_val
    loop_i, idx_vec, leaf_val_vec, prev_node_val, d = state     # “d” is the delta tree

    # jax.debug.print("[backwardexplorer] idx_vec: {}", idx_vec)


    # parents / actions of *every* explorer
    ### opt1
    par = tree.parents[idx_vec]                  # (M,)
    act = tree.action_from_parent[idx_vec]       # (M,)

    # ### opt2
    # par = fast_gather_1d(tree.parents,             idx_vec)  # [M]
    # act = fast_gather_1d(tree.action_from_parent,  idx_vec)  # [M]




    # jax.debug.print("[backwardz] @{} par: {}", loop_i, par)
    # jax.debug.print("[backwardz] @{} act: {}", loop_i, act)

    # which explorers are doing work this step?
    live = jnp.logical_and(active_mask, still_climbing(idx_vec))  # (M,) bool
    live_f = live.astype(tree.node_visits.dtype)                  # 0/1
    # jax.debug.print("[backwardz] @{} live: {}", loop_i, live)

    # ----- Bellman backup -------------------------------------------
    ### opt1
    reward   = tree.children_rewards  [par, act]
    discount = tree.children_discounts[par, act]

    # ## opt2
    # reward   = fast_gather_child2d(tree.children_rewards,   par, act)
    # discount = fast_gather_child2d(tree.children_discounts, par, act)

    # ### opt3
    # reward = gather_NA(tree.children_rewards, par, act)
    # discount = gather_NA(tree.children_discounts, par, act)


    leaf_val_vec = jnp.where(live,              # only live explorers update
                             reward + discount * leaf_val_vec,
                             leaf_val_vec)

    ###### calc new parent val
    ### opt1:
    old_parent_cnt = tree.node_visits[par]
    old_parent_val = tree.node_values[par]

    # ### opt2:
    # old_parent_cnt = fast_gather_1d(tree.node_visits, par)          # int32
    # old_parent_val = fast_gather_1d(tree.node_values, par)          # float

    new_parent_cnt = old_parent_cnt + live_f                  # add 1 where live == 1
    new_parent_val = jnp.where(
        live,
        (old_parent_val * old_parent_cnt + leaf_val_vec) / jnp.maximum(new_parent_cnt, 1),
        old_parent_val)                                # keep old_val if not live

    # # [opt1] ---------- accumulate deltas with scatter-add -------------------
    # d = d.replace(
    #     children_visits = d.children_visits.at[par, act].add(live_f),
    #     children_values = d.children_values.at[par, act].add(
    #         # live_f * (tree.node_values[idx_vec] -
    #         #           tree.children_values[par, act])),
    #         live_f * (prev_node_val - tree.children_values[par, act])),
    #     node_visits     = d.node_visits.at[par].add(live_f),
    #     node_values     = d.node_values.at[par].add(live_f * (new_parent_val - old_parent_val))
    # )

    # [opt2] ---- helper for scatter_add equivalent --------------------------------
    def _add_children(tab, row_idx, col_idx, inc):
      """
      tab      : [N, A]   int/float
      row_idx  : [M]      int32   (parent rows)
      col_idx  : [M]      int32   (action cols)
      inc      : [M]      same dtype as tab
      returns  : tab + Σ_inc  (scatter-free)
      """
      N, A = tab.shape
      rmask = jax.nn.one_hot(row_idx, N, dtype=tab.dtype)        # [M,N]
      cmask = jax.nn.one_hot(col_idx, A, dtype=tab.dtype)        # [M,A]
      # outer product → [M,N,A] then sum along explorer axis
      delta = (rmask[:, :, None] * cmask[:, None, :] * inc[:, None, None]).sum(0)
      return tab + delta


    def _add_nodes(tab, row_idx, inc):
      """
      tab      : [N]      int/float
      row_idx  : [M]      int32
      inc      : [M]      same dtype
      """
      N  = tab.shape[0]
      rmask = jax.nn.one_hot(row_idx, N, dtype=tab.dtype)        # [M,N]
      delta = (rmask * inc[:, None]).sum(0)                      # [N]
      return tab + delta

    # ---- build the per-explorer increments --------------------------------
    ### opt1:
    old_children_values = tree.children_values[par, act]
    # ### opt2:
    # old_children_values = fast_gather_child2d(tree.children_values, par, act)

    child_vis_inc = live_f                         # [M] (0/1)
    child_val_inc = live_f * (prev_node_val        # [M]
                              - old_children_values)
    node_vis_inc  = live_f                         # [M]
    node_val_inc  = live_f * (new_parent_val
                              - old_parent_val)

    # ---- scatter-free accumulation into the delta-tree -------------------
    d = d.replace(
        children_visits = _add_children(d.children_visits,
                                        par, act, child_vis_inc),
        children_values = _add_children(d.children_values,
                                        par, act, child_val_inc),
        node_visits     = _add_nodes(d.node_visits,
                                    par, node_vis_inc),
        node_values     = _add_nodes(d.node_values,
                                    par, node_val_inc)
    )








    # ---------- next indices (live explorers move up) ----------------
    next_idx_vec = jnp.where(live, par, idx_vec)

    # optional tracing
    # jax.debug.print("[bw] iter={} live={} max_idx={}",
    #                 loop_i, live, idx_vec.max())

    return (loop_i + 1, next_idx_vec, leaf_val_vec, new_parent_val, d)

  leaf_node_values = tree.node_values[leaf_idx_vec]
  init_state = (0, leaf_idx_vec, leaf_node_values, leaf_node_values, delta)

  # *_unused, delta_out = jax.lax.while_loop(cond_fun, body_fun, init_state)
  loop_i, idx_vec_fin, leaf_val_vec, explorer_new_node_val, delta_out = jax.lax.while_loop(
      cond_fun, body_fun, init_state)

  # Update the root
  num_active_explorers = jnp.sum(active_mask)
  root_actions  = tree.action_from_parent[root_child_ids]          # (M,)
  # active_explorers_values = jnp.sum(tree.node_values[root_child_ids] * active_mask) # [M] -> ()

  # ----- Bellman backup -------------------------------------------
  reward   = tree.children_rewards  [Tree.ROOT_INDEX, root_actions]
  discount = tree.children_discounts[Tree.ROOT_INDEX, root_actions]
  root_leaf_explorer_val = jnp.where(active_mask, reward + discount * leaf_val_vec, 0)

  # jax.debug.print("leaf_idx_vec: {}, active_mask: {}, explorer_new_node_val: {}", leaf_idx_vec, active_mask, explorer_new_node_val)
  active_explorers_leaf_contributions = jnp.sum(root_leaf_explorer_val) # [M] -> ()
  # jax.debug.print("leaf_idx_vec: {}, active_mask: {}, active_explorers_leaf_contributions: {}", leaf_idx_vec, active_mask, active_explorers_leaf_contributions)
  root_node_value = tree.node_values[Tree.ROOT_INDEX]

  # jax.debug.print("leaf_idx_vec: {}, active_mask: {}, root_node_value: {}", leaf_idx_vec, active_mask, root_node_value)
  # jax.debug.print("leaf_idx_vec: {}, active_mask: {}, num_active_explorers: {}", leaf_idx_vec, active_mask, num_active_explorers)
  root_node_visits = tree.node_visits[Tree.ROOT_INDEX]
  # jax.debug.print("leaf_idx_vec: {}, active_mask: {}, root_node_visits: {}", leaf_idx_vec, active_mask, root_node_visits)
  new_root_value = (root_node_value * root_node_visits + active_explorers_leaf_contributions) / (root_node_visits + num_active_explorers)
  # jax.debug.print("leaf_idx_vec: {}, active_mask: {}, new_root_value: {}", leaf_idx_vec, active_mask, new_root_value)

  # increments for children_visits[root, action]
  visit_inc     = active_mask.astype(delta_out.children_visits.dtype)  # (M,)

  # child-value increment = same increment that went into node_values (don't grab from tree.node_values / children_values cuz stale)
  value_inc       = delta_out.node_values[root_child_ids]



  ### opt1


  # delta_out = delta_out.replace(
  #     node_visits     = delta_out.node_visits.at[Tree.ROOT_INDEX].add(num_active_explorers),
  #     node_values     = delta_out.node_values.at[Tree.ROOT_INDEX].add(new_root_value - root_node_value),
  #     # children_visits = delta_out.children_visits.at[Tree.ROOT_INDEX, ..., # fill in
  #     # children_values = delta_out.children_values.at[Tree.ROOT_INDEX, ...].add() # fill in
  #     # per-action tables in the root row
  #     children_visits = delta_out.children_visits.at[Tree.ROOT_INDEX,
  #                                                    root_actions].add(visit_inc),
  #     children_values = delta_out.children_values.at[Tree.ROOT_INDEX,
  #                                                    root_actions].add(value_inc),
  # )


  ### opt2
  # ---- 1. scalar columns in the root row --------------------------------
  root_mask = jax.nn.one_hot(Tree.ROOT_INDEX, delta_out.node_visits.shape[0],
                            dtype=delta_out.node_visits.dtype)        # [N] one-hot
  delta_out = delta_out.replace(
      node_visits = delta_out.node_visits + root_mask * num_active_explorers,
      node_values = delta_out.node_values + root_mask * (new_root_value - root_node_value),
  )

  # ---- 2. per-action tables (rank-2: [N, A]) ----------------------------
  # Build a vector that already contains **all** increments for row 0
  #   children_visits_inc[a] = Σ visit_inc[m] if root_actions[m] == a
  A     = delta_out.children_visits.shape[-1]   # #actions
  col_mask      = jax.nn.one_hot(root_actions, A,
                                dtype=delta_out.children_visits.dtype)    # [M, A]
  vis_row_inc   = jnp.sum(col_mask * visit_inc[:, None], axis=0)           # [A]
  val_row_inc   = jnp.sum(col_mask * value_inc[:, None], axis=0)           # [A]

  # Add those vectors to row 0 with the helper -- pure add/mul, no scatter
  delta_out = delta_out.replace(
      children_visits = add_row_vec_2d(delta_out.children_visits,
                                      Tree.ROOT_INDEX,
                                      vis_row_inc),
      children_values = add_row_vec_2d(delta_out.children_values,
                                      Tree.ROOT_INDEX,
                                      val_row_inc),
  )


  return delta_out

@jax.vmap
def backward2(
    tree: Tree[T],
    leaf_index: chex.Numeric) -> Tree[T]:
  """Goes up and updates the tree until all nodes reached the root.

  Args:
    tree: the MCTS tree state to update, without the batch size.
    leaf_index: the node index from which to do the backward.

  Returns:
    Updated MCTS tree state.
  """

  def cond_fun(loop_state):
    _, _, index = loop_state
    return index != Tree.ROOT_INDEX

  def body_fun(loop_state):
    # Here we update the value of our parent, so we start by reversing.
    tree, leaf_value, index = loop_state
    parent = tree.parents[index]
    count = tree.node_visits[parent]
    action = tree.action_from_parent[index]
    reward = tree.children_rewards[parent, action]
    leaf_value = reward + tree.children_discounts[parent, action] * leaf_value
    parent_value = (
        tree.node_values[parent] * count + leaf_value) / (count + 1.0)
    children_values = tree.node_values[index]
    children_counts = tree.children_visits[parent, action] + 1

    tree = tree.replace(
        # rank‑3 scalar updates
        children_visits  = set_cell(tree.children_visits,
                                    parent, action, children_counts),
        children_values  = set_cell(tree.children_values,
                                    parent, action, children_values),

        # rank‑2 updates
        node_values = set_row(tree.node_values, parent, parent_value),
        node_visits = set_row(tree.node_visits, parent, count + 1),
    )

    # tree = tree.replace(
    #     node_values=update(tree.node_values, parent_value, parent),
    #     node_visits=update(tree.node_visits, count + 1, parent),
    #     children_values=update(
    #         tree.children_values, children_values, parent, action),
    #     children_visits=update(
    #         tree.children_visits, children_counts, parent, action))

    return tree, leaf_value, parent

  leaf_index = jnp.asarray(leaf_index, dtype=jnp.int32)
  loop_state = (tree, tree.node_values[leaf_index], leaf_index)
  tree, _, _ = jax.lax.while_loop(cond_fun, body_fun, loop_state)

  return tree


# Utility function to set the values of certain indices to prescribed values.
# This is vmapped to operate seamlessly on batches.
def update(x, vals, *indices):
  return x.at[indices].set(vals)


batch_update = jax.vmap(update)


def update_tree_node(
    tree: Tree[T],
    node_index: chex.Array,
    prior_logits: chex.Array,
    value: chex.Array,
    embedding: chex.Array) -> Tree[T]:
  """Updates the tree at node index.

  Args:
    tree: `Tree` to whose node is to be updated.
    node_index: the index of the expanded node. Shape `[B]`.
    prior_logits: the prior logits to fill in for the new node, of shape
      `[B, num_actions]`.
    value: the value to fill in for the new node. Shape `[B]`.
    embedding: the state embeddings for the node. Shape `[B, ...]`.

  Returns:
    The new tree with updated nodes.
  """
  batch_size = tree_lib.infer_batch_size(tree)
  batch_range = jnp.arange(batch_size)
  chex.assert_shape(prior_logits, (batch_size, tree.num_actions))

  # When using max_depth, a leaf can be expanded multiple times.
  new_visit = tree.node_visits[batch_range, node_index] + 1

  # ### [opt1]
  updates = dict(  # pylint: disable=use-dict-literal
      children_prior_logits=batch_update(
          tree.children_prior_logits, prior_logits, node_index),
      raw_values=batch_update(
          tree.raw_values, value, node_index),
      node_values=batch_update(
          tree.node_values, value, node_index),
      node_visits=batch_update(
          tree.node_visits, new_visit, node_index),
      embeddings=jax.tree.map(
          lambda t, s: batch_update(t, s, node_index),
          tree.embeddings, embedding))

  ### [opt2]
  # updates = dict(  # pylint: disable=use-dict-literal
  #     children_prior_logits = add_row_vec(
  #               tree.children_prior_logits, node_index, prior_logits),
  #     ### rank‑3 row‑vector
  #     # children_prior_logits = set_row_vec(tree.children_prior_logits,
  #     #                                       node_index, prior_logits),
  #     ### og
  #     # children_prior_logits=batch_update(
  #     #     tree.children_prior_logits, prior_logits, node_index),
  #     raw_values=set_row(tree.raw_values, node_index, value),
  #     node_values=set_row(tree.node_values, node_index, value),
  #     node_visits=set_row(tree.node_visits, node_index, new_visit),
  #     embeddings = jax.tree.map(
  #           lambda t, s: set_row_any_sparse(t, node_index, s),
  #           tree.embeddings, embedding),
  # )

  return tree.replace(**updates)


def update_tree_node_masked(
    tree: Tree[T],
    node_index: chex.Array,
    prior_logits: chex.Array,
    value: chex.Array,
    embedding: chex.Array,
    is_dummy_op: chex.Array) -> Tree[T]:
  """Updates the tree at node index.

  Args:
    tree: `Tree` to whose node is to be updated.
    node_index: the index of the expanded node. Shape `[B]`.
    prior_logits: the prior logits to fill in for the new node, of shape
      `[B, num_actions]`.
    value: the value to fill in for the new node. Shape `[B]`.
    embedding: the state embeddings for the node. Shape `[B, ...]`.

  Returns:
    The new tree with updated nodes.
  """
  batch_size = tree_lib.infer_batch_size(tree)
  batch_range = jnp.arange(batch_size)
  chex.assert_shape(prior_logits, (batch_size, tree.num_actions))

  # When using max_depth, a leaf can be expanded multiple times.
  inc   = (~is_dummy_op).astype(tree.node_visits.dtype)      # [B,] 0/1
  new_visit = tree.node_visits[batch_range, node_index] + inc

  # ### [opt1]
  updates = dict(  # pylint: disable=use-dict-literal
      children_prior_logits=batch_update(
          tree.children_prior_logits, prior_logits, node_index),
      raw_values=batch_update(
          tree.raw_values, value, node_index),
      node_values=batch_update(
          tree.node_values, value, node_index),
      node_visits=batch_update(
          tree.node_visits, new_visit, node_index),
      embeddings=jax.tree.map(
          lambda t, s: batch_update(t, s, node_index),
          tree.embeddings, embedding))

  return tree.replace(**updates)

def update_tree_node_multi(
    tree: Tree[T],
    node_indices:    jnp.ndarray,   # [B, K]
    prior_logits:    jnp.ndarray,   # [B, K, A]
    values:          jnp.ndarray,   # [B, K]
    new_embeds:      Any,           # pytree with leaves [B, K, …]
    active_mask:     jnp.ndarray,   # [B, K]  bool   ← NEW!
) -> Tree[T]:
  """Masked scatter-free update for K leaf nodes per batch element."""
  B, K = node_indices.shape
  A = tree.num_actions

  # --- children_prior_logits ----------------------------------------------
  cpl = set_rows_anydtype_masked(tree.children_prior_logits,
                                 node_indices,
                                 prior_logits,
                                 active_mask)

  # --- raw_values / node_values -------------------------------------------
  rv = set_rows_anydtype_masked(tree.raw_values,
                                node_indices,
                                values,
                                active_mask)
  nv = set_rows_anydtype_masked(tree.node_values,
                                node_indices,
                                values,
                                active_mask)

  # --- node_visits (increment only when active) ---------------------------
  inc   = active_mask.astype(tree.node_visits.dtype)      # [B,K] 0/1
  old   = jnp.take_along_axis(tree.node_visits,
                              node_indices, axis=1)       # [B,K]
  new   = old + inc
  nvst  = set_rows_anydtype_masked(tree.node_visits,
                                   node_indices,
                                   new,
                                   active_mask)

  # --- embeddings ----------------------------------------------------------
  def _upd(path, old_leaf, new_leaf):
    if not isinstance(old_leaf, jnp.ndarray):
      return old_leaf
    return set_rows_anydtype_masked(old_leaf, node_indices, new_leaf,
                                    active_mask)

  updated_embeddings = jax.tree.map_with_path(_upd,
                                              tree.embeddings,
                                              new_embeds)

  # --- replace & return ----------------------------------------------------
  return tree.replace(children_prior_logits=cpl,
                      raw_values=rv,
                      node_values=nv,
                      node_visits=nvst,
                      embeddings=updated_embeddings)


def instantiate_tree_from_root(
    root: base.RootFnOutput,
    num_simulations: int,
    root_invalid_actions: chex.Array,
    extra_data: Any) -> Tree:
  """Initializes tree state at search root."""
  chex.assert_rank(root.prior_logits, 2)
  batch_size, num_actions = root.prior_logits.shape
  chex.assert_shape(root.value, [batch_size])
  num_nodes = num_simulations + 1
  data_dtype = root.value.dtype
  batch_node = (batch_size, num_nodes)
  batch_node_action = (batch_size, num_nodes, num_actions)

  def _zeros(x):
    return jnp.zeros(batch_node + x.shape[1:], dtype=x.dtype)

  # Create a new empty tree state and fill its root.
  tree = Tree(
      node_visits=jnp.zeros(batch_node, dtype=jnp.int32),
      raw_values=jnp.zeros(batch_node, dtype=data_dtype),
      node_values=jnp.zeros(batch_node, dtype=data_dtype),
      parents=jnp.full(batch_node, Tree.NO_PARENT, dtype=jnp.int32),
      action_from_parent=jnp.full(
          batch_node, Tree.NO_PARENT, dtype=jnp.int32),
      children_index=jnp.full(
          batch_node_action, Tree.UNVISITED, dtype=jnp.int32),
      children_prior_logits=jnp.zeros(
          batch_node_action, dtype=root.prior_logits.dtype),
      children_values=jnp.zeros(batch_node_action, dtype=data_dtype),
      children_visits=jnp.zeros(batch_node_action, dtype=jnp.int32),
      children_rewards=jnp.zeros(batch_node_action, dtype=data_dtype),
      children_discounts=jnp.zeros(batch_node_action, dtype=data_dtype),
      embeddings=jax.tree.map(_zeros, root.embedding),
      root_invalid_actions=root_invalid_actions,
      extra_data=extra_data)

  root_index = jnp.full([batch_size], Tree.ROOT_INDEX)
  tree = update_tree_node(
      tree, root_index, root.prior_logits, root.value, root.embedding)
  return tree

def _gather2d(src: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
    """Fast replacement for take_along_axis(src, idx, axis=1) on [B,N]."""
    mask = jax.nn.one_hot(idx, src.shape[1], dtype=src.dtype)  # [B,K,N]
    return jnp.einsum('bkn,bn->bk', mask, src)

def init_root_children_fast(
    params,
    rng_key,
    tree,
    root,                               # RootFnOutput
    recurrent_fn,
    max_num_considered_actions,         # == M
    invalid_actions,
    extra_data,
):
    B, A   = root.prior_logits.shape
    M      = max_num_considered_actions
    ROOT   = tree_lib.Tree.ROOT_INDEX
    root_row = jnp.zeros((B,), dtype=jnp.int32)

    # ------------------------------------------------------------------
    # 1.  Gumbel-Top-K  — sample the M root actions once
    # ------------------------------------------------------------------
    glogits = extra_data.glogits                         # [B, A]
    sampled_glogits, sampled_actions = jax.lax.top_k(glogits, M)   # [B,M]

    # legal_mask[b,m] == 1  iff action is legal
    legal_mask = 1 - _gather2d(invalid_actions, sampled_actions)   # [B,M]

    # how many simulations really happen in each batch row?
    total_sims = legal_mask.sum(-1).astype(jnp.int32)              # [B]

    # ------------------------------------------------------------------
    # 2.  Run the model on all B×M children in one call
    # ------------------------------------------------------------------
    BxM = B * M
    flat_act   = sampled_actions.reshape(-1)                       # (B*M,)
    rng_key, subkey = jax.random.split(rng_key)
    flat_keys  = jax.random.split(subkey, BxM).reshape(BxM, -1)
    flat_emb   = jax.tree_map(lambda x: jnp.repeat(x, M, axis=0),
                              root.embedding)

    step_flat, emb_flat = recurrent_fn(params, flat_keys, flat_act, flat_emb)

    def unflat(x): return x.reshape(B, M, *x.shape[1:])
    step      = jax.tree_map(unflat, step_flat)                    # [B,M,…]
    emb_new   = jax.tree_map(unflat, emb_flat)

    # ------------------------------------------------------------------
    # 3.  Allocate node indices 1…M (same for every batch row)
    # ------------------------------------------------------------------
    node_ids = jnp.arange(1, M + 1, dtype=jnp.int32)               # (M,)
    batch_node_ids = jnp.broadcast_to(node_ids, (B, M))            # [B,M]

    # ------------------------------------------------------------------
    # 4.  Write **all** children rows (1…M) in one masked operation
    # ------------------------------------------------------------------
    tree = update_tree_node_multi(
        tree,
        batch_node_ids,               # [B,M] indices
        step.prior_logits,            # [B,M,A]
        step.value,                   # [B,M]
        emb_new,                      # pytree [B,M,…]
        active_mask=legal_mask.astype(bool)
    )

    def _masked_write(tab, val):
        # tab:  [B,N,A], val: [B,M]
        return set_row_cols_masked(
            tab,
            root_row,                 # row 0
            sampled_actions,          # columns [B,M]
            val,
            legal_mask,               # same mask
        )

    # tree = tree.replace(
    #     children_index     = _masked_write(
    #                             tree.children_index, batch_node_ids),
    #     children_rewards   = _masked_write(tree.children_rewards, step.reward),
    #     children_discounts = _masked_write(tree.children_discounts,
    #                                        step.discount),
    # )

    visit_ones = jnp.ones_like(step.reward, dtype=tree.children_visits.dtype)

    tree = tree.replace(
        children_index     = _masked_write(tree.children_index,     batch_node_ids),
        children_rewards   = _masked_write(tree.children_rewards,   step.reward),
        children_discounts = _masked_write(tree.children_discounts, step.discount),
        children_visits    = _masked_write(tree.children_visits,    visit_ones),
        children_values    = _masked_write(tree.children_values,    step.value),
        # … rank-2 parents/action_from_parent block stays unchanged …
    )

    # ------------------------------------------------------------------
    # 4-bis) write rank-2 tables (parents, action_from_parent)
    # ------------------------------------------------------------------
    root_parent_ids = jnp.zeros_like(batch_node_ids)        # 0 for every child

    tree = tree.replace(
        parents = set_rows_anydtype_masked(                 # [B,N]
            tree.parents,
            batch_node_ids,            # rows 1…M we just created
            root_parent_ids,           # each points to ROOT (0)
            legal_mask),
        action_from_parent = set_rows_anydtype_masked(      # [B,N]
            tree.action_from_parent,
            batch_node_ids,
            sampled_actions,           # the action that led to the child
            legal_mask),
    )



    # ------------------------------------------------------------------
    # 5.  Backup root stats in vector form (no scatter)
    # ------------------------------------------------------------------
    q = step.reward + step.discount * step.value          # [B,M]
    visit_inc = legal_mask.astype(tree.node_visits.dtype) # [B,M]

    add_cnt  = visit_inc.sum(-1)                          # [B]
    add_vsum = (q * visit_inc).sum(-1)                    # [B]

    old_vis  = tree.node_visits[:, ROOT]
    old_val  = tree.node_values[:, ROOT]

    new_vis = old_vis + add_cnt
    new_val = (old_val * old_vis + add_vsum) / jnp.maximum(new_vis, 1)

    tree = tree.replace(
        node_visits = set_row(tree.node_visits, root_row, new_vis),
        node_values = set_row(tree.node_values, root_row, new_val),
    )

    return tree, total_sims, sampled_glogits, sampled_actions


def init_root_children(params, rng_key, tree, root, recurrent_fn, max_num_considered_actions, invalid_actions, extra_data):

  batch_size = tree_lib.infer_batch_size(tree)

  n_legal_actions = jnp.sum(~invalid_actions, axis=-1)
  glogits = extra_data.glogits
  sampled_glogits, sampled_root_actions = jax.lax.top_k(glogits, max_num_considered_actions)     # [B, 16]

  # Expand root in parallel -------------------------------------------------
  BxM = batch_size * max_num_considered_actions
  root_flat_actions = sampled_root_actions.reshape(-1)

  rng_key, _rng = jax.random.split(rng_key)
  root_flat_keys    = jax.random.split(_rng, BxM).reshape(BxM, -1)
  root_flat_embed   = jax.tree_map(lambda x: jnp.repeat(x, max_num_considered_actions, axis=0),
                              root.embedding)

  layer1_flat_out, layer1_flat_emb = recurrent_fn(params, root_flat_keys, root_flat_actions,
                                          root_flat_embed)

  def unflat(x):                                          # helper unchanged
      return x.reshape(batch_size, max_num_considered_actions, *x.shape[1:])

  layer1_out     = jax.tree_map(unflat, layer1_flat_out) # [Bx16,] -> [B, 16]
  layer1_embeddings = jax.tree_map(unflat, layer1_flat_emb) # [B, 16]
  layer1_qvalues = layer1_out.reward + layer1_out.discount * layer1_out.value
  layer1_visits  = jnp.ones_like(layer1_qvalues, dtype=jnp.int32) # visits = 1

  # Fill in the values into Tree
  ### [opt2]
  parent_index = Tree.ROOT_INDEX

  ##
  xs = jnp.arange(max_num_considered_actions, dtype=jnp.int32)
  new_node_idxs = jnp.arange(1, max_num_considered_actions+1, dtype=jnp.int32)
  batch_new_node_idxs = jnp.broadcast_to(new_node_idxs, (batch_size, max_num_considered_actions))  # shape (B, K)
  print("[opt] batch_new_node_idxs", batch_new_node_idxs.shape)
  action_idxs = sampled_root_actions # [B, M]

  root_idx_vec   = jnp.full((batch_size,), Tree.ROOT_INDEX, dtype=jnp.int32)

  def expand_root_child(carry, i):
    next_node_index = new_node_idxs[i] # ()
    # jax.debug.print("[legal] next_node_index: {}", next_node_index)
    batch_next_node_index = jnp.full((batch_size,), new_node_idxs[i], dtype=jnp.int32)  # shape [B]

    action_idx = action_idxs[:, i] # [B]
    print("action_idx", action_idx.shape)

    tree, num_valid_sims = carry

    # action = action_idx
    # next_node_index = num_valid_sims

    step_reward = layer1_out.reward[:, i] # [B,]
    step_discount = layer1_out.discount[:, i] # [B,]
    step_prior_logits = layer1_out.prior_logits[:, i] # [B, A]
    step_value = layer1_out.value[:, i] # [B,]
    # step_embedding = layer1_embeddings[:, action_idx] # [B, ...]
    step_embedding = jax.tree.map(lambda x: x[:, i, ...], layer1_embeddings)

    # Fill values
    is_action_invalid = invalid_actions[:, i] # [B,]
    # jax.debug.print("[legal] is_action_invalid0: {}", is_action_invalid)
    is_action_invalid = jnp.take_along_axis(
                      invalid_actions,
                      action_idx[:, None],              # gather per‑batch
                      axis=1)[:, 0]                     # -> [B]
    # jax.debug.print("[legal] is_action_invalid: {}", is_action_invalid)
    print("[opt] is_action_invalid",is_action_invalid.shape)
    print("[opt] action_idx",action_idx.shape)
    print("[opt] tree.children_index",tree.children_index.shape)
    print("[opt] tree.children_rewards",tree.children_rewards.shape)
    print("[opt] tree.children_discounts",tree.children_discounts.shape)
    print("[opt] tree.parents",tree.parents.shape)
    print("[opt] tree.action_from_parent",tree.action_from_parent.shape)
    # [opt] is_action_invalid (1, 1, 16)
    # [opt] tree.children_index (1, 17, 16)
    # [opt] tree.children_rewards (1, 17, 16)
    # [opt] tree.children_discounts (1, 17, 16)
    # [opt] tree.parents (1, 17)

    print("tree.children_index[:, Tree.ROOT_INDEX, action_idx]",tree.children_index[:, Tree.ROOT_INDEX, action_idx].shape)
    print("tree.children_index[:, Tree.ROOT_INDEX, action_idx]2",next_node_index.shape)

    batch_arange = jnp.arange(tree.children_index.shape[0]) # [B]
    print("arange tree.children_index[:, Tree.ROOT_INDEX, action_idx]",tree.children_index[batch_arange, Tree.ROOT_INDEX, action_idx].shape)
    print("arange tree.children_index[:, Tree.ROOT_INDEX, action_idx]2",next_node_index.shape)

    # child_index = jnp.where(is_action_invalid, tree.children_index[batch_arange, Tree.ROOT_INDEX, action_idx], next_node_index)
    child_index = jnp.where(is_action_invalid, Tree.UNVISITED, next_node_index)
    child_reward = jnp.where(is_action_invalid, tree.children_rewards[batch_arange, Tree.ROOT_INDEX, action_idx], step_reward)
    child_discount = jnp.where(is_action_invalid, tree.children_discounts[batch_arange, Tree.ROOT_INDEX, action_idx], step_discount)
    child_parent = jnp.where(is_action_invalid, tree.parents[:, next_node_index], Tree.ROOT_INDEX)
    child_action_from_parent = jnp.where(is_action_invalid, tree.action_from_parent[:, next_node_index], action_idx)
    print("[opt] maybe_children_index", child_index.shape)
    print("[opt] maybe_child_reward", child_reward.shape)
    print("[opt] maybe_child_discount", child_discount.shape)
    print("[opt] maybe_child_parent", child_parent.shape)
    print("[opt] maybe_child_action_from_parent", child_action_from_parent.shape)

    tree = tree.replace(
      ### somewhat optimized ,but has drift
      # # ---------- rank‑3 tables (parent, action) ----------
      # children_index      = set_cell(tree.children_index,
      #                                 root_idx_vec, action_idx, child_index),
      # children_rewards    = set_cell(tree.children_rewards,
      #                                 root_idx_vec, action_idx, child_reward),
      # children_discounts  = set_cell(tree.children_discounts,
      #                                 root_idx_vec, action_idx, child_discount),
      # # ---------- rank‑2 tables (row = node) --------------
      # parents             = set_row(tree.parents, next_node_index, child_parent),
      # action_from_parent  = set_row(tree.action_from_parent, next_node_index, child_action_from_parent),

      #### original
      children_index=batch_update(
          tree.children_index, child_index, root_idx_vec, action_idx),
      children_rewards=batch_update(
          tree.children_rewards, child_reward, root_idx_vec, action_idx),
      children_discounts=batch_update(
          tree.children_discounts, child_discount, root_idx_vec, action_idx),
      # parents=batch_update(tree.parents, child_parent, next_node_index),
      parents             = set_row(tree.parents, next_node_index, child_parent),
      action_from_parent  = set_row(tree.action_from_parent, next_node_index, child_action_from_parent),
      # action_from_parent=batch_update(
      #     tree.action_from_parent, child_action_from_parent, next_node_index)
    )

    print("[opt] 2children_index", tree.children_index.shape)
    print("[opt] 2child_reward", tree.children_rewards.shape)
    print("[opt] 2child_discount", tree.children_discounts.shape)


    batch_next = batch_new_node_idxs[:, i]   # this is shape [B]
    print("[opt] batch_next.shape", batch_next.shape)

    # tree = update_tree_node(
    #   tree, batch_next, step_prior_logits, step_value, step_embedding)
    tree = update_tree_node_masked(
      tree, batch_next, step_prior_logits, step_value, step_embedding, is_action_invalid)

    # Prep next update
    num_valid_sims = jnp.where(is_action_invalid, num_valid_sims, num_valid_sims + 1)
    carry = (tree, num_valid_sims)
    return carry, None

  init_valid_sims = jnp.zeros([batch_size], dtype=jnp.int32)
  carry = (tree, init_valid_sims)
  carry, _nones = jax.lax.scan(expand_root_child, carry, xs)
  tree, total_sims = carry

  def backward_root_children(tree):
    # ----- data for the 16 children we have just evaluated -------------
    child_value     = layer1_out.value                    # [B, M]
    child_qvalues = layer1_out.reward + layer1_out.discount * layer1_out.value # [B, M]
    child_visit = jnp.ones_like(child_qvalues, dtype=jnp.int32)

    # validity mask: 1 for legal, 0 for illegal
    # legal_mask  = (~invalid_actions[batch_range, action_idxs])  # [B,M]
    legal_mask = ~jnp.take_along_axis(
        invalid_actions,            # (B, A)
        action_idxs,                # (B, M)
        axis=1                      # gather along action dimension
    )                               # result → (B, M)
    # jax.debug.print("[backward root children] legal_mask: {}", legal_mask)

    # ---------- update per-child tables --------------------------------
    root_children_values  = set_row_cols_masked(
        tree.children_values, root_idx_vec, action_idxs, child_value, legal_mask)
    root_children_visits  = set_row_cols_masked(
        tree.children_visits, root_idx_vec, action_idxs, child_visit, legal_mask)

    # ---------- aggregate back into the root node ----------------------
    added_vsum   = jnp.sum(child_qvalues * child_visit * legal_mask, axis=-1)   # [B]
    added_cnt    = (child_visit * legal_mask).sum(-1)                     # [B]

    # old_vsum     = tree.node_values[batch_range, Tree.ROOT_INDEX] # [B]
    new_cnt      = (1 + added_cnt)
    new_val      = (root.value + added_vsum) / new_cnt                   # [B]

    root_value   = set_row(tree.node_values, root_idx_vec, new_val)
    root_visits  = set_row(tree.node_visits, root_idx_vec, new_cnt)

    # ---------- write everything back into the tree --------------------
    return tree.replace(node_values      = root_value,
                        node_visits      = root_visits,
                        children_values  = root_children_values,
                        children_visits  = root_children_visits)

  # ------------------------------------------------------------------
  # call the new function right after the lax.scan that inserts nodes
  # ------------------------------------------------------------------
  tree = backward_root_children(tree)

  # jax.debug.print("total sims?: {}", total_sims)
  return tree, total_sims, sampled_glogits, sampled_root_actions
