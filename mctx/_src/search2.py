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

def search2(
    params: base.Params,
    rng_key: chex.PRNGKey,
    *,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    root_action_selection_fn: base.RootActionSelectionFn,
    interior_action_selection_fn: base.InteriorActionSelectionFn,
    num_simulations: int,
    max_num_considered_actions: int,
    max_depth: Optional[int] = None,
    invalid_actions: Optional[chex.Array] = None,
    extra_data: Any = None,
    loop_fn: base.LoopFn = jax.lax.fori_loop) -> Tree:
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

  action_selection_fn = action_selection.switching_action_selection_wrapper(
      root_action_selection_fn=root_action_selection_fn,
      interior_action_selection_fn=interior_action_selection_fn
  )

  # Do simulation, expansion, and backward steps.
  batch_size = root.value.shape[0]
  batch_range = jnp.arange(batch_size)
  if max_depth is None:
    max_depth = num_simulations
  if invalid_actions is None:
    invalid_actions = jnp.zeros_like(root.prior_logits)

  # Instead of root action selection, we parallel expand and fill in the tree
  # Each root explorer keeps track of its num_sims expanded thus far
  active_explorer_table = seq_halving.get_active_explorer_table(max_num_considered_actions, num_simulations, max_num_considered_actions) # [M+1, nsim, M]

  def cond_fun(loop_state):
    tree, sims, round, _rng_key = loop_state
    # [todo] what about cases where M is greater than sim?
    # return ~jnp.all(sims >= num_simulations)
    return round < 1

  # def body_fun(sim, loop_state):
  def body_fun(loop_state):
    tree, sims, round_i, rng_key = loop_state

    rng_key, simulate_key, expand_key = jax.random.split(rng_key, 3)

    # --- 1. Simulate
    # simulate is vmapped and expects batched rng keys.
    simulate_keys = jax.random.split(simulate_key, batch_size)

    num_valid_actions = jnp.sum(1 - invalid_actions, axis=-1).astype(jnp.int32)
    num_considered = jnp.minimum(
        max_num_considered_actions, num_valid_actions)
    active_explorer_mask = active_explorer_table[num_considered, round_i]
    parent_index, action = simulate2(
        simulate_keys, tree, sims, active_explorer_mask, action_selection_fn, max_depth, root=root) # [B,M] for both, -1 if inactive explorer

    jax.debug.print("[sim] parent indx: {}, action: {}", parent_index, action)
    # A node first expanded on simulation `i`, will have node index `i`.
    # Node 0 corresponds to the root node.

    # ### [optblock]
    # # # opt1
    # # next_node_index = tree.children_index[batch_range, parent_index, action]
    # next_node_index = fast_gather_child(tree.children_index.astype(jnp.int32),
    #                                 parent_index,    # [B]
    #                                 action)          # [B]
    # next_node_index = jnp.where(next_node_index == Tree.UNVISITED,
    #                             sim + 1, next_node_index)

    # tree = expand2(
    #     params, expand_key, tree, recurrent_fn, parent_index,
    #     action, next_node_index)
    # tree = backward2(tree, next_node_index)
    loop_state = (tree, sims, round_i + 1, rng_key)
    return loop_state

  # Allocate all necessary storage.
  tree = instantiate_tree_from_root(root, num_simulations,
                                    root_invalid_actions=invalid_actions,
                                    extra_data=extra_data)

  ### opt2
  tree, total_sims = init_root_children(
    params,
    rng_key,
    tree,
    root,
    recurrent_fn,
    max_num_considered_actions,
    invalid_actions,
    extra_data) # equiv. to simulate, expand, backwards for first layer

  # search tree, total sims expanded, round index, rng

  init_carry = (tree, total_sims, 0, rng_key)
  jax.debug.print("total sims2?: {}", total_sims)
  # total_sims = jnp.full_like(total_sims, fill_value=16)

  # [todo] reenable]
  tree, _total_sims, _round, _rng = jax.lax.while_loop(cond_fun, body_fun, init_carry)

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
  top_root_actions: chex.Array


# @functools.partial(jax.vmap, in_axes=[0, None, 0, 0, 0, None, None], out_axes=0)
# def simulate_root_child(
#     rng_key: chex.PRNGKey,
#     tree: Tree, # unbatched [N, ...] tree
#     top_root_actions,
#     node_index,
#     depth,
#     action_selection_fn: base.InteriorActionSelectionFn,
#     max_depth: int,
#     *,
#     top_m: int = 16) -> Tuple[chex.Array, chex.Array]:
#   """Traverses the tree for a single "root child explorer" until reaching an unvisited action or `max_depth`.

#   Each simulation starts from the root and keeps selecting actions traversing
#   the tree until a leaf or `max_depth` is reached.

#   Args:
#     rng_key: random number generator state, the key is consumed.
#     tree: _unbatched_ MCTS tree state.
#     action_selection_fn: function used to select an action during simulation.
#     max_depth: maximum search tree depth allowed during simulation.

#   Returns:
#     `(parent_index, action)` tuple, where `parent_index` is the index of the
#     node reached at the end of the simulation, and the `action` is the action to
#     evaluate from the `parent_index`.
#   """
#   def cond_fun(state):
#     return state.is_continuing

#   def body_fun(state):
#     # Preparing the next simulation state.
#     node_index = state.next_node_index
#     rng_key, action_selection_key = jax.random.split(state.rng_key)
#     action = action_selection_fn(action_selection_key, tree, node_index,
#                                  state.depth)
#     next_node_index = tree.children_index[node_index, action]
#     # The returned action will be visited.
#     depth = state.depth + 1
#     is_before_depth_cutoff = depth < max_depth
#     is_visited = next_node_index != Tree.UNVISITED
#     is_continuing = jnp.logical_and(is_visited, is_before_depth_cutoff)
#     return _SimulationState(  # pytype: disable=wrong-arg-types  # jax-types
#         rng_key=rng_key,
#         node_index=node_index,
#         action=action,
#         next_node_index=next_node_index,
#         depth=depth,
#         is_continuing=is_continuing,
#         top_root_actions=state.top_root_actions)



#   # pytype: disable=wrong-arg-types  # jnp-type
#   initial_state = _SimulationState(
#       rng_key=rng_key,
#       node_index=tree.NO_PARENT,
#       action=tree.NO_PARENT,
#       next_node_index=node_index,
#       depth=depth,
#       is_continuing=jnp.array(True),
#       top_root_actions=top_root_actions)
#   # pytype: enable=wrong-arg-types
#   end_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)

#   # Returning a node with a selected action.
#   # The action can be already visited, if the max_depth is reached.
#   return end_state.node_index, end_state.action

# ---------------------------------------------------------------------------
# Roll out ONE root-child explorer (vectorised over axis-0)
# ---------------------------------------------------------------------------
# VMAP layout:
#   rng_key            – 0
#   tree               – None  (shared across the M explorers)
#   is_active          – 0     (bool, True = run, False = skip)
#   top_root_actions   – 0     (unused – kept for API compatibility)
#   start_node_index   – 0
#   depth              – 0
#   action_selection_fn--None
#   max_depth--None
@functools.partial(
    jax.vmap,
    in_axes=(0, None, 0, 0, 0, 0, None, None),
    out_axes=(0, 0))
def simulate_root_child(
    rng_key: chex.PRNGKey,
    tree: Tree,
    is_active: chex.Array,             # bool[M]
    top_root_actions: chex.Array,      # int32[M]  (ignored here)
    node_index: chex.Array,            # int32[M]  (1 … M)
    depth: chex.Array,                 # int32[M]  (=1)
    action_selection_fn: base.InteriorActionSelectionFn,
    max_depth: int
) -> Tuple[chex.Array, chex.Array]:
  """Idle explorers return (Tree.NO_PARENT, Tree.NO_PARENT)."""

  NO_PARENT = jnp.asarray(Tree.NO_PARENT, jnp.int32)

  # -- initial state --------------------------------------------------------
  init_state = _SimulationState(
      rng_key          = rng_key,
      node_index       = NO_PARENT,          # sentinel
      action           = NO_PARENT,          # sentinel
      next_node_index  = node_index,         # 1…M (or anything)
      depth            = depth,
      is_continuing    = is_active,          # <- key line!
      top_root_actions = top_root_actions)

  # -- body of the MuZero roll-out -----------------------------------------
  def cond_fun(state):
    return state.is_continuing

  def body_fun(state):
    cur_node             = state.next_node_index
    rng_key, sk          = jax.random.split(state.rng_key)
    act                  = action_selection_fn(sk, tree, cur_node, state.depth)
    nxt                  = tree.children_index[cur_node, act]
    d                    = state.depth + 1
    cont                 = jnp.logical_and(d < max_depth,
                                           nxt != Tree.UNVISITED)
    return _SimulationState(rng_key, cur_node, act, nxt, d, cont,
                            state.top_root_actions)

  end_state = jax.lax.while_loop(cond_fun, body_fun, init_state)
  return end_state.node_index, end_state.action



# @functools.partial(jax.vmap, in_axes=[0, 0, 0, None, None, None], out_axes=0)
# def simulate2(
#     rng_key: chex.PRNGKey,
#     tree: Tree,
#     sims_done: chex.Array,   # (B,)   current #simulations / game
#     active_explorer_mask: chex.Array  # (M)
#     action_selection_fn: base.InteriorActionSelectionFn,
#     max_depth: int,
#     *,

#     root: base.RootFnOutput,
#     top_m: int = 16) -> Tuple[chex.Array, chex.Array]:
#   """Traverses the tree until reaching an unvisited action or `max_depth`.

#   Each simulation starts from the root and keeps selecting actions traversing
#   the tree until a leaf or `max_depth` is reached.

#   Args:
#     rng_key: random number generator state, the key is consumed.
#     tree: _unbatched_ MCTS tree state.
#     action_selection_fn: function used to select an action during simulation.
#     max_depth: maximum search tree depth allowed during simulation.

#   Returns:
#     `(parent_index, action)` tuple, where `parent_index` is the index of the
#     node reached at the end of the simulation, and the `action` is the action to
#     evaluate from the `parent_index`.
#   """


#   ### [p]
#   glogits = tree.extra_data.glogits
#   _, top_root_actions = jax.lax.top_k(glogits, top_m)     # [B, 16]
#   root_children_idxs = jnp.arange(top_m, dtype=jnp.int32) + 1
#   depths = jnp.ones_like(node_idxs)


#   root_children_keys = jax.random.split(rng_key, top_m)

#   # Returning multiple node indices w/ size [m] with size [m] actions tensor.
#   # The action can be already visited, if the max_depth is reached.
#   return simulate_root_child(root_children_keys, tree, top_root_actions, root_children_idxs, depths, action_selection_fn, max_depth)


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
    in_axes=(0, 0, 0, 0, None, None),   # rng, tree, sims_done, mask are batched
    out_axes=(0, 0))                    # outputs → [B, M]
def simulate2(
    rng_key: chex.PRNGKey,
    tree: Tree,
    sims_done: chex.Array,              # (B,)  – unused here but kept for API
    active_explorer_mask: chex.Array,   # (B, M) after vmap ⇒ (M) here
    action_selection_fn: base.InteriorActionSelectionFn,
    max_depth: int,
    *,
    root: base.RootFnOutput,            # not needed inside function
    top_m: int = 16                     # == M
) -> Tuple[chex.Array, chex.Array]:
  """
  Runs one simulation for every *active* root-child explorer.

  Returns
  -------
  parent_index : int32[ M ]   – parent node where rollout stopped
  action       : int32[ M ]   – action to expand from that parent
                               (both are Tree.NO_PARENT for idle explorers)
  """

  # -- 1. Identify the top-M root actions (per game) --------------------------
  #     We use the log-policy stored in extra_data for that game.
  glogits           = tree.extra_data.glogits                    # (A,)
  _, root_actions   = jax.lax.top_k(glogits, top_m)              # (M,)

  # -- 2. Boolean mask of live explorers -------------------------------------
  active            = active_explorer_mask != -1                 # (M,)

  # -- 3. Prepare per-explorer start nodes, depths & keys --------------------
  start_nodes       = jnp.arange(1, top_m + 1, dtype=jnp.int32)  # node 1…M
  depths            = jnp.ones((top_m,), dtype=jnp.int32)        # depth = 1
  subkeys           = jax.random.split(rng_key, top_m)           # (M,)

  # -- 4. Roll out every explorer in parallel -------------------------------
  #     simulate_root_child is already vmapped over axis 0.
  parent_idx, act = simulate_root_child(
      subkeys,
      tree,
      active,           # <- pass boolean mask third
      root_actions,     # <- now fourth
      start_nodes,
      depths,
      action_selection_fn,
      max_depth)


  # -- 5. Mask-out idle explorers -------------------------------------------
  parent_idx = jnp.where(active, parent_idx, Tree.NO_PARENT)
  act        = jnp.where(active, act,    Tree.NO_PARENT)

  return parent_idx.astype(jnp.int32), act.astype(jnp.int32)


def expand2(
    params: chex.Array,
    rng_key: chex.PRNGKey,
    tree: Tree[T],
    recurrent_fn: base.RecurrentFn,
    parent_index: chex.Array,
    action: chex.Array,
    next_node_index: chex.Array) -> Tree[T]:
  """Create and evaluate child nodes from given nodes and unvisited actions.

  Args:
    params: params to be forwarded to recurrent function.
    rng_key: random number generator state.
    tree: the MCTS tree state to update.
    recurrent_fn: a callable to be called on the leaf nodes and unvisited
      actions retrieved by the simulation step, which takes as args
      `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
      and the new state embedding. The `rng_key` argument is consumed.
    parent_index: the index of the parent node, from which the action will be
      expanded. Shape `[B]`.
    action: the action to expand. Shape `[B]`.
    next_node_index: the index of the newly expanded node. This can be the index
      of an existing node, if `max_depth` is reached. Shape `[B]`.

  Returns:
    tree: updated MCTS tree state.
  """
  batch_size = tree_lib.infer_batch_size(tree)
  batch_range = jnp.arange(batch_size)
  chex.assert_shape([parent_index, action, next_node_index], (batch_size,))

  # Retrieve states for nodes to be evaluated.

  ### optblock
  # # [opt1]
  # embedding = jax.tree.map(
  #     lambda x: x[batch_range, parent_index], tree.embeddings)

  # [opt2]
  embedding = jax.tree.map(
    lambda x: jnp.einsum('bn,bn...->b...',               # generic rank≥2
                         jax.nn.one_hot(parent_index, x.shape[1], dtype=x.dtype),
                         x),
    tree.embeddings)

  # Evaluate and create a new node.
  step, embedding = recurrent_fn(params, rng_key, action, embedding)
  chex.assert_shape(step.prior_logits, [batch_size, tree.num_actions])
  chex.assert_shape(step.reward, [batch_size])
  chex.assert_shape(step.discount, [batch_size])
  chex.assert_shape(step.value, [batch_size])


  ### [opt2]
  tree = tree.replace(
      # ---------- rank‑3 tables (parent, action) ----------
      children_index      = set_cell(tree.children_index,
                                      parent_index, action, next_node_index),
      children_rewards    = set_cell(tree.children_rewards,
                                      parent_index, action, step.reward),
      children_discounts  = set_cell(tree.children_discounts,
                                      parent_index, action, step.discount),
      # ---------- rank‑2 tables (row = node) --------------
      parents             = set_row(tree.parents,
                                      next_node_index, parent_index),
      action_from_parent  = set_row(tree.action_from_parent,
                                      next_node_index, action),
  )
  # embed row write (rank ≥3) ------------------------------
  # tree = tree.replace(
  #     embeddings = jax.tree.map(
  #         lambda t, s: set_row_vec(t, next_node_index, s),   # whole vec
  #         tree.embeddings, embedding)
  # )
  tree = update_tree_node(
    tree, next_node_index, step.prior_logits, step.value, embedding)

  return tree

  # ### [opt1]
  tree = update_tree_node(
    tree, next_node_index, step.prior_logits, step.value, embedding)

  # # Return updated tree topology.
  # return tree.replace(
  #     children_index=batch_update(
  #         tree.children_index, next_node_index, parent_index, action),
  #     children_rewards=batch_update(
  #         tree.children_rewards, step.reward, parent_index, action),
  #     children_discounts=batch_update(
  #         tree.children_discounts, step.discount, parent_index, action),
  #     parents=batch_update(tree.parents, parent_index, next_node_index),
  #     action_from_parent=batch_update(
  #         tree.action_from_parent, action, next_node_index))


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
  # updates = dict(  # pylint: disable=use-dict-literal
  #     children_prior_logits=batch_update(
  #         tree.children_prior_logits, prior_logits, node_index),
  #     raw_values=batch_update(
  #         tree.raw_values, value, node_index),
  #     node_values=batch_update(
  #         tree.node_values, value, node_index),
  #     node_visits=batch_update(
  #         tree.node_visits, new_visit, node_index),
  #     embeddings=jax.tree.map(
  #         lambda t, s: batch_update(t, s, node_index),
  #         tree.embeddings, embedding))

  ### [opt2]
  updates = dict(  # pylint: disable=use-dict-literal
      children_prior_logits = add_row_vec(
                tree.children_prior_logits, node_index, prior_logits),
      ### rank‑3 row‑vector
      # children_prior_logits = set_row_vec(tree.children_prior_logits,
      #                                       node_index, prior_logits),
      ### og
      # children_prior_logits=batch_update(
      #     tree.children_prior_logits, prior_logits, node_index),
      raw_values=set_row(tree.raw_values, node_index, value),
      node_values=set_row(tree.node_values, node_index, value),
      node_visits=set_row(tree.node_visits, node_index, new_visit),
      embeddings = jax.tree.map(
            lambda t, s: set_row_any_sparse(t, node_index, s),
            tree.embeddings, embedding),
  )

  return tree.replace(**updates)

def update_tree_node_multi(
    tree: Tree[T],
    node_indices: chex.Array,      # [B, K]
    prior_logits: chex.Array,      # [B, K, num_actions]
    values: chex.Array,            # [B, K]
    new_embeds: chex.Array,        # [B, K, ...]
) -> Tree[T]:
  """Scatter-free update of K nodes per batch element in the Tree."""
  B, K = node_indices.shape
  num_actions = tree.num_actions

  # --- 1) update children_prior_logits: shape [B, N, A] ---
  cpl = set_rows(
      tree.children_prior_logits,     # [B, N, A]
      node_indices,                   # [B, K]
      prior_logits                    # [B, K, A]
  )

  # --- 2) update raw_values & node_values: [B, N] ---
  rv = set_rows(
      tree.raw_values[..., None],     # make float 3D [B, N, 1]
      node_indices,                   # [B, K]
      values[..., None]               # [B, K, 1]
  )[..., 0]                          # back to [B, N]

  nv = set_rows(
      tree.node_values[..., None],
      node_indices,
      values[..., None]
  )[..., 0]

  # --- 3) update node_visits: increment old visits by 1 at each index ---
  # gather old visits at each new node
  old_visits = jnp.take_along_axis(
      tree.node_visits, node_indices, axis=1)   # [B, K]
  new_visits = old_visits + 1
  nvst = set_rows(
      tree.node_visits[..., None],
      node_indices,
      new_visits[..., None]
  )[..., 0]

  # --- 4) update embeddings: shape [B, N, ...] ---
  # embeddings = jax.tree_map(
  #     lambda emb_leaf: set_rows(    # your arithmetic-only helper
  #         emb_leaf,                      # [B, N, ...F]
  #         node_indices,                  # [B, K]
  #         new_embeds                 # [B, K, ...F]
  #     ),
  #     tree.embeddings
  # )
  # 4) scatter‐free multi‐row update of every embedding leaf:
  updated_embeddings = jax.tree_map(
      lambda emb_leaf: set_rows(
          emb_leaf,           # [B, N, ...F]
          node_indices,       # [B, K]
          new_embeds          # [B, K, ...F]   <-- use this!
      ),
      tree.embeddings
  )

  # emb = set_rows_arith(
  #     tree.embeddings,                # [B, N, ...]
  #     node_indices,                   # [B, K]
  #     embeddings                      # [B, K, ...]
  # )

  return tree.replace(
      children_prior_logits=cpl,
      raw_values=rv,
      node_values=nv,
      node_visits=nvst,
      # embeddings=embeddings,
      embeddings=updated_embeddings,
  )


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

def init_root_children(params, rng_key, tree, root, recurrent_fn, max_num_considered_actions, invalid_actions, extra_data):

  batch_size = tree_lib.infer_batch_size(tree)

  n_legal_actions = jnp.sum(~invalid_actions, axis=-1)
  glogits = extra_data.glogits
  _, root_actions = jax.lax.top_k(glogits, max_num_considered_actions)     # [B, 16]

  # Expand root in parallel -------------------------------------------------
  BxM = batch_size * max_num_considered_actions
  root_flat_actions = root_actions.reshape(-1)

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
  action_idxs = root_actions # [B, M]

  root_idx_vec   = jnp.full((batch_size,), Tree.ROOT_INDEX, dtype=jnp.int32)

  def expand_root_child(carry, i):
    next_node_index = new_node_idxs[i] # ()
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

    child_index = jnp.where(is_action_invalid, tree.children_index[batch_arange, Tree.ROOT_INDEX, action_idx], next_node_index)
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
      # ---------- rank‑3 tables (parent, action) ----------
      children_index      = set_cell(tree.children_index,
                                      root_idx_vec, action_idx, child_index),
      children_rewards    = set_cell(tree.children_rewards,
                                      root_idx_vec, action_idx, child_reward),
      children_discounts  = set_cell(tree.children_discounts,
                                      root_idx_vec, action_idx, child_discount),
      # ---------- rank‑2 tables (row = node) --------------
      parents             = set_row(tree.parents, next_node_index, child_parent),
      action_from_parent  = set_row(tree.action_from_parent, next_node_index, child_action_from_parent),
    )
    print("[opt] 2children_index", tree.children_index.shape)
    print("[opt] 2child_reward", tree.children_rewards.shape)
    print("[opt] 2child_discount", tree.children_discounts.shape)



    # def update_tree(tree, is_action_invalid):
    #   # [todo] somehow maybe update this node conditionally
    #   def do_update_tree_node(tree):
    #     return update_tree_node_multi(
    #       tree, batch_new_node_idxs, step_prior_logits, step_value, step_embedding)
    #   def dont_update_tree_node(tree):
    #     return tree
    #   tree = jax.lax.cond(is_action_invalid, dont_update_tree_node, do_update_tree_node, tree)
    #   return tree
    # tree = jax.vmap(update_tree)(tree, is_action_invalid)

    # tree = update_tree_node_multi(tree, batch_new_node_idxs, step_prior_logits, step_value, step_embedding)
    # tree = jax.lax.select(is_action_invalid, dont_update_tree_node, do_update_tree_node)

    # inside fill_child, we know batch_size
    batch_next = batch_new_node_idxs[:, i]   # this is shape [B]
    print("[opt] batch_next.shape", batch_next.shape)
    tree = update_tree_node(
      tree, batch_next, step_prior_logits, step_value, step_embedding)

    # Prep next update
    num_valid_sims = jnp.where(is_action_invalid, num_valid_sims, num_valid_sims + 1)
    carry = (tree, num_valid_sims)
    return carry, None

  init_valid_sims = jnp.ones([batch_size], dtype=jnp.int32)
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
  return tree, total_sims
