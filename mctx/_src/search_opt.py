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
from mctx._src import tree as tree_lib

Tree = tree_lib.Tree
T = TypeVar("T")


def search_opt(
    params: base.Params,
    rng_key: chex.PRNGKey,
    *,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    root_action_selection_fn: base.RootActionSelectionFn,
    interior_action_selection_fn: base.InteriorActionSelectionFn,
    num_k_actions: int,
    num_simulations: int,
    max_depth: Optional[int] = None,
    invalid_actions: Optional[chex.Array] = None,
    extra_data: Any = None,
) -> Tree:
  """Performs a full search and returns sampled actions (while_loop version)."""

  # Build the (root/interior) action selector once; it’s loop-invariant.
  action_selection_fn = action_selection.switching_action_selection_wrapper(
      root_action_selection_fn=root_action_selection_fn,
      interior_action_selection_fn=interior_action_selection_fn
  )

  # Shapes / defaults
  batch_size = root.value.shape[0]
  batch_range = jnp.arange(batch_size)
  if max_depth is None:
    max_depth = num_simulations
  if invalid_actions is None:
    invalid_actions = jnp.zeros_like(root.prior_logits)

  # Allocate all necessary tree storage upfront (loop-invariant).
  tree = instantiate_tree_from_root(
      root,
      num_simulations,
      root_invalid_actions=invalid_actions,
      extra_data=extra_data,
  )

  # ---- while_loop body over simulations ----
  # Carry only what *changes*: (i, rng_key, tree).
  def cond_fun(carry):
    i, _, _ = carry
    return i < num_simulations

  def body_fun(carry):
    i, key, tree = carry
    # Split RNG: one for simulate (batched), one for expand.
    key, simulate_key, expand_key = jax.random.split(key, 3)

    # simulate is vmapped; give it B distinct keys.
    simulate_keys = jax.random.split(simulate_key, batch_size)

    # Select parent/action according to the policy for this simulation.

    # parent_index, k_action, path_parent, path_action, path_depth = simulate(
    parent_index, k_action, path_memo, path_depth = simulate(
        simulate_keys, tree, action_selection_fn, max_depth
    )

    # Node created at sim i will have index i+1; 0 is root.
    next_node_index = tree.children_index[batch_range, parent_index, k_action]
    next_node_index = jnp.where(
        next_node_index == Tree.UNVISITED, i + 1, next_node_index
    )
    action = tree.children_k_indices[batch_range, parent_index, k_action]

    # Expand the chosen leaf; recurrent_fn uses params and expand_key.
    with jax.named_scope("opt_expand"):
      tree, leaf_memo = expand(
          params, expand_key, tree, recurrent_fn, parent_index, action, k_action, next_node_index, num_k_actions
      )

    # Backpropagate value/visits.
    # tree = backward(tree, next_node_index)
    tree = backward_wrap(tree, next_node_index, path_memo, path_depth, leaf_memo)

    return (i + 1, key, tree)

  # Run the loop with a lean carry. params/root/etc. are closed-over constants.
  def run_loop(rng_key, tree):
    init = (jnp.array(0, dtype=jnp.int32), rng_key, tree)
    _, _, tree = jax.lax.while_loop(cond_fun, body_fun, init)
    return tree

  # Donate the big mutable state (tree) so XLA can alias its buffers.
  run_loop_donate = jax.jit(run_loop, donate_argnums=(1,))
  new_tree = run_loop_donate(rng_key, tree)
  return new_tree

import chex

# @chex.dataclass(frozen=True)
# class _PathMemo:
class _PathMemo(NamedTuple):
  """Memoization structure for path computations."""
  parent: chex.Array  # [num_actions]
  action: chex.Array  # [num_actions]
  node_values: chex.Array  # [num_actions]
  node_visits: chex.Array  # [num_actions]
  children_discounts: chex.Array  # [num_actions]
  children_values: chex.Array  # [num_actions]
  children_rewards: chex.Array  # [num_actions]

class _LeafMemo(NamedTuple):
  """Memoization structure for path computations."""
  leaf_reward: Optional[chex.Array]
  leaf_discount: Optional[chex.Array]
  leaf_value: Optional[chex.Array]

class _SimulationState(NamedTuple):
  """The state for the simulation while loop."""
  rng_key: chex.PRNGKey
  node_index: int
  action: int
  next_node_index: int
  depth: int
  is_continuing: bool
  path_memo: Optional[_PathMemo] = None
  # path_parent: list[int]
  # path_action: list[int]
  # path_node_values: list[float]
  # path_children_values: list[float]
  # path_children_rewards: list[float]
  # path_children_discounts: list[float]

# class _SimulateMemo(NamedTuple):
#   """Memoization structure for simulate function."""
#   children_discounts: chex.Array  # [num_actions]
#   children_values: chex.Array  # [num_actions]
#   children_rewards: chex.Array  # [num_actions]
#   children_visits: chex.Array  # [num_actions]
#   children_prior_logits: chex.Array  # [num_actions]


@functools.partial(jax.vmap, in_axes=[0, 0, None, None], out_axes=0)
def simulate(
    rng_key: chex.PRNGKey,
    tree: Tree,
    action_selection_fn: base.InteriorActionSelectionFn,
    max_depth: int) -> Tuple[chex.Array, chex.Array]:
  """Traverses the tree until reaching an unvisited action or `max_depth`.

  Each simulation starts from the root and keeps selecting actions traversing
  the tree until a leaf or `max_depth` is reached.

  Args:
    rng_key: random number generator state, the key is consumed.
    tree: _unbatched_ MCTS tree state.
    action_selection_fn: function used to select an action during simulation.
    max_depth: maximum search tree depth allowed during simulation.

  Returns:
    `(parent_index, action)` tuple, where `parent_index` is the index of the
    node reached at the end of the simulation, and the `action` is the action to
    evaluate from the `parent_index`.
  """
  def cond_fun(state):
    return state.is_continuing

  def body_fun(state):
    # Preparing the next simulation state.
    node_index = state.next_node_index
    rng_key, action_selection_key = jax.random.split(state.rng_key)

    k_action = action_selection_fn(action_selection_key, tree, node_index,
                                 state.depth)
    next_node_index = tree.children_index[node_index, k_action]

    path_parent = state.path_memo.parent.at[state.depth].set(node_index)
    path_action = state.path_memo.action.at[state.depth].set(k_action)
    path_node_values = state.path_memo.node_values.at[state.depth].set(tree.node_values[node_index])
    path_node_visits = state.path_memo.node_visits.at[state.depth].set(tree.node_visits[node_index])
    path_children_values = state.path_memo.children_values.at[state.depth].set(tree.children_values[node_index, k_action])
    # path_children_node_values = state.path_memo.children_node_values.at[state.depth].set(tree.node_values[next_node_index])
    path_children_rewards = state.path_memo.children_rewards.at[state.depth].set(tree.children_rewards[node_index, k_action])
    path_children_discounts = state.path_memo.children_discounts.at[state.depth].set(tree.children_discounts[node_index, k_action])
    path_memo = _PathMemo(
        parent=path_parent,
        action=path_action,
        node_values=path_node_values,
        node_visits=path_node_visits,
        children_discounts=path_children_discounts,
        children_values=path_children_values,
        children_rewards=path_children_rewards,
    )

    # The returned action will be visited.
    depth = state.depth + 1
    is_before_depth_cutoff = depth < max_depth
    is_visited = next_node_index != Tree.UNVISITED
    is_continuing = jnp.logical_and(is_visited, is_before_depth_cutoff)
    return _SimulationState(  # pytype: disable=wrong-arg-types  # jax-types
        rng_key=rng_key,
        node_index=node_index,
        action=k_action,
        next_node_index=next_node_index,
        depth=depth,
        # is_continuing=is_continuing)
        is_continuing=is_continuing,
        path_memo=path_memo)
        # path_parent=path_parent,
        # path_action=path_action)

  node_index = jnp.array(Tree.ROOT_INDEX, dtype=jnp.int32)
  depth = jnp.zeros((), dtype=jnp.int32)

  path_memo = _PathMemo(
      parent=jnp.zeros((max_depth,)),
      action=jnp.zeros((max_depth,)),
      node_values=jnp.zeros((max_depth,)),
      node_visits=jnp.zeros((max_depth,)),
      children_discounts=jnp.zeros((max_depth,)),
      children_values=jnp.zeros((max_depth,)),
      children_rewards=jnp.zeros((max_depth,)),
  )
  # path_parent = jnp.zeros((max_depth,))
  # path_action = jnp.zeros((max_depth,))
  # path_node_values = jnp.zeros((max_depth,))
  # path_children_values = jnp.zeros((max_depth,))
  # path_children_rewards = jnp.zeros((max_depth,))
  # path_children_discounts = jnp.zeros((max_depth,))


  # pytype: disable=wrong-arg-types  # jnp-type
  initial_state = _SimulationState(
      rng_key=rng_key,
      node_index=tree.NO_PARENT,
      action=tree.NO_PARENT,
      next_node_index=node_index,
      depth=depth,
      # is_continuing=jnp.array(True))
      is_continuing=jnp.array(True),
      path_memo=path_memo,
  )
      # path_parent=path_parent,
      # path_action=path_action)

  # pytype: enable=wrong-arg-types
  with jax.named_scope("opt_simulate"):
    end_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)

  # Returning a node with a selected action.
  # The action can be already visited, if the max_depth is reached.

  # return end_state.node_index, end_state.action, end_state.path_parent, end_state.path_action, end_state.depth
  return end_state.node_index, end_state.action, end_state.path_memo, end_state.depth
  # return end_state.node_index, end_state.action


def expand(
    params: chex.Array,
    rng_key: chex.PRNGKey,
    tree: Tree[T],
    recurrent_fn: base.RecurrentFn,
    parent_index: chex.Array,
    action: chex.Array,
    k_action: chex.Array,
    next_node_index: chex.Array,
    num_k_actions: int) -> tuple[Tree[T], chex.Array]:
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
    k_action: the k index that maps to the real action. Shape `[B]`.
    next_node_index: the index of the newly expanded node. This can be the index
      of an existing node, if `max_depth` is reached. Shape `[B]`.

  Returns:
    tree: updated MCTS tree state.
  """
  batch_size = tree_lib.infer_batch_size(tree)
  batch_range = jnp.arange(batch_size)
  chex.assert_shape([parent_index, action, k_action, next_node_index], (batch_size,))

  # Retrieve states for nodes to be evaluated.
  embedding = jax.tree.map(
      lambda x: x[batch_range, parent_index], tree.embeddings)

  # Evaluate and create a new node.
  step, embedding = recurrent_fn(params, rng_key, action, embedding)
  k_logits, k_indices = jax.lax.top_k(step.prior_logits, k=num_k_actions)
  chex.assert_shape(k_logits, [batch_size, tree.num_actions])
  chex.assert_shape(step.reward, [batch_size])
  chex.assert_shape(step.discount, [batch_size])
  chex.assert_shape(step.value, [batch_size])
  tree = update_tree_node(
      tree, next_node_index, k_indices, k_logits, step.value, embedding)

  # Calculate the depth of the newly expanded node
  new_depths = tree.node_depths[batch_range, parent_index] + 1

  # Return updated tree topology.
  tree = tree.replace(
      children_index=batch_update(
          tree.children_index, next_node_index, parent_index, k_action),
      children_rewards=batch_update(
          tree.children_rewards, step.reward, parent_index, k_action),
      children_discounts=batch_update(
          tree.children_discounts, step.discount, parent_index, k_action),
      parents=batch_update(tree.parents, parent_index, next_node_index),
      action_from_parent=batch_update(
          tree.action_from_parent, k_action, next_node_index),
      # Save the computed depth
      node_depths=batch_update(
          tree.node_depths, new_depths, next_node_index)
      )
  leaf_memo = _LeafMemo(
      leaf_reward=step.reward,
      leaf_discount=step.discount,
      leaf_value=step.value
  )
  return (tree, leaf_memo)

def backward_wrap(tree, leaf_index, path_memo, path_depth, leaf_memo):
  return backward_opt(tree, path_memo, path_depth, leaf_memo)
  # return backward(tree, leaf_index)

def backward_opt(tree, path_memo, path_depth, leaf_memo):
    batch_size = tree.node_visits.shape[0]
    # Static MAX_DEPTH from the allocation (e.g., 512)
    MAX_DEPTH = path_memo.parent.shape[1]

    # 1. Allocate Delta Arrays [MAX_DEPTH, Batch]
    # We transpose to [L, B] to match the 'apply' helper expectation and for easier loop indexing
    delta_shape = (MAX_DEPTH, batch_size)
    d_node_visits = jnp.zeros(delta_shape, dtype=jnp.int32)
    d_node_values = jnp.zeros(delta_shape, dtype=jnp.float32)
    d_edge_visits = jnp.zeros(delta_shape, dtype=jnp.int32)
    d_edge_values = jnp.zeros(delta_shape, dtype=jnp.float32)

    # 2. Setup Loop State
    # Start at the deepest active path in the batch
    start_i = jnp.max(path_depth) - 1

    init_state = (
        start_i,
        jnp.zeros(batch_size, dtype=jnp.float32), # running_g
        jnp.zeros(batch_size, dtype=jnp.float32), # child_node_val_prev
        d_node_visits,
        d_node_values,
        d_edge_visits,
        d_edge_values
    )

    def cond_fun(state):
        i, _, _, _, _, _, _ = state
        return i >= 0

    def body_fun(state):
        i, running_g, child_node_val_prev, d_n_vis, d_n_val, d_e_vis, d_e_val = state

        active = i < path_depth
        is_leaf = i == (path_depth - 1)

        # Fetch stats from MEMO (Slice [B, L] -> [B])
        # This is fast/coalesced memory access
        old_node_val = path_memo.node_values[:, i]
        old_node_visits = path_memo.node_visits[:, i]
        memo_reward = path_memo.children_rewards[:, i]
        memo_discount = path_memo.children_discounts[:, i]
        old_edge_val = path_memo.children_values[:, i]

        # Calculate Values
        reward = jnp.where(is_leaf, leaf_memo.leaf_reward, memo_reward)
        discount = jnp.where(is_leaf, leaf_memo.leaf_discount, memo_discount)
        val_coming_up = jnp.where(is_leaf, leaf_memo.leaf_value, running_g)

        # Bellman: G = R + gamma * V_next
        new_g = reward + discount * val_coming_up

        # Mean Update: V_new = (V_old * N + G) / (N + 1)
        new_node_val = (old_node_val * old_node_visits + new_g) / (old_node_visits + 1.0)

        # Edge Update (Standard MCTS: edge tracks child value)
        new_edge_val = jnp.where(is_leaf, leaf_memo.leaf_value, child_node_val_prev)

        # Compute Deltas (Masked by active)
        # Note: We store difference to apply via scatter-add style matmul later
        # dnv = jnp.where(active, 1, 0).astype(jnp.int32)
        dnv = active.astype(jnp.int32)
        dnval = jnp.where(active, new_node_val - old_node_val, 0.0).astype(jnp.float32)
        # dev = jnp.where(active, 1, 0).astype(jnp.int32)
        dev = active.astype(jnp.int32)
        deval = jnp.where(active, new_edge_val - old_edge_val, 0.0).astype(jnp.float32)

        # Store in Delta Arrays
        # .at[i].set(...) inside while_loop lowers to efficient dynamic_update_slice
        d_n_vis = d_n_vis.at[i].set(dnv)
        d_n_val = d_n_val.at[i].set(dnval)
        d_e_vis = d_e_vis.at[i].set(dev)
        d_e_val = d_e_val.at[i].set(deval)

        return (i - 1, new_g, new_node_val, d_n_vis, d_n_val, d_e_vis, d_e_val)

    # 3. Run Loop
    _, _, _, final_d_n_vis, final_d_n_val, final_d_e_vis, final_d_e_val = jax.lax.while_loop(
        cond_fun, body_fun, init_state
    )

    # 4. Apply Updates using Dense Matmul Helpers
    def _apply_backward_update(node_visits, deltas, path_indices):
        """
        Updates node_visits using deltas via matrix multiplication.

        Args:
            node_visits: [B, N] Current visit counts.
            deltas: [L, B] The update values computed by backward.
            path_indices: [B, L] The node indices for the paths (aligned_parents).

        Returns:
            Updated node_visits [B, N].
        """
        B, N = node_visits.shape
        L, _ = deltas.shape

        # 1. Create the selection matrix (One-Hot)
        # Shape: [B, L] -> [B, L, N]
        # Note: Ensure indices are within [0, N).
        # Invalid/Padding indices usually point to 0 (Root) or a dummy index.
        path_mask = jax.nn.one_hot(path_indices, num_classes=N, dtype=deltas.dtype)

        # 2. Compute the dense updates using Einstein Summation
        # 'lb' = deltas [L, B]
        # 'bln' = path_mask [B, L, N]
        # 'bn' = output [B, N]
        # We sum over 'l' (the path depth dimension).
        # This effectively distributes the delta at depth L to the node at path_indices[L].
        updates = jnp.einsum('lb, bln -> bn', deltas, path_mask)

        # 3. Apply the updates
        new_node_visits = node_visits + updates

        return new_node_visits


    def _apply_backward_edge_update(tree_edge_array, deltas_LB, path_node_indices, path_action_indices):
        """
        Updates a [B, N, A] tensor (like children_visits/values) using [L, B] deltas.

        Math: Update[b, n, a] = Sum_l( Mask_Node[b, l, n] * Mask_Action[b, l, a] * Delta[l, b] )
        Efficient Implementation: BatchMatMul( Mask_Node.T,  (Mask_Action * Delta) )
        """
        B, N, A = tree_edge_array.shape
        L, _ = deltas_LB.shape
        dtype = deltas_LB.dtype

        # 1. Prepare Weighted Actions [B, L, A]
        # Create One-Hot for Actions
        action_mask = jax.nn.one_hot(path_action_indices, A, dtype=dtype) # [B, L, A]

        # Broadcast Deltas [L, B] -> [B, L, 1] and multiply
        # This assigns the delta value to the specific action taken at step L
        weighted_actions = action_mask * deltas_LB.T[..., None]

        # 2. Prepare Node Selector [B, L, N]
        node_mask = jax.nn.one_hot(path_node_indices, N, dtype=dtype)

        # 3. Compute Updates via Matrix Multiplication
        # We want to map [L] -> [N, A].
        # Operation: [B, N, L] @ [B, L, A] -> [B, N, A]
        # We transpose node_mask to [B, N, L] to line up the 'L' dimension for contraction.
        updates = jnp.matmul(node_mask.transpose(0, 2, 1), weighted_actions)

        return tree_edge_array + updates
    new_node_visits = _apply_backward_update(
        tree.node_visits, final_d_n_vis, path_memo.parent
    )

    new_children_visits = _apply_backward_edge_update(
        tree.children_visits, final_d_e_vis, path_memo.parent, path_memo.action
    )
    with jax.default_matmul_precision('float32'):
      new_node_values = _apply_backward_update(
          tree.node_values, final_d_n_val, path_memo.parent
      )
      new_children_values = _apply_backward_edge_update(
          tree.children_values, final_d_e_val, path_memo.parent, path_memo.action
      )

    return tree.replace(
        node_visits=new_node_visits,
        node_values=new_node_values,
        children_visits=new_children_visits,
        children_values=new_children_values
    )

@jax.vmap
def backward(
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
    k_action = tree.action_from_parent[index]
    reward = tree.children_rewards[parent, k_action]
    leaf_value = reward + tree.children_discounts[parent, k_action] * leaf_value
    parent_value = (
        tree.node_values[parent] * count + leaf_value) / (count + 1.0)
    children_values = tree.node_values[index]
    children_counts = tree.children_visits[parent, k_action] + 1

    tree = tree.replace(
        node_values=update(tree.node_values, parent_value, parent),
        node_visits=update(tree.node_visits, count + 1, parent),
        children_values=update(
            tree.children_values, children_values, parent, k_action),
        children_visits=update(
            tree.children_visits, children_counts, parent, k_action))

    return tree, leaf_value, parent

  leaf_index = jnp.asarray(leaf_index, dtype=jnp.int32)
  loop_state = (tree, tree.node_values[leaf_index], leaf_index)
  with jax.named_scope("opt_backward"):
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
    k_indices: Any,
    prior_logits: chex.Array,
    value: chex.Array,
    embedding: chex.Array) -> Tree[T]:
  """Updates the tree at node index.

  Args:
    tree: `Tree` to whose node is to be updated.
    k_indices: the k indices of the expanded node. Shape `[B, K]`.
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
  chex.assert_shape([prior_logits, k_indices], (batch_size, tree.num_actions))

  # When using max_depth, a leaf can be expanded multiple times.
  new_visit = tree.node_visits[batch_range, node_index] + 1
  updates = dict(  # pylint: disable=use-dict-literal
      children_prior_logits=batch_update(
          tree.children_prior_logits, prior_logits, node_index),
      children_k_indices=batch_update(
          tree.children_k_indices, k_indices, node_index),
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
      node_depths=jnp.zeros(batch_node, dtype=jnp.int32),
      parents=jnp.full(batch_node, Tree.NO_PARENT, dtype=jnp.int32),
      action_from_parent=jnp.full(
          batch_node, Tree.NO_PARENT, dtype=jnp.int32),
      children_index=jnp.full(
          batch_node_action, Tree.UNVISITED, dtype=jnp.int32),
      children_prior_logits=jnp.zeros(
          batch_node_action, dtype=root.prior_logits.dtype),
      children_k_indices=jnp.full(
          batch_node_action, -1, dtype=jnp.int16),
      children_values=jnp.zeros(batch_node_action, dtype=data_dtype),
      children_visits=jnp.zeros(batch_node_action, dtype=jnp.int32),
      children_rewards=jnp.zeros(batch_node_action, dtype=data_dtype),
      children_discounts=jnp.zeros(batch_node_action, dtype=data_dtype),
      embeddings=jax.tree.map(_zeros, root.embedding),
      root_invalid_actions=root_invalid_actions,
      extra_data=extra_data)

  root_index = jnp.full([batch_size], Tree.ROOT_INDEX)
  tree = update_tree_node(
      tree, root_index, root.k_indices, root.prior_logits, root.value, root.embedding)
  return tree
