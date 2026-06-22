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


def search(
    params: base.Params,
    rng_key: chex.PRNGKey,
    *,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    root_action_selection_fn: base.RootActionSelectionFn,
    interior_action_selection_fn: base.InteriorActionSelectionFn,
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
    parent_index, action = simulate(
        simulate_keys, tree, action_selection_fn, max_depth
    )

    # Node created at sim i will have index i+1; 0 is root.
    next_node_index = tree.children_index[batch_range, parent_index, action]
    next_node_index = jnp.where(
        next_node_index == Tree.UNVISITED, i + 1, next_node_index
    )

    # Expand the chosen leaf; recurrent_fn uses params and expand_key.
    tree = expand(
        params, expand_key, tree, recurrent_fn, parent_index, action, next_node_index
    )

    # Backpropagate value/visits.
    tree = backward(tree, next_node_index)

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

def _tree_select(mask: chex.Array, tree_old: Tree, tree_new: Tree) -> Tree:
  """Per-batch select between two trees: keep `tree_old` where `mask` is True.

  `mask` has shape `[B]`; it is broadcast over each leaf's trailing dims. Used
  to make an over-full simulation a true no-op for the affected batch elements.
  """
  def sel(a, b):
    m = mask.reshape((mask.shape[0],) + (1,) * (a.ndim - 1))
    return jnp.where(m, a, b)
  return jax.tree.map(sel, tree_old, tree_new)


def search_to_target(
    params: base.Params,
    rng_key: chex.PRNGKey,
    tree: Tree,
    target_visits: chex.Numeric,
    active_mask: chex.Array,
    *,
    recurrent_fn: base.RecurrentFn,
    root_action_selection_fn: base.RootActionSelectionFn,
    interior_action_selection_fn: base.InteriorActionSelectionFn,
    max_depth: Optional[int] = None,
    max_iters: Optional[int] = None,
) -> Tree:
  """Tops up the `active_mask` rows until their root reaches `target_visits`.

  Phase-1 kernel for batched subtree reuse: warm-started rows only need a few
  new simulations to reach the per-move budget, so instead of a fixed loop this
  runs a *dynamic* `while_loop` that stops as soon as every active row has hit
  its target. The per-move top-up count therefore varies with how much was
  inherited, with no recompilation.

  Each iteration runs a batched sim/expand/backward, but a row advances only if
  it is active, below `target_visits`, and has room (`< capacity`); every other
  row -- including all non-active (e.g. terminated) rows -- is a true no-op.
  Non-active rows are never touched and can be searched separately at a smaller
  batch size.

  Args:
    params: params forwarded to `recurrent_fn`.
    rng_key: rng state, consumed.
    tree: pre-populated batched `Tree` to continue from.
    target_visits: scalar (or `[B]`) target root visit count per row.
    active_mask: `[B]` bool; only these rows are advanced / gate termination.
    recurrent_fn, *_action_selection_fn: as in `search`.
    max_depth: maximum tree depth during simulation. Defaults to `num_nodes`.
    max_iters: hard cap on iterations (also the static loop bound). Defaults to
      the storage capacity, which bounds how many root visits a row can gain.

  Returns:
    The updated batched `Tree`.
  """
  action_selection_fn = action_selection.switching_action_selection_wrapper(
      root_action_selection_fn=root_action_selection_fn,
      interior_action_selection_fn=interior_action_selection_fn,
  )

  batch_size = tree.node_visits.shape[0]
  batch_range = jnp.arange(batch_size)
  num_nodes = tree.node_visits.shape[1]
  capacity = num_nodes - 1      # last slot is overflow scratch
  scratch_index = num_nodes - 1
  if max_depth is None:
    max_depth = num_nodes
  if max_iters is None:
    max_iters = capacity

  target = jnp.asarray(target_visits)
  active = active_mask
  next_free0 = jnp.sum(
      tree.node_visits[:, :capacity] > 0, axis=1).astype(jnp.int32)

  def _below_target(tree):
    return tree.node_visits[:, Tree.ROOT_INDEX] < target

  def cond_fun(carry):
    i, _, tree, _ = carry
    not_done = jnp.logical_and(active, _below_target(tree))
    return jnp.logical_and(i < max_iters, jnp.any(not_done))

  def body_fun(carry):
    i, key, tree, next_free = carry
    key, simulate_key, expand_key = jax.random.split(key, 3)
    simulate_keys = jax.random.split(simulate_key, batch_size)

    parent_index, action = simulate(
        simulate_keys, tree, action_selection_fn, max_depth)

    child = tree.children_index[batch_range, parent_index, action]
    is_unvisited = child == Tree.UNVISITED
    no_room = next_free >= capacity
    # A row does real work only while it is active and still below target.
    should = jnp.logical_and(active, _below_target(tree))

    write_index = jnp.where(
        is_unvisited,
        jnp.where(no_room, scratch_index, next_free),
        child)

    tree_new = expand(
        params, expand_key, tree, recurrent_fn,
        parent_index, action, write_index)
    tree_new = backward(tree_new, write_index)

    # Keep the new tree only where the row should advance and isn't over-full;
    # everything else (idle, done, terminated, overflow) is left unchanged.
    overflow = jnp.logical_and(is_unvisited, no_room)
    do_work = jnp.logical_and(should, jnp.logical_not(overflow))
    tree = _tree_select(jnp.logical_not(do_work), tree, tree_new)

    created = jnp.logical_and(should, jnp.logical_and(
        is_unvisited, jnp.logical_not(no_room)))
    next_free = next_free + created.astype(jnp.int32)

    return (i + 1, key, tree, next_free)

  def run_loop(rng_key, tree):
    init = (jnp.array(0, dtype=jnp.int32), rng_key, tree, next_free0)
    _, _, tree, _ = jax.lax.while_loop(cond_fun, body_fun, init)
    return tree

  run_loop_donate = jax.jit(run_loop, donate_argnums=(1,))
  return run_loop_donate(rng_key, tree)


class _SimulationState(NamedTuple):
  """The state for the simulation while loop."""
  rng_key: chex.PRNGKey
  node_index: int
  action: int
  next_node_index: int
  depth: int
  is_continuing: bool


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
    action = action_selection_fn(action_selection_key, tree, node_index,
                                 state.depth)
    next_node_index = tree.children_index[node_index, action]
    # The returned action will be visited.
    depth = state.depth + 1
    is_before_depth_cutoff = depth < max_depth
    is_visited = next_node_index != Tree.UNVISITED
    is_continuing = jnp.logical_and(is_visited, is_before_depth_cutoff)
    return _SimulationState(  # pytype: disable=wrong-arg-types  # jax-types
        rng_key=rng_key,
        node_index=node_index,
        action=action,
        next_node_index=next_node_index,
        depth=depth,
        is_continuing=is_continuing)

  node_index = jnp.array(Tree.ROOT_INDEX, dtype=jnp.int32)
  depth = jnp.zeros((), dtype=tree.children_prior_logits.dtype)
  # pytype: disable=wrong-arg-types  # jnp-type
  initial_state = _SimulationState(
      rng_key=rng_key,
      node_index=tree.NO_PARENT,
      action=tree.NO_PARENT,
      next_node_index=node_index,
      depth=depth,
      is_continuing=jnp.array(True))
  # pytype: enable=wrong-arg-types
  end_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)

  # Returning a node with a selected action.
  # The action can be already visited, if the max_depth is reached.
  return end_state.node_index, end_state.action


def expand(
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
  embedding = jax.tree.map(
      lambda x: x[batch_range, parent_index], tree.embeddings)

  # Evaluate and create a new node.
  step, embedding = recurrent_fn(params, rng_key, action, embedding)
  chex.assert_shape(step.prior_logits, [batch_size, tree.num_actions])
  chex.assert_shape(step.reward, [batch_size])
  chex.assert_shape(step.discount, [batch_size])
  chex.assert_shape(step.value, [batch_size])
  tree = update_tree_node(
      tree, next_node_index, step.prior_logits, step.value, embedding)

  # Calculate the depth of the newly expanded node
  new_depths = tree.node_depths[batch_range, parent_index] + 1

  # Return updated tree topology.
  return tree.replace(
      children_index=batch_update(
          tree.children_index, next_node_index, parent_index, action),
      children_rewards=batch_update(
          tree.children_rewards, step.reward, parent_index, action),
      children_discounts=batch_update(
          tree.children_discounts, step.discount, parent_index, action),
      parents=batch_update(tree.parents, parent_index, next_node_index),
      action_from_parent=batch_update(
          tree.action_from_parent, action, next_node_index),
      # Save the computed depth
      node_depths=batch_update(
          tree.node_depths, new_depths, next_node_index)
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
    action = tree.action_from_parent[index]
    reward = tree.children_rewards[parent, action]
    leaf_value = reward + tree.children_discounts[parent, action] * leaf_value
    parent_value = (
        tree.node_values[parent] * count + leaf_value) / (count + 1.0)
    children_values = tree.node_values[index]
    children_counts = tree.children_visits[parent, action] + 1

    tree = tree.replace(
        node_values=update(tree.node_values, parent_value, parent),
        node_visits=update(tree.node_visits, count + 1, parent),
        children_values=update(
            tree.children_values, children_values, parent, action),
        children_visits=update(
            tree.children_visits, children_counts, parent, action))

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
