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
"""A data structure used to hold / inspect search data for a batch of inputs."""

from __future__ import annotations
from typing import Any, ClassVar, Generic, TypeVar, Optional

import chex
import jax
import jax.numpy as jnp


T = TypeVar("T")


@chex.dataclass(frozen=True)
class Tree(Generic[T]):
  """State of a search tree.

  The `Tree` dataclass is used to hold and inspect search data for a batch of
  inputs. In the fields below `B` denotes the batch dimension, `N` represents
  the number of nodes in the tree, and `num_actions` is the number of discrete
  actions.

  node_visits: `[B, N]` the visit counts for each node.
  raw_values: `[B, N]` the raw value for each node.
  node_values: `[B, N]` the cumulative search value for each node.
  parents: `[B, N]` the node index for the parents for each node.
  action_from_parent: `[B, N]` action to take from the parent to reach each
    node.
  children_index: `[B, N, num_actions]` the node index of the children for each
    action.
  children_prior_logits: `[B, N, num_actions]` the action prior logits of each
    node.
  children_visits: `[B, N, num_actions]` the visit counts for children for
    each action.
  children_rewards: `[B, N, num_actions]` the immediate reward for each action.
  children_discounts: `[B, N, num_actions]` the discount between the
    `children_rewards` and the `children_values`.
  children_values: `[B, N, num_actions]` the value of the next node after the
    action.
  embeddings: `[B, N, ...]` the state embeddings of each node.
  root_invalid_actions: `[B, num_actions]` a mask with invalid actions at the
    root. In the mask, invalid actions have ones, and valid actions have zeros.
  extra_data: `[B, ...]` extra data passed to the search.
  """
  node_visits: chex.Array  # [B, N]
  raw_values: chex.Array  # [B, N]
  node_values: chex.Array  # [B, N]
  node_depths: chex.Array  # [B, N]
  parents: chex.Array  # [B, N]
  action_from_parent: chex.Array  # [B, N]
  children_index: chex.Array  # [B, N, num_actions]
  children_prior_logits: chex.Array  # [B, N, num_actions]
  children_visits: chex.Array  # [B, N, num_actions]
  children_rewards: chex.Array  # [B, N, num_actions]
  children_discounts: chex.Array  # [B, N, num_actions]
  children_values: chex.Array  # [B, N, num_actions]
  embeddings: Any  # [B, N, ...]
  root_invalid_actions: chex.Array  # [B, num_actions]
  extra_data: T  # [B, ...]
  # completed_qvalues: chex.Array  # [B, num_actions]
  # to_argmax: chex.Array          # [B, num_actions]
  children_k_indices: Optional[chex.Array] = None
  completed_qvalues: Optional[chex.Array] = None
  to_argmax: Optional[chex.Array] = None

  # The following attributes are class variables (and should not be set on
  # Tree instances).
  ROOT_INDEX: ClassVar[int] = 0
  NO_PARENT: ClassVar[int] = -1
  UNVISITED: ClassVar[int] = -1

  @property
  def num_actions(self):
    return self.children_index.shape[-1]

  @property
  def num_simulations(self):
    return self.node_visits.shape[-1] - 1

  def qvalues(self, indices):
    """Compute q-values for any node indices in the tree."""
    # pytype: disable=wrong-arg-types  # jnp-type
    if jnp.asarray(indices).shape:
      return jax.vmap(_unbatched_qvalues)(self, indices)
    else:
      return _unbatched_qvalues(self, indices)
    # pytype: enable=wrong-arg-types

  def summary(self, include_metrics: bool = True) -> SearchSummary:
    """Extract summary statistics for the root node."""
    # Root-level stats
    chex.assert_rank(self.node_values, 2)
    value = self.node_values[:, Tree.ROOT_INDEX]
    prior_logits = self.children_prior_logits[:, self.ROOT_INDEX, :]
    batch_size, num_actions = prior_logits.shape
    root_indices = jnp.full((batch_size,), Tree.ROOT_INDEX)
    qvalues = self.qvalues(root_indices)
    visit_counts = self.children_visits[:, Tree.ROOT_INDEX].astype(value.dtype)
    total_counts = jnp.sum(visit_counts, axis=-1, keepdims=True)
    visit_probs = visit_counts / jnp.maximum(total_counts, 1)
    visit_probs = jnp.where(
        total_counts > 0, visit_probs, 1 / self.num_actions)

    if not include_metrics:
      return SearchSummary(
          visit_counts=visit_counts,
          visit_probs=visit_probs,
          value=value,
          qvalues=qvalues)

    # Simple Regret (Root)
    max_q_root = jnp.max(qvalues, axis=-1)
    expected_q_root = jnp.sum(visit_probs * qvalues, axis=-1)
    simple_regret_root = max_q_root - expected_q_root

    # Allocation Efficiency (Root, Min-Max Normalized)
    min_q_root = jnp.min(qvalues, axis=-1)
    q_range_root = jnp.maximum(max_q_root - min_q_root, 1e-8)
    allocation_efficiency_root = (expected_q_root - min_q_root) / q_range_root

    # KL Divergence (Root)
    prior_probs = jax.nn.softmax(prior_logits, axis=-1)
    eps = 1e-8
    kl_divergence = jnp.sum(
        visit_probs * (jnp.log(visit_probs + eps) - jnp.log(prior_probs + eps)),
        axis=-1)

    # Value Improvement (Root)
    raw_root_value = self.raw_values[:, self.ROOT_INDEX]
    value_improvement = value - raw_root_value

    # Top-5 Precision (Root)
    num_legal_actions = jnp.sum(jnp.isfinite(prior_logits), axis=-1)
    k_static = 5
    k_dynamic = jnp.minimum(k_static, num_legal_actions)
    top_k_q_indices = jax.lax.top_k(qvalues, k=k_static)[1]
    top_k_visit_indices = jax.lax.top_k(visit_counts, k=k_static)[1]
    set_q = jnp.sum(jax.nn.one_hot(top_k_q_indices, num_actions), axis=1)
    set_visits = jnp.sum(jax.nn.one_hot(top_k_visit_indices, num_actions), axis=1)
    intersection_size = jnp.sum(set_q * set_visits, axis=-1)
    top_5_precision_root = intersection_size / jnp.maximum(k_dynamic, 1)

    # Tree-level stats
    visited_mask = self.node_visits > 0
    num_visited_nodes = jnp.sum(visited_mask, axis=-1)
    num_children_per_node = jnp.sum(
        self.children_index != Tree.UNVISITED, axis=-1)

    # Average Children Per *Internal* Node
    internal_node_mask = (visited_mask) & (num_children_per_node > 0)
    num_internal_nodes = jnp.sum(internal_node_mask, axis=-1)
    total_children = jnp.sum(
        num_children_per_node * internal_node_mask, axis=-1)
    avg_children_per_node = total_children / jnp.maximum(num_internal_nodes, 1)

    # Tree-wide Allocation Efficiency (Min-Max Normalized)
    all_qvalues = (
        self.children_rewards +
        self.children_discounts * self.children_values)
    all_child_visits = self.children_visits
    total_child_visits = jnp.sum(all_child_visits, axis=-1, keepdims=True)
    all_visit_probs = all_child_visits / jnp.maximum(total_child_visits, 1)
    max_q_per_node = jnp.max(all_qvalues, axis=-1)
    min_q_per_node = jnp.min(all_qvalues, axis=-1)
    expected_q_per_node = jnp.sum(all_visit_probs * all_qvalues, axis=-1)
    q_range_per_node = jnp.maximum(max_q_per_node - min_q_per_node, 1e-8)
    efficiency_per_node = (
        (expected_q_per_node - min_q_per_node) / q_range_per_node)
    avg_allocation_efficiency_tree = (
        jnp.sum(efficiency_per_node * visited_mask, axis=-1) /
        jnp.maximum(num_visited_nodes, 1))

    # Max Depth
    max_depth = jnp.max(self.node_depths * visited_mask, axis=-1)

    return SearchSummary(
        visit_counts=visit_counts,
        visit_probs=visit_probs,
        value=value,
        qvalues=qvalues,
        max_depth=max_depth,
        simple_regret_root=simple_regret_root,
        allocation_efficiency_root=allocation_efficiency_root,
        kl_divergence=kl_divergence,
        value_improvement=value_improvement,
        top_5_precision_root=top_5_precision_root,
        avg_children_per_node=avg_children_per_node,
        avg_allocation_efficiency_tree=avg_allocation_efficiency_tree)


def infer_batch_size(tree: Tree) -> int:
  """Recovers batch size from `Tree` data structure."""
  if tree.node_values.ndim != 2:
    raise ValueError("Input tree is not batched.")
  chex.assert_equal_shape_prefix(jax.tree_util.tree_leaves(tree), 1)
  return tree.node_values.shape[0]


# A number of aggregate statistics and predictions are extracted from the
# search data and returned to the user for further processing.
@chex.dataclass(frozen=True)
class SearchSummary:
  """Stats from MCTS search."""
  visit_counts: chex.Array
  visit_probs: chex.Array
  value: chex.Array
  qvalues: chex.Array
  max_depth: chex.Array
  # Root-level metrics
  simple_regret_root: chex.Array
  allocation_efficiency_root: chex.Array
  kl_divergence: chex.Array
  value_improvement: chex.Array
  top_5_precision_root: chex.Array
  # Tree-level metrics
  avg_children_per_node: chex.Array
  avg_allocation_efficiency_tree: chex.Array


def _unbatched_qvalues(tree: Tree, index: int) -> int:
  chex.assert_rank(tree.children_discounts, 2)
  return (  # pytype: disable=bad-return-type  # numpy-scalars
      tree.children_rewards[index]
      + tree.children_discounts[index] * tree.children_values[index]
  )
