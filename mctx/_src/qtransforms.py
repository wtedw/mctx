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
"""Monotonic transformations for the Q-values."""

import chex
import jax
import jax.numpy as jnp

from mctx._src import tree as tree_lib
from typing import Tuple

def qtransform_completed_by_mix_value_bfs2(
    root_qvalues,
    root_raw_value,
    root_prior_logits,
    layer1_visit_counts,
    *,
    value_scale: chex.Numeric = 0.1,
    maxvisit_init: chex.Numeric = 50.0,
    rescale_values: bool = True,
    use_mixed_value: bool = True,
    epsilon: chex.Numeric = 1e-8,
# ) -> chex.Array:
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]: # [debug]
  """Returns completed qvalues.

  The missing Q-values of the unvisited actions are replaced by the
  mixed value, defined in Appendix D of
  "Policy improvement by planning with Gumbel":
  https://openreview.net/forum?id=bERaNdoegnO

  The Q-values are transformed by a linear transformation:
    `(maxvisit_init + max(visit_counts)) * value_scale * qvalues`.

  Args:
    tree: _unbatched_ MCTS tree state.
    node_index: scalar index of the parent node.
    value_scale: scale for the Q-values.
    maxvisit_init: offset to the `max(visit_counts)` in the scaling factor.
    rescale_values: if True, scale the qvalues by `1 / (max_q - min_q)`.
    use_mixed_value: if True, complete the Q-values with mixed value,
      otherwise complete the Q-values with the raw value.
    epsilon: the minimum denominator when using `rescale_values`.

  Returns:
    Completed Q-values. Shape `[num_actions]`.
  """
  qvalues = root_qvalues
  visit_counts = layer1_visit_counts
  raw_value = root_raw_value
  prior_probs = jax.nn.softmax(
    root_prior_logits)

  # chex.assert_shape(node_index, ())
  # qvalues = tree.qvalues(node_index)
  # visit_counts = tree.children_visits[node_index]

  # Computing the mixed value and producing completed_qvalues.
  # raw_value = tree.raw_values[node_index]
  # prior_probs = jax.nn.softmax(
  #     tree.children_prior_logits[node_index])
  mixed_value = _compute_mixed_value(
      raw_value,
      qvalues=qvalues,
      visit_counts=visit_counts,
      prior_probs=prior_probs)
  if use_mixed_value:

    value = mixed_value
    # value = _compute_mixed_value(
    #     raw_value,
    #     qvalues=qvalues,
    #     visit_counts=visit_counts,
    #     prior_probs=prior_probs)
  else:
    value = raw_value
  completed_qvalues = _complete_qvalues(
      qvalues, visit_counts=visit_counts, value=value)

  # Scaling the Q-values.
  rescaled_qvalues = _rescale_qvalues(completed_qvalues, epsilon) # [debug]
  if rescale_values:
    # completed_qvalues = _rescale_qvalues(completed_qvalues, epsilon)
    completed_qvalues = rescaled_qvalues # [debug]
  maxvisit = jnp.max(visit_counts, axis=-1)
  visit_scale = maxvisit_init + maxvisit

  original_res = visit_scale * value_scale * completed_qvalues
  return (raw_value, mixed_value, maxvisit, rescaled_qvalues, original_res)
  # return visit_scale * value_scale * completed_qvalues

def compute_bfs_completed_qvalues(
    children_outputs,  # a pytree with fields: reward, discount, value; shape [B, num_actions]
    root,              # a RootFnOutput (only root.value might be needed for reference)
    value_scale: chex.Numeric = 0.1,
    maxvisit_init: chex.Numeric = 50.0,
    rescale_values: bool = False,
    # [ted] maybe reintroduce if larger board sizes
    # and we only visit a subset of actions
    # use_mixed_value: bool = True,
    epsilon: chex.Numeric = 1e-8,
):
  """
  Computes completed Q-values for the root from a one-layer expansion.

  Args:
    children_outputs: Output of recurrent_fn for all actions, with fields:
                      reward, discount, value. Each is [B, num_actions].
    root: The root output. Here, root.value is available if needed.
    value_scale: Scaling factor for the Q-values.
    maxvisit_init: Constant used for scaling (with all actions visited once, max=1).
    rescale_values: If True, rescale the Q-values to [0,1].
    epsilon: Small constant to avoid division by zero in rescaling.

  Returns:
    final_qvalues: A [B, num_actions] array with the completed (and scaled) Q-values.
  """
  # Compute raw Q-values:
  qvalues = children_outputs.reward + children_outputs.discount * children_outputs.value

  # Optionally, rescale the Q-values to [0, 1].
  if rescale_values:
    qvalues = _rescale_qvalues(qvalues, epsilon)

  # Compute scaling factor:
  # With every action visited exactly once, max(visit_counts) == 1.
  visit_scale = maxvisit_init + 1
  final_qvalues = (visit_scale * value_scale) * qvalues
  return final_qvalues

def qtransform_by_min_max(
    tree: tree_lib.Tree,
    node_index: chex.Numeric,
    *,
    min_value: chex.Numeric,
    max_value: chex.Numeric,
) -> chex.Array:
  """Returns Q-values normalized by the given `min_value` and `max_value`.

  Args:
    tree: _unbatched_ MCTS tree state.
    node_index: scalar index of the parent node.
    min_value: given minimum value. Usually the `min_value` is minimum possible
      untransformed Q-value.
    max_value: given maximum value. Usually the `max_value` is maximum possible
      untransformed Q-value.

  Returns:
    Q-values normalized by `(qvalues - min_value) / (max_value - min_value)`.
    The unvisited actions will have zero Q-value. Shape `[num_actions]`.
  """
  chex.assert_shape(node_index, ())
  qvalues = tree.qvalues(node_index)
  visit_counts = tree.children_visits[node_index]
  value_score = jnp.where(visit_counts > 0, qvalues, min_value)
  value_score = (value_score - min_value) / ((max_value - min_value))
  return value_score


def qtransform_by_parent_and_siblings(
    tree: tree_lib.Tree,
    node_index: chex.Numeric,
    *,
    epsilon: chex.Numeric = 1e-8,
) -> chex.Array:
  """Returns qvalues normalized by min, max over V(node) and qvalues.

  Args:
    tree: _unbatched_ MCTS tree state.
    node_index: scalar index of the parent node.
    epsilon: the minimum denominator for the normalization.

  Returns:
    Q-values normalized to be from the [0, 1] interval. The unvisited actions
    will have zero Q-value. Shape `[num_actions]`.
  """
  chex.assert_shape(node_index, ())
  qvalues = tree.qvalues(node_index)
  visit_counts = tree.children_visits[node_index]
  chex.assert_rank([qvalues, visit_counts, node_index], [1, 1, 0])
  node_value = tree.node_values[node_index]
  safe_qvalues = jnp.where(visit_counts > 0, qvalues, node_value)
  chex.assert_equal_shape([safe_qvalues, qvalues])
  min_value = jnp.minimum(node_value, jnp.min(safe_qvalues, axis=-1))
  max_value = jnp.maximum(node_value, jnp.max(safe_qvalues, axis=-1))

  completed_by_min = jnp.where(visit_counts > 0, qvalues, min_value)
  normalized = (completed_by_min - min_value) / (
      jnp.maximum(max_value - min_value, epsilon))
  chex.assert_equal_shape([normalized, qvalues])
  return normalized


def qtransform_completed_by_mix_value(
    tree: tree_lib.Tree,
    node_index: chex.Numeric,
    *,
    value_scale: chex.Numeric = 0.1,
    maxvisit_init: chex.Numeric = 50.0,
    rescale_values: bool = True,
    use_mixed_value: bool = True,
    epsilon: chex.Numeric = 1e-8,
    use_sqrt_scaling: bool = False,
    use_log_scaling: bool = False,
    use_normalized_advantages: bool = False,
) -> chex.Array:
  """Returns completed qvalues.

  The missing Q-values of the unvisited actions are replaced by the
  mixed value, defined in Appendix D of
  "Policy improvement by planning with Gumbel":
  https://openreview.net/forum?id=bERaNdoegnO

  The Q-values are transformed by a linear transformation:
    `(maxvisit_init + max(visit_counts)) * value_scale * qvalues`.

  Args:
    tree: _unbatched_ MCTS tree state.
    node_index: scalar index of the parent node.
    value_scale: scale for the Q-values.
    maxvisit_init: offset to the `max(visit_counts)` in the scaling factor.
    rescale_values: if True, scale the qvalues by `1 / (max_q - min_q)`.
    use_mixed_value: if True, complete the Q-values with mixed value,
      otherwise complete the Q-values with the raw value.
    epsilon: the minimum denominator when using `rescale_values`.

  Returns:
    Completed Q-values. Shape `[num_actions]`.
  """
  chex.assert_shape(node_index, ())
  qvalues = tree.qvalues(node_index)
  visit_counts = tree.children_visits[node_index]

  # Computing the mixed value and producing completed_qvalues.
  raw_value = tree.raw_values[node_index]
  prior_probs = jax.nn.softmax(
      tree.children_prior_logits[node_index])
  if use_mixed_value:
    value = _compute_mixed_value(
        raw_value,
        qvalues=qvalues,
        visit_counts=visit_counts,
        prior_probs=prior_probs)
  else:
    value = raw_value
  # jax.debug.print("[OG qtransform]@{}, qvalues: {}", node_index, qvalues)
  # jax.debug.print("[OG qtransform]@{}, visit_counts: {}", node_index, visit_counts)
  # jax.debug.print("[OG qtransform]@{}, value: {}", node_index, value)
  completed_qvalues = _complete_qvalues(
      qvalues, visit_counts=visit_counts, value=value)

  # Scaling the Q-values.
  if rescale_values:
    completed_qvalues = _rescale_qvalues(completed_qvalues, epsilon)


  max_visit = jnp.max(visit_counts, axis=-1)

  if use_log_scaling:
      # Use natural logarithm.
      # Add 1.0 inside just in case maxvisit_init is 0 to prevent log(0)
      visit_scale = jnp.log(maxvisit_init + max_visit + 1.0)
  elif use_sqrt_scaling:
      visit_scale = maxvisit_init + jnp.sqrt(max_visit)
  else:
      visit_scale = maxvisit_init + max_visit


  if use_normalized_advantages:
    # 1. Calculate raw advantages (Q(a) - V(s))
    advantages = completed_qvalues - value

    # 2. Normalize by standard deviation
    adv_std = jnp.std(advantages, axis=-1, keepdims=True)
    normalized_advantages = advantages / (adv_std + epsilon)

    # 3. Clip
    clipped_advantages = jnp.clip(normalized_advantages, -5.0, 5.0)

    return visit_scale * value_scale * clipped_advantages
  else:
    return visit_scale * value_scale * completed_qvalues


def _rescale_qvalues(qvalues, epsilon):
  """Rescales the given completed Q-values to be from the [0, 1] interval."""
  min_value = jnp.min(qvalues, axis=-1, keepdims=True)
  max_value = jnp.max(qvalues, axis=-1, keepdims=True)
  return (qvalues - min_value) / jnp.maximum(max_value - min_value, epsilon)


def _complete_qvalues(qvalues, *, visit_counts, value):
  """Returns completed Q-values, with the `value` for unvisited actions."""
  chex.assert_equal_shape([qvalues, visit_counts])
  chex.assert_shape(value, [])

  # The missing qvalues are replaced by the value.
  completed_qvalues = jnp.where(
      visit_counts > 0,
      qvalues,
      value)
  chex.assert_equal_shape([completed_qvalues, qvalues])
  return completed_qvalues


def _compute_mixed_value(raw_value, qvalues, visit_counts, prior_probs):
  """Interpolates the raw_value and weighted qvalues.

  Args:
    raw_value: an approximate value of the state. Shape `[]`.
    qvalues: Q-values for all actions. Shape `[num_actions]`. The unvisited
      actions have undefined Q-value.
    visit_counts: the visit counts for all actions. Shape `[num_actions]`.
    prior_probs: the action probabilities, produced by the policy network for
      each action. Shape `[num_actions]`.

  Returns:
    An estimator of the state value. Shape `[]`.
  """
  sum_visit_counts = jnp.sum(visit_counts, axis=-1)
  # Ensuring non-nan weighted_q, even if the visited actions have zero
  # prior probability.
  prior_probs = jnp.maximum(jnp.finfo(prior_probs.dtype).tiny, prior_probs)
  # Summing the probabilities of the visited actions.
  sum_probs = jnp.sum(jnp.where(visit_counts > 0, prior_probs, 0.0),
                      axis=-1)
  weighted_q = jnp.sum(jnp.where(
      visit_counts > 0,
      prior_probs * qvalues / jnp.where(visit_counts > 0, sum_probs, 1.0),
      0.0), axis=-1)
  return (raw_value + sum_visit_counts * weighted_q) / (sum_visit_counts + 1)
