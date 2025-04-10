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
"""Search policies."""
import functools
from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp

from mctx._src import action_selection
from mctx._src import base
from mctx._src import qtransforms
from mctx._src import search
from mctx._src import seq_halving


def muzero_policy(
    params: base.Params,
    rng_key: chex.PRNGKey,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    num_simulations: int,
    invalid_actions: Optional[chex.Array] = None,
    max_depth: Optional[int] = None,
    loop_fn: base.LoopFn = jax.lax.fori_loop,
    *,
    qtransform: base.QTransform = qtransforms.qtransform_by_parent_and_siblings,
    dirichlet_fraction: chex.Numeric = 0.25,
    dirichlet_alpha: chex.Numeric = 0.3,
    pb_c_init: chex.Numeric = 1.25,
    pb_c_base: chex.Numeric = 19652,
    temperature: chex.Numeric = 1.0) -> base.PolicyOutput[None]:
  """Runs MuZero search and returns the `PolicyOutput`.

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
    num_simulations: the number of simulations.
    invalid_actions: a mask with invalid actions. Invalid actions
      have ones, valid actions have zeros in the mask. Shape `[B, num_actions]`.
    max_depth: maximum search tree depth allowed during simulation.
    loop_fn: Function used to run the simulations. It may be required to pass
      hk.fori_loop if using this function inside a Haiku module.
    qtransform: function to obtain completed Q-values for a node.
    dirichlet_fraction: float from 0 to 1 interpolating between using only the
      prior policy or just the Dirichlet noise.
    dirichlet_alpha: concentration parameter to parametrize the Dirichlet
      distribution.
    pb_c_init: constant c_1 in the PUCT formula.
    pb_c_base: constant c_2 in the PUCT formula.
    temperature: temperature for acting proportionally to
      `visit_counts**(1 / temperature)`.

  Returns:
    `PolicyOutput` containing the proposed action, action_weights and the used
    search tree.
  """
  rng_key, dirichlet_rng_key, search_rng_key = jax.random.split(rng_key, 3)

  # Adding Dirichlet noise.
  noisy_logits = _get_logits_from_probs(
      _add_dirichlet_noise(
          dirichlet_rng_key,
          jax.nn.softmax(root.prior_logits),
          dirichlet_fraction=dirichlet_fraction,
          dirichlet_alpha=dirichlet_alpha))
  root = root.replace(
      prior_logits=_mask_invalid_actions(noisy_logits, invalid_actions))

  # Running the search.
  interior_action_selection_fn = functools.partial(
      action_selection.muzero_action_selection,
      pb_c_base=pb_c_base,
      pb_c_init=pb_c_init,
      qtransform=qtransform)
  root_action_selection_fn = functools.partial(
      interior_action_selection_fn,
      depth=0)
  search_tree = search.search(
      params=params,
      rng_key=search_rng_key,
      root=root,
      recurrent_fn=recurrent_fn,
      root_action_selection_fn=root_action_selection_fn,
      interior_action_selection_fn=interior_action_selection_fn,
      num_simulations=num_simulations,
      max_depth=max_depth,
      invalid_actions=invalid_actions,
      loop_fn=loop_fn)

  # Sampling the proposed action proportionally to the visit counts.
  summary = search_tree.summary()
  action_weights = summary.visit_probs
  action_logits = _apply_temperature(
      _get_logits_from_probs(action_weights), temperature)
  action = jax.random.categorical(rng_key, action_logits)
  return base.PolicyOutput(
      action=action,
      action_weights=action_weights,
      search_tree=search_tree)


def gumbel_muzero_policy_bfs3(
  params: base.Params,
  rng_key: chex.PRNGKey,
  root: base.RootFnOutput,
  recurrent_fn: base.RecurrentFn,
  *,
  num_simulations: chex.Numeric = 2, # [scrap] unused, kept to keep testing easier
  top_k_first: chex.Numeric = 2,
  top_k_second: chex.Numeric = 4, # [scrap] unused,
  gumbel_scale: chex.Numeric = 1.0,
  # invalid_actions: Optional[chex.Array] = None,
  invalid_actions: chex.Array = None,
  # Extra BFS Q transform parameters:
  value_scale: chex.Numeric = 0.1,
  maxvisit_init: chex.Numeric = 50.0,
  rescale_values: bool = True,
  epsilon: chex.Numeric = 1e-8
) -> base.PolicyOutput[None]:
  """
  Performs a 1-layer BFS:
    1) From the root, pick top_k_first actions by root_gumbel + root_logits + root_completed_qvalues.
        Expand them all in parallel.
    2) Combine that child Q with parent's reward, discount, etc. to get the BFS Q for each of the
        top_k_first actions.
    4) Choose the final root action by argmax of root_logits + root_gumbel + BFS Q.
  """
  # --- 1. Preprocess the root.
  # Mask out any invalid actions in the root logits.
  masked_root_logits = _mask_invalid_actions(root.prior_logits, invalid_actions)
  root = root.replace(prior_logits=masked_root_logits)

  chex.assert_rank(root.prior_logits, 2)
  batch_size, num_actions = root.prior_logits.shape

  # --- 2. Generate Gumbel noise (same shape as the logits).
  rng_key, gumbel_rng = jax.random.split(rng_key)
  root_gumbel = gumbel_scale * jax.random.gumbel(
      gumbel_rng, shape=root.prior_logits.shape, dtype=root.prior_logits.dtype)

  # --- 2.5 Compute initial qvalues to ensure
  # logits don't dominate exploration at root '
  #
  # Score = gumbel + logits + initial completed_qvalues
  root_raw_value = root.value # [B,]
  root_prior_logits = root.prior_logits # [B, num_actions]
  root_init_qvalues = jnp.zeros((batch_size, num_actions))
  root_children_visit_counts = jnp.zeros((batch_size, num_actions), dtype=jnp.int32)
  qtransform_fn = functools.partial(
      qtransforms.qtransform_completed_by_mix_value_bfs2,
      value_scale=value_scale,
      maxvisit_init=maxvisit_init,
      rescale_values=rescale_values,
      epsilon=epsilon,
  )
  root_init_completed_qvalues = jax.vmap(qtransform_fn, in_axes=[0, 0, 0, 0])(
      root_init_qvalues,
      root_raw_value,
      root.prior_logits,
      root_children_visit_counts,
  )
  chex.assert_equal_shape([root_init_completed_qvalues, root_prior_logits, root_children_visit_counts])

  score0 = root_gumbel + masked_root_logits + root_init_completed_qvalues

  # --- 3. Pick the top_k_first from the root
  topk_vals, topk_idx = jax.lax.top_k(score0, top_k_first)
    # shape of topk_idx is [batch_size, top_k_first]

  # --- 4. Expand each chosen action in parallel, calling recurrent_fn for every (batch, child).
  # Flatten them out to shape [B * top_k_first] so we can do one pass:
  BxK = batch_size * top_k_first
  rng_keys = jax.random.split(rng_key, BxK).reshape(batch_size, top_k_first, -1)

  flat_idx = topk_idx.reshape((BxK,))  # shape [B*K]
  # For the embeddings, replicate each root embedding top_k_first times:
  def replicate_leaf(x):
    # x has shape [B, ...], replicate each batch item top_k_first times.
    return jnp.repeat(x, top_k_first, axis=0)

  batched_root_embedding = jax.tree_map(replicate_leaf, root.embedding)

  # Now gather each action from flat_idx, pass to recurrent_fn
  flat_actions = flat_idx  # shape [B*K]
  flat_keys = rng_keys.reshape(BxK, rng_keys.shape[-1])

  # Recurrent function => (RecurrentFnOutput, new_embed)
  flat_outputs, flat_child_embed = recurrent_fn(
      params, flat_keys, flat_actions, batched_root_embedding
  )
  # Reshape back to [B, K, ...]
  def unflatten(x):
    return x.reshape(batch_size, top_k_first, *x.shape[1:])
  # child_outputs = jax.tree_map(unflatten, flat_outputs)
  # child_embeddings = jax.tree_map(unflatten, flat_child_embed)
  # # child_outputs.reward, child_outputs.value, child_outputs.prior_logits, child_outputs.discount




  # --- 8. Re-compute completed Q–values at root.
  # final_qvalues has shape [B, num_actions]
  layer1_outputs = jax.tree_map(unflatten, flat_outputs)
  layer1_qvalues = layer1_outputs.reward + layer1_outputs.discount * layer1_outputs.value # [B, K]

  batch_idx = jnp.arange(batch_size)[:, None]            # shape [B, 1]
  batch_idx = jnp.tile(batch_idx, (1, top_k_first))      # shape [B, K]
  chex.assert_equal_shape([batch_idx, topk_idx, layer1_qvalues]) # all [B, K]

  # ### [original]
  # root_qvalues = jnp.zeros((batch_size, num_actions))
  # root_qvalues = root_qvalues.at[batch_idx, topk_idx].set(layer1_qvalues)

  ### [optimized] Trick to avoid scatter op
  #  One-hot mask for [B, num_actions]
  #  Multiply each one-hot with its corresponding q-value
  mask_q = jax.nn.one_hot(topk_idx, num_actions)  # [B, K, A]
  root_qvalues = jnp.sum(mask_q * layer1_qvalues[:, :, None], axis=1)  # [B, A]

  root_raw_value = root.value # [B,]
  root_prior_logits = root.prior_logits # [B, num_actions]


  # Right now layer1_visits is shape [B, K], .
  # root children visit_counts will be [B, num_actions]
  layer1_visits = jnp.ones((batch_size, top_k_first), dtype=jnp.int32)

  # ### [original]
  # root_children_visit_counts = jnp.zeros((batch_size, num_actions), dtype=jnp.int32)
  # root_children_visit_counts = root_children_visit_counts.at[batch_idx, topk_idx].set(layer1_visits)

  ### [optimized]
  mask_v = jax.nn.one_hot(topk_idx, num_actions, dtype=jnp.int32)  # [B, K, A]
  root_children_visit_counts = jnp.sum(mask_v * layer1_visits[:, :, None], axis=1)  # [B, A]



  chex.assert_rank(root_qvalues, 2) # (B, num_actions)
  chex.assert_rank(root_raw_value, 1) # (B,)
  chex.assert_equal_shape([root_qvalues, root_prior_logits, root_children_visit_counts])
  qtransform_fn = functools.partial(
      qtransforms.qtransform_completed_by_mix_value_bfs2,
      value_scale=value_scale,
      maxvisit_init=maxvisit_init,
      rescale_values=rescale_values,
      epsilon=epsilon,
  )

  final_qvalues = jax.vmap(qtransform_fn, in_axes=[0, 0, 0, 0])(
      root_qvalues,
      root_raw_value,
      root_prior_logits,
      root_children_visit_counts,
  )

  # --- 9. Score and select action.
  score = root_gumbel + root.prior_logits + final_qvalues
  selected_action = action_selection.masked_argmax(score, invalid_actions)

  # Compute action weights for training.
  search_logits = root.prior_logits + final_qvalues # for debugging
  completed_search_logits = _mask_invalid_actions(search_logits, invalid_actions)
  action_weights = jax.nn.softmax(completed_search_logits)
  return base.PolicyOutput(
      action=selected_action,
      action_weights=action_weights,
      search_logits=search_logits,
      children_values=root_qvalues,
      root_gumbel=root_gumbel,
      root_prior_logits=root.prior_logits,
      final_qvalues=final_qvalues,
      final_score=score,
  )


def gumbel_muzero_policy_bfs2(
  params: base.Params,
  rng_key: chex.PRNGKey,
  root: base.RootFnOutput,
  recurrent_fn: base.RecurrentFn,
  *,
  num_simulations: chex.Numeric = 2, # [scrap] unused, kept to keep testing easier
  top_k_first: chex.Numeric = 2,
  top_k_second: chex.Numeric = 4,
  gumbel_scale: chex.Numeric = 1.0,
  # invalid_actions: Optional[chex.Array] = None,
  invalid_actions: chex.Array = None,
  # Extra BFS Q transform parameters:
  value_scale: chex.Numeric = 0.1,
  maxvisit_init: chex.Numeric = 50.0,
  rescale_values: bool = True,
  epsilon: chex.Numeric = 1e-8
) -> base.PolicyOutput[None]:
  """
  Performs a 2-layer BFS:
    1) From the root, pick top_k_first actions by root_gumbel + root_logits.
        Expand them all in parallel.
    2) For each of those children, pick top_k_second subactions via child_gumbel + child_logits,
        expand each in parallel, then produce a Q-value for that child by some rule (e.g. max).
    3) Combine that child Q with parent's reward, discount, etc. to get the BFS Q for each of the
        top_k_first actions.
    4) Choose the final root action by argmax of root_logits + root_gumbel + BFS Q.
  """
  # --- 1. Preprocess the root.
  # Mask out any invalid actions in the root logits.
  masked_root_logits = _mask_invalid_actions(root.prior_logits, invalid_actions)
  root = root.replace(prior_logits=masked_root_logits)

  chex.assert_rank(root.prior_logits, 2)
  batch_size, num_actions = root.prior_logits.shape

  # --- 2. Generate Gumbel noise (same shape as the logits).
  rng_key, gumbel_rng = jax.random.split(rng_key)
  root_gumbel = gumbel_scale * jax.random.gumbel(
      gumbel_rng, shape=root.prior_logits.shape, dtype=root.prior_logits.dtype)

  # --- 2.5 Compute initial qvalues to ensure
  # logits don't dominate exploration at root '
  #
  # Score = gumbel + logits + initial completed_qvalues
  root_raw_value = root.value # [B,]
  root_prior_logits = root.prior_logits # [B, num_actions]
  root_init_qvalues = jnp.zeros((batch_size, num_actions))
  root_children_visit_counts = jnp.zeros((batch_size, num_actions), dtype=jnp.int32)
  qtransform_fn = functools.partial(
      qtransforms.qtransform_completed_by_mix_value_bfs2,
      value_scale=value_scale,
      maxvisit_init=maxvisit_init,
      rescale_values=rescale_values,
      epsilon=epsilon,
  )
  root_init_completed_qvalues = jax.vmap(qtransform_fn, in_axes=[0, 0, 0, 0])(
      root_init_qvalues,
      root_raw_value,
      root.prior_logits,
      root_children_visit_counts,
  )
  chex.assert_equal_shape([root_init_completed_qvalues, root_prior_logits, root_children_visit_counts])

  score0 = root_gumbel + masked_root_logits + root_init_completed_qvalues

  # --- 3. Pick the top_k_first from the root
  topk_vals, topk_idx = jax.lax.top_k(score0, top_k_first)
    # shape of topk_idx is [batch_size, top_k_first]

  # --- 4. Expand each chosen action in parallel, calling recurrent_fn for every (batch, child).
  # Flatten them out to shape [B * top_k_first] so we can do one pass:
  BxK = batch_size * top_k_first
  rng_keys = jax.random.split(rng_key, BxK).reshape(batch_size, top_k_first, -1)

  flat_idx = topk_idx.reshape((BxK,))  # shape [B*K]
  # For the embeddings, replicate each root embedding top_k_first times:
  def replicate_leaf(x):
    # x has shape [B, ...], replicate each batch item top_k_first times.
    return jnp.repeat(x, top_k_first, axis=0)

  batched_root_embedding = jax.tree_map(replicate_leaf, root.embedding)

  # Now gather each action from flat_idx, pass to recurrent_fn
  flat_actions = flat_idx  # shape [B*K]
  flat_keys = rng_keys.reshape(BxK, rng_keys.shape[-1])

  # Recurrent function => (RecurrentFnOutput, new_embed)
  flat_outputs, flat_child_embed = recurrent_fn(
      params, flat_keys, flat_actions, batched_root_embedding
  )
  # Reshape back to [B, K, ...]
  def unflatten(x):
    return x.reshape(batch_size, top_k_first, *x.shape[1:])
  # child_outputs = jax.tree_map(unflatten, flat_outputs)
  # child_embeddings = jax.tree_map(unflatten, flat_child_embed)
  # # child_outputs.reward, child_outputs.value, child_outputs.prior_logits, child_outputs.discount

  layer1_outputs = jax.tree_map(unflatten, flat_outputs)
  layer1_embeddings = jax.tree_map(unflatten, flat_child_embed)


  # --- 5. Now for each of those (B, top_k_first) children, pick the top_k_second subactions:
  #     For each child, we have child_outputs.prior_logits shaped [B, K, num_actions].
  #     We'll sample a new Gumbel for each (B, K, A), apply invalid-actions mask,
  #     then do a top_k to expand them.

  # sample Gumbel for second layer
  rng_key, gumbel2_rng = jax.random.split(rng_key)
  second_gumbel = gumbel_scale * jax.random.gumbel(
      gumbel2_rng,
      shape=(batch_size, top_k_first, num_actions),
      dtype=root.prior_logits.dtype
  )

  # Extract child logits: shape [B, K, A].
  # We assume layer 1's prior_logits have invalid_actions masked out'
  child_logits = layer1_outputs.prior_logits

  # Now add the Gumbel noise to the masked logits
  second_score = child_logits + second_gumbel

  # Finally pick top_k_second for each [B, K] child, shape => [B, K, top_k_second].
  topk2_vals, topk2_idx = jax.lax.top_k(second_score, top_k_second)

  # --- 6. Expand each second-layer child in parallel. Flatten from [B, K, top_k_second].
  BxKxK2 = batch_size * top_k_first * top_k_second
  rng_keys2 = jax.random.split(rng_key, BxKxK2).reshape(batch_size, top_k_first, top_k_second, -1)

  flat_idx2 = topk2_idx.reshape((BxKxK2,))
  # replicate embeddings:
  def replicate_child_leaf(x):
    # x has shape [B, K, ...]; we replicate each [B, K] entry top_k_second times
    # first flatten => shape [B*K, ...], then repeat top_k_second times
    x_flat = x.reshape((batch_size*top_k_first,) + x.shape[2:])
    return jnp.repeat(x_flat, top_k_second, axis=0)

  layer1_x_k2_embedding = jax.tree_map(replicate_child_leaf, layer1_embeddings)
  # batched_child_embedding = jax.tree_map(replicate_child_leaf, first_layer_embeddings)

  # flatten out rng keys:
  flat_keys2 = rng_keys2.reshape(BxKxK2, rng_keys2.shape[-1])
  flat_actions2 = flat_idx2  # shape [B*K*K2]

  # Call recurrent_fn for second layer expansions
  flat2_outputs, _ = recurrent_fn(
      params, flat_keys2, flat_actions2, layer1_x_k2_embedding
  )

  # shape [B*K*K2, ...], now unflatten to [B, K, K2, ...]
  def unflatten2(x):
    return x.reshape((batch_size, top_k_first, top_k_second) + x.shape[1:])

  layer2_outputs = jax.tree_map(unflatten2, flat2_outputs)
  # e.g. second_layer_out.value: shape [B, K, K2]

  # --- 7. Compute layer 2 leaf values
  # Two cases to consider for each layer
  # 1) Expanding when using illegal action / on an illegal state
  # - layer1 might have been expanding using illegal action (e.g only one move left)
  # - layer2 might expand on an "illegal" leaf (this is not fine)
  # 2) Expanding when terminal node
  # - layer1's node may be a terminal leaf (win / lose)
  # - layer2 might expand upon said terminal leaf (this is fine as it'll backprop terminal path more)
  layer2_values = layer2_outputs.reward + layer2_outputs.discount * layer2_outputs.value # [B, K, K2]

  # From layer1, we may take some illegal actions
  # In which case, those paths in layer 2 should be masked out
  # Arrays have shape [B, K, num_actions]
  layer1_illegal_actions = jnp.isneginf(layer1_outputs.prior_logits) # [B, K, num_actions], we assume logits are always masked

  # layer1_illegal_actions is [B, K, A] (True where illegal).
  # topk2_idx is [B, K, top_k2], each entry in [0, A).
  # We gather the "illegal" boolean for each chosen child action:
  layer2_illegal_mask = jnp.take_along_axis(
      layer1_illegal_actions,
      topk2_idx,  # shape [B, K, top_k2]
      axis=-1     # gather along the action dimension
  )

  # Now layer2_illegal_mask is shape [B, K, top_k2].
  # If layer2_illegal_mask[b, k, x] is True, it means "the x-th chosen
  # action from child k of batch element b was illegal."
  layer2_values = jnp.where(~layer2_illegal_mask, layer2_values, 0.0) # [B, K, K2]
  layer2_total_value = jnp.sum(layer2_values, axis=-1) # [B, K]
  layer2_visits = jnp.sum(~layer2_illegal_mask, axis=-1) # calculate how many valid visits to layer 2 from each layer 1 embedding [B, K]

  # Update layer1's value with its layer2 values (backwards step)
  layer1_backward_value = (
    (layer1_outputs.value * layer2_visits + layer2_total_value) / (layer2_visits + 1.0)
  )
  layer1_values = (
    layer1_outputs.reward + layer1_outputs.discount * layer1_backward_value
  )
  layer1_visits = layer2_visits + 1

  # --- 8. Re-compute completed Q–values at root.
  # final_qvalues has shape [B, num_actions]
  root_qvalues = jnp.zeros((batch_size, num_actions))
  batch_idx = jnp.arange(batch_size)[:, None]            # shape [B, 1]
  batch_idx = jnp.tile(batch_idx, (1, top_k_first))      # shape [B, K]
  chex.assert_equal_shape([batch_idx, topk_idx, layer1_values]) # all [B, K]

  root_qvalues = root_qvalues.at[batch_idx, topk_idx].set(layer1_values)
  root_raw_value = root.value # [B,]
  root_prior_logits = root.prior_logits # [B, num_actions]

  # Right now layer1_visits is shape [B, K], but we want shape [B, A].
  layer1_visit_counts = jnp.zeros((batch_size, num_actions), dtype=jnp.int32)
  layer1_visit_counts = layer1_visit_counts.at[batch_idx, topk_idx].set(layer1_visits)

  chex.assert_rank(root_qvalues, 2) # (B, num_actions)
  chex.assert_rank(root_raw_value, 1) # (B,)

  chex.assert_equal_shape([root_qvalues, root_prior_logits, layer1_visit_counts])
  qtransform_fn = functools.partial(
      qtransforms.qtransform_completed_by_mix_value_bfs2,
      value_scale=value_scale,
      maxvisit_init=maxvisit_init,
      rescale_values=rescale_values,
      epsilon=epsilon,
  )

  final_qvalues = jax.vmap(qtransform_fn, in_axes=[0, 0, 0, 0])(
      root_qvalues,
      root_raw_value,
      root_prior_logits,
      layer1_visit_counts,
  )

  # --- 9. Score and select action.
  score = root_gumbel + root.prior_logits + final_qvalues
  selected_action = action_selection.masked_argmax(score, invalid_actions)

  # Compute action weights for training.
  search_logits = root.prior_logits + final_qvalues # for debugging
  completed_search_logits = _mask_invalid_actions(search_logits, invalid_actions)
  action_weights = jax.nn.softmax(completed_search_logits)
  return base.PolicyOutput(
      action=selected_action,
      action_weights=action_weights,
      search_logits=search_logits,
      children_values=root_qvalues,
      root_gumbel=root_gumbel,
      root_prior_logits=root.prior_logits,
      final_qvalues=final_qvalues,
      final_score=score,
  )

def gumbel_muzero_policy_bfs(
    params: base.Params,
    rng_key: chex.PRNGKey,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    num_simulations: int,
    *,
    invalid_actions: Optional[chex.Array] = None,
    gumbel_scale: chex.Numeric = 1.0,
    value_scale: chex.Numeric = 0.1,
    maxvisit_init: chex.Numeric = 50.0,
    rescale_values: bool = True,
    epsilon: chex.Numeric = 1e-8
) -> base.PolicyOutput[action_selection.GumbelMuZeroExtraData]:
  """Optimized Gumbel MuZero policy for num_simulations=2 via a parallel BFS expansion.

  This version assumes only one “expansion” level is needed, so that all actions from the
  root are expanded in parallel. This eliminates loops and is TPU‐friendly.
  """
  # --- 1. Preprocess the root.
  # Mask out any invalid actions in the root logits.
  root = root.replace(
      prior_logits=_mask_invalid_actions(root.prior_logits, invalid_actions))

  chex.assert_rank(root.prior_logits, 2)
  batch_size, num_actions = root.prior_logits.shape

  # --- 2. Generate Gumbel noise (same shape as the logits).
  rng_key, gumbel_rng = jax.random.split(rng_key)
  gumbel = gumbel_scale * jax.random.gumbel(
      gumbel_rng, shape=root.prior_logits.shape, dtype=root.prior_logits.dtype)

  # --- 3. Expand all actions from the root in parallel.
  # Create an array of actions: shape [num_actions]
  actions = jnp.arange(num_actions, dtype=jnp.int32)

  # For each batch element, we want to apply the recurrent function to every action.
  # We need to provide a separate rng_key per (batch, action) pair.
  total_keys = batch_size * num_actions
  rng_keys = jax.random.split(rng_key, total_keys).reshape(batch_size, num_actions, -1)

  # opt 3
  def expand_all_actions_flat(params, recurrent_fn, root_embedding, rng_keys, actions, batch_size, num_actions):
    """
    Merges the batch dimension (B) and the number of actions (N), calls recurrent_fn
    once over the merged batch, and then reshapes the outputs back to [B, N, ...].

    Args:
    params: parameters for recurrent_fn.
    recurrent_fn: function with signature (params, rng_key, action, embedding)
                    that expects a batched embedding with shape [B, ...].
    root_embedding: the embeddings from the root, a pytree (e.g. a dataclass)
                    whose array leaves have shape [B, ...].
    rng_keys: an array of RNG keys with shape [B, N, key_dim].
    actions: a 1D array of actions of shape [N].
    batch_size: number of examples (B).
    num_actions: number of actions (N).

    Returns:
    outputs: a pytree of outputs with shape [B, N, ...].
    new_embedding: a pytree of new embeddings with shape [B, N, ...].
    """
    # Flatten actions: shape [B * N]
    flat_actions = jnp.broadcast_to(actions, (batch_size, num_actions)).reshape(-1)

    # Flatten the rng_keys from [B, N, key_dim] to [B*N, key_dim].
    flat_keys = rng_keys.reshape(-1, rng_keys.shape[-1])

    # Replicate the embedding for each action. If root_embedding is a pytree,
    # we use jax.tree_map to replicate each array leaf.
    def replicate_leaf(x):
        # x has shape [B, ...]; we want each batch element repeated N times along axis 0.
        return jnp.repeat(x, num_actions, axis=0)
    flat_embedding = jax.tree_map(replicate_leaf, root_embedding)

    # Call recurrent_fn once over the flattened (B*N) dimension.
    flat_outputs, flat_new_embedding = recurrent_fn(params, flat_keys, flat_actions, flat_embedding)

    # Reshape outputs back to [B, N, ...]. We do this for every array leaf.
    def unflatten(x):
      return x.reshape((batch_size, num_actions) + x.shape[1:])
    outputs = jax.tree_map(unflatten, flat_outputs)
    new_embedding = jax.tree_map(unflatten, flat_new_embedding)

    return outputs, new_embedding

  # children_outputs contains: reward, discount, value for each [B, num_actions].
  children_outputs, _ = expand_all_actions_flat(
      params,           # your parameters
      recurrent_fn,     # your recurrent function
      root.embedding,   # batched embedding, shape: [B, ...]
      rng_keys,         # shape: [B, num_actions, key_dim]
      actions,          # shape: [num_actions]
      batch_size,
      num_actions
  )

  # --- 4. Compute completed Q–values directly.
  # final_qvalues has shape [B, num_actions]
  final_qvalues = qtransforms.compute_bfs_completed_qvalues(
    children_outputs,
    root,
    value_scale=value_scale,
    maxvisit_init=maxvisit_init,
    rescale_values=rescale_values,
    epsilon=epsilon,
  )

  # --- 5. Score and select action.
  score = gumbel + root.prior_logits + final_qvalues
  selected_action = action_selection.masked_argmax(score, invalid_actions)

  # Compute action weights for training.
  search_logits = root.prior_logits + final_qvalues # for debugging
  completed_search_logits = _mask_invalid_actions(search_logits, invalid_actions)
  action_weights = jax.nn.softmax(completed_search_logits)
  return base.PolicyOutput(
      action=selected_action,
      action_weights=action_weights,
      search_logits=search_logits,
      children_values=children_outputs.value,
      root_gumbel=gumbel,
      root_prior_logits=root.prior_logits,
      final_qvalues=final_qvalues,
      final_score=score,
  )

def gumbel_muzero_policy(
    params: base.Params,
    rng_key: chex.PRNGKey,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    num_simulations: int,
    invalid_actions: Optional[chex.Array] = None,
    max_depth: Optional[int] = None,
    loop_fn: base.LoopFn = jax.lax.fori_loop,
    *,
    qtransform: base.QTransform = qtransforms.qtransform_completed_by_mix_value,
    max_num_considered_actions: int = 16,
    gumbel_scale: chex.Numeric = 1.,
) -> base.PolicyOutput[action_selection.GumbelMuZeroExtraData]:
  """Runs Gumbel MuZero search and returns the `PolicyOutput`.

  This policy implements Full Gumbel MuZero from
  "Policy improvement by planning with Gumbel".
  https://openreview.net/forum?id=bERaNdoegnO

  At the root of the search tree, actions are selected by Sequential Halving
  with Gumbel. At non-root nodes (aka interior nodes), actions are selected by
  the Full Gumbel MuZero deterministic action selection.

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
    num_simulations: the number of simulations.
    invalid_actions: a mask with invalid actions. Invalid actions
      have ones, valid actions have zeros in the mask. Shape `[B, num_actions]`.
    max_depth: maximum search tree depth allowed during simulation.
    loop_fn: Function used to run the simulations. It may be required to pass
      hk.fori_loop if using this function inside a Haiku module.
    qtransform: function to obtain completed Q-values for a node.
    max_num_considered_actions: the maximum number of actions expanded at the
      root node. A smaller number of actions will be expanded if the number of
      valid actions is smaller.
    gumbel_scale: scale for the Gumbel noise. Evalution on perfect-information
      games can use gumbel_scale=0.0.

  Returns:
    `PolicyOutput` containing the proposed action, action_weights and the used
    search tree.
  """
  # Masking invalid actions.
  root = root.replace(
      prior_logits=_mask_invalid_actions(root.prior_logits, invalid_actions))

  # Generating Gumbel.
  rng_key, gumbel_rng = jax.random.split(rng_key)
  gumbel = gumbel_scale * jax.random.gumbel(
      gumbel_rng, shape=root.prior_logits.shape, dtype=root.prior_logits.dtype)

  # Searching.
  extra_data = action_selection.GumbelMuZeroExtraData(root_gumbel=gumbel)
  search_tree = search.search(
      params=params,
      rng_key=rng_key,
      root=root,
      recurrent_fn=recurrent_fn,
      root_action_selection_fn=functools.partial(
          action_selection.gumbel_muzero_root_action_selection,
          num_simulations=num_simulations,
          max_num_considered_actions=max_num_considered_actions,
          qtransform=qtransform,
      ),
      interior_action_selection_fn=functools.partial(
          action_selection.gumbel_muzero_interior_action_selection,
          qtransform=qtransform,
      ),
      num_simulations=num_simulations,
      max_depth=max_depth,
      invalid_actions=invalid_actions,
      extra_data=extra_data,
      loop_fn=loop_fn)
  summary = search_tree.summary()

  # Acting with the best action from the most visited actions.
  # The "best" action has the highest `gumbel + logits + q`.
  # Inside the minibatch, the considered_visit can be different on states with
  # a smaller number of valid actions.
  considered_visit = jnp.max(summary.visit_counts, axis=-1, keepdims=True)
  # The completed_qvalues include imputed values for unvisited actions.
  completed_qvalues = jax.vmap(qtransform, in_axes=[0, None])(  # pytype: disable=wrong-arg-types  # numpy-scalars  # pylint: disable=line-too-long
      search_tree, search_tree.ROOT_INDEX)
  to_argmax = seq_halving.score_considered(
      considered_visit, gumbel, root.prior_logits, completed_qvalues,
      summary.visit_counts)
  action = action_selection.masked_argmax(to_argmax, invalid_actions)




  # Update the search_tree with completed_qvalues and to_argmax
  search_tree = search_tree.replace(
      completed_qvalues=completed_qvalues,
      to_argmax=to_argmax
  )

  # Producing action_weights usable to train the policy network.
  completed_search_logits = _mask_invalid_actions(
      root.prior_logits + completed_qvalues, invalid_actions)

  action_weights = jax.nn.softmax(completed_search_logits)
  # return base.PolicyOutput(
  #     action=action,
  #     action_weights=action_weights,
  #     search_tree=search_tree)

  # [bfs] for debugging
  search_logits= root.prior_logits + completed_qvalues # for debugging
  children_indices = search_tree.children_index[:, 0]  # [B, num_actions]
  children_values = jnp.take_along_axis(search_tree.node_values, children_indices, axis=1)  # [B, num_actions]

  return base.PolicyOutput(
      action=action,
      action_weights=action_weights,
      search_tree=search_tree,
      search_logits=search_logits,
      children_values=children_values,
      root_gumbel=gumbel,
      root_prior_logits=root.prior_logits,
      final_qvalues=completed_qvalues,
      final_score=to_argmax,
  )


def stochastic_muzero_policy(
    params: chex.ArrayTree,
    rng_key: chex.PRNGKey,
    root: base.RootFnOutput,
    decision_recurrent_fn: base.DecisionRecurrentFn,
    chance_recurrent_fn: base.ChanceRecurrentFn,
    num_simulations: int,
    invalid_actions: Optional[chex.Array] = None,
    max_depth: Optional[int] = None,
    loop_fn: base.LoopFn = jax.lax.fori_loop,
    *,
    qtransform: base.QTransform = qtransforms.qtransform_by_parent_and_siblings,
    dirichlet_fraction: chex.Numeric = 0.25,
    dirichlet_alpha: chex.Numeric = 0.3,
    pb_c_init: chex.Numeric = 1.25,
    pb_c_base: chex.Numeric = 19652,
    temperature: chex.Numeric = 1.0) -> base.PolicyOutput[None]:
  """Runs Stochastic MuZero search.

  Implements search as described in the Stochastic MuZero paper:
    (https://openreview.net/forum?id=X6D9bAHhBQ1).

  In the shape descriptions, `B` denotes the batch dimension.
  Args:
    params: params to be forwarded to root and recurrent functions.
    rng_key: random number generator state, the key is consumed.
    root: a `(prior_logits, value, embedding)` `RootFnOutput`. The
      `prior_logits` are from a policy network. The shapes are `([B,
      num_actions], [B], [B, ...])`, respectively.
    decision_recurrent_fn: a callable to be called on the leaf decision nodes
      and unvisited actions retrieved by the simulation step, which takes as
      args `(params, rng_key, action, state_embedding)` and returns a
      `(DecisionRecurrentFnOutput, afterstate_embedding)`.
    chance_recurrent_fn:  a callable to be called on the leaf chance nodes and
      unvisited actions retrieved by the simulation step, which takes as args
      `(params, rng_key, chance_outcome, afterstate_embedding)` and returns a
      `(ChanceRecurrentFnOutput, state_embedding)`.
    num_simulations: the number of simulations.
    invalid_actions: a mask with invalid actions. Invalid actions have ones,
      valid actions have zeros in the mask. Shape `[B, num_actions]`.
    max_depth: maximum search tree depth allowed during simulation.
    loop_fn: Function used to run the simulations. It may be required to pass
      hk.fori_loop if using this function inside a Haiku module.
    qtransform: function to obtain completed Q-values for a node.
    dirichlet_fraction: float from 0 to 1 interpolating between using only the
      prior policy or just the Dirichlet noise.
    dirichlet_alpha: concentration parameter to parametrize the Dirichlet
      distribution.
    pb_c_init: constant c_1 in the PUCT formula.
    pb_c_base: constant c_2 in the PUCT formula.
    temperature: temperature for acting proportionally to `visit_counts**(1 /
      temperature)`.

  Returns:
    `PolicyOutput` containing the proposed action, action_weights and the used
    search tree.
  """

  num_actions = root.prior_logits.shape[-1]

  rng_key, dirichlet_rng_key, search_rng_key = jax.random.split(rng_key, 3)

  # Adding Dirichlet noise.
  noisy_logits = _get_logits_from_probs(
      _add_dirichlet_noise(
          dirichlet_rng_key,
          jax.nn.softmax(root.prior_logits),
          dirichlet_fraction=dirichlet_fraction,
          dirichlet_alpha=dirichlet_alpha))

  root = root.replace(
      prior_logits=_mask_invalid_actions(noisy_logits, invalid_actions))

  # construct a dummy afterstate embedding
  batch_size = jax.tree_util.tree_leaves(root.embedding)[0].shape[0]
  dummy_action = jnp.zeros([batch_size], dtype=jnp.int32)
  dummy_output, dummy_afterstate_embedding = decision_recurrent_fn(
      params, rng_key, dummy_action, root.embedding)
  num_chance_outcomes = dummy_output.chance_logits.shape[-1]

  root = root.replace(
      # pad action logits with num_chance_outcomes so dim is A + C
      prior_logits=jnp.concatenate([
          root.prior_logits,
          jnp.full([batch_size, num_chance_outcomes], fill_value=-jnp.inf)
      ], axis=-1),
      # replace embedding with wrapper.
      embedding=base.StochasticRecurrentState(
          state_embedding=root.embedding,
          afterstate_embedding=dummy_afterstate_embedding,
          is_decision_node=jnp.ones([batch_size], dtype=bool)))

  # Stochastic MuZero Change: We need to be able to tell if different nodes are
  # decision or chance. This is accomplished by imposing a special structure
  # on the embeddings stored in each node. Each embedding is an instance of
  # StochasticRecurrentState which maintains this information.
  recurrent_fn = _make_stochastic_recurrent_fn(
      decision_node_fn=decision_recurrent_fn,
      chance_node_fn=chance_recurrent_fn,
      num_actions=num_actions,
      num_chance_outcomes=num_chance_outcomes,
  )

  # Running the search.

  interior_decision_node_selection_fn = functools.partial(
      action_selection.muzero_action_selection,
      pb_c_base=pb_c_base,
      pb_c_init=pb_c_init,
      qtransform=qtransform)

  interior_action_selection_fn = _make_stochastic_action_selection_fn(
      interior_decision_node_selection_fn, num_actions)

  root_action_selection_fn = functools.partial(
      interior_action_selection_fn, depth=0)

  search_tree = search.search(
      params=params,
      rng_key=search_rng_key,
      root=root,
      recurrent_fn=recurrent_fn,
      root_action_selection_fn=root_action_selection_fn,
      interior_action_selection_fn=interior_action_selection_fn,
      num_simulations=num_simulations,
      max_depth=max_depth,
      invalid_actions=invalid_actions,
      loop_fn=loop_fn)

  # Sampling the proposed action proportionally to the visit counts.
  search_tree = _mask_tree(search_tree, num_actions, 'decision')
  summary = search_tree.summary()
  action_weights = summary.visit_probs
  action_logits = _apply_temperature(
      _get_logits_from_probs(action_weights), temperature)
  action = jax.random.categorical(rng_key, action_logits)
  return base.PolicyOutput(
      action=action, action_weights=action_weights, search_tree=search_tree)


def _mask_invalid_actions(logits, invalid_actions):
  """Returns logits with zero mass to invalid actions."""
  if invalid_actions is None:
    return logits
  chex.assert_equal_shape([logits, invalid_actions])
  logits = logits - jnp.max(logits, axis=-1, keepdims=True)
  # At the end of an episode, all actions can be invalid. A softmax would then
  # produce NaNs, if using -inf for the logits. We avoid the NaNs by using
  # a finite `min_logit` for the invalid actions.
  min_logit = jnp.finfo(logits.dtype).min
  return jnp.where(invalid_actions, min_logit, logits)


def _get_logits_from_probs(probs):
  tiny = jnp.finfo(probs.dtype).tiny
  return jnp.log(jnp.maximum(probs, tiny))


def _add_dirichlet_noise(rng_key, probs, *, dirichlet_alpha,
                         dirichlet_fraction):
  """Mixes the probs with Dirichlet noise."""
  chex.assert_rank(probs, 2)
  chex.assert_type([dirichlet_alpha, dirichlet_fraction], float)

  batch_size, num_actions = probs.shape
  noise = jax.random.dirichlet(
      rng_key,
      alpha=jnp.full([num_actions], fill_value=dirichlet_alpha),
      shape=(batch_size,))
  noisy_probs = (1 - dirichlet_fraction) * probs + dirichlet_fraction * noise
  return noisy_probs


def _apply_temperature(logits, temperature):
  """Returns `logits / temperature`, supporting also temperature=0."""
  # The max subtraction prevents +inf after dividing by a small temperature.
  logits = logits - jnp.max(logits, keepdims=True, axis=-1)
  tiny = jnp.finfo(logits.dtype).tiny
  return logits / jnp.maximum(tiny, temperature)


def _make_stochastic_recurrent_fn(
    decision_node_fn: base.DecisionRecurrentFn,
    chance_node_fn: base.ChanceRecurrentFn,
    num_actions: int,
    num_chance_outcomes: int,
) -> base.RecurrentFn:
  """Make Stochastic Recurrent Fn."""

  def stochastic_recurrent_fn(
      params: base.Params,
      rng: chex.PRNGKey,
      action_or_chance: base.Action,  # [B]
      state: base.StochasticRecurrentState
  ) -> Tuple[base.RecurrentFnOutput, base.StochasticRecurrentState]:
    batch_size = jax.tree_util.tree_leaves(state.state_embedding)[0].shape[0]
    # Internally we assume that there are `A' = A + C` "actions";
    # action_or_chance can take on values in `{0, 1, ..., A' - 1}`,.
    # To interpret it as an action we can leave it as is:
    action = action_or_chance - 0
    # To interpret it as a chance outcome we subtract num_actions:
    chance_outcome = action_or_chance - num_actions

    decision_output, afterstate_embedding = decision_node_fn(
        params, rng, action, state.state_embedding)
    # Outputs from DecisionRecurrentFunction produce chance logits with
    # dim `C`, to respect our internal convention that there are `A' = A + C`
    # "actions" we pad with `A` dummy logits which are ultimately ignored:
    # see `_mask_tree`.
    output_if_decision_node = base.RecurrentFnOutput(
        prior_logits=jnp.concatenate([
            jnp.full([batch_size, num_actions], fill_value=-jnp.inf),
            decision_output.chance_logits], axis=-1),
        value=decision_output.afterstate_value,
        reward=jnp.zeros_like(decision_output.afterstate_value),
        discount=jnp.ones_like(decision_output.afterstate_value))

    chance_output, state_embedding = chance_node_fn(params, rng, chance_outcome,
                                                    state.afterstate_embedding)
    # Outputs from ChanceRecurrentFunction produce action logits with dim `A`,
    # to respect our internal convention that there are `A' = A + C` "actions"
    # we pad with `C` dummy logits which are ultimately ignored: see
    # `_mask_tree`.
    output_if_chance_node = base.RecurrentFnOutput(
        prior_logits=jnp.concatenate([
            chance_output.action_logits,
            jnp.full([batch_size, num_chance_outcomes], fill_value=-jnp.inf)
            ], axis=-1),
        value=chance_output.value,
        reward=chance_output.reward,
        discount=chance_output.discount)

    new_state = base.StochasticRecurrentState(
        state_embedding=state_embedding,
        afterstate_embedding=afterstate_embedding,
        is_decision_node=jnp.logical_not(state.is_decision_node))

    def _broadcast_where(decision_leaf, chance_leaf):
      extra_dims = [1] * (len(decision_leaf.shape) - 1)
      expanded_is_decision = jnp.reshape(state.is_decision_node,
                                         [-1] + extra_dims)
      return jnp.where(
          # ensure state.is_decision node has appropriate shape.
          expanded_is_decision,
          decision_leaf, chance_leaf)

    output = jax.tree.map(_broadcast_where,
                          output_if_decision_node,
                          output_if_chance_node)
    return output, new_state

  return stochastic_recurrent_fn


def _mask_tree(tree: search.Tree, num_actions: int, mode: str) -> search.Tree:
  """Masks out parts of the tree based upon node type.

  "Actions" in our tree can either be action or chance values: A' = A + C. This
  utility function masks the parts of the tree containing dimensions of shape
  A' to be either A or C depending upon `mode`.

  Args:
    tree: The tree to be masked.
    num_actions: The number of environment actions A.
    mode: Either "decision" or "chance".

  Returns:
    An appropriately masked tree.
  """

  def _take_slice(x):
    if mode == 'decision':
      return x[..., :num_actions]
    elif mode == 'chance':
      return x[..., num_actions:]
    else:
      raise ValueError(f'Unknown mode: {mode}.')

  return tree.replace(
      children_index=_take_slice(tree.children_index),
      children_prior_logits=_take_slice(tree.children_prior_logits),
      children_visits=_take_slice(tree.children_visits),
      children_rewards=_take_slice(tree.children_rewards),
      children_discounts=_take_slice(tree.children_discounts),
      children_values=_take_slice(tree.children_values),
      root_invalid_actions=_take_slice(tree.root_invalid_actions))


def _make_stochastic_action_selection_fn(
    decision_node_selection_fn: base.InteriorActionSelectionFn,
    num_actions: int,
) -> base.InteriorActionSelectionFn:
  """Make Stochastic Action Selection Fn."""

  # NOTE: trees are unbatched here.

  def _chance_node_selection_fn(
      tree: search.Tree,
      node_index: chex.Array,
  ) -> chex.Array:
    num_chance = tree.children_visits[node_index]
    chance_logits = tree.children_prior_logits[node_index]
    prob_chance = jax.nn.softmax(chance_logits)
    argmax_chance = jnp.argmax(prob_chance / (num_chance + 1), axis=-1).astype(
        jnp.int32
    )
    return argmax_chance

  def _action_selection_fn(key: chex.PRNGKey, tree: search.Tree,
                           node_index: chex.Array,
                           depth: chex.Array) -> chex.Array:
    is_decision = tree.embeddings.is_decision_node[node_index]
    chance_selection = _chance_node_selection_fn(
        tree=_mask_tree(tree, num_actions, 'chance'),
        node_index=node_index) + num_actions
    decision_selection = decision_node_selection_fn(
        key, _mask_tree(tree, num_actions, 'decision'), node_index, depth)
    return jax.lax.cond(is_decision, lambda: decision_selection,
                        lambda: chance_selection)

  return _action_selection_fn
