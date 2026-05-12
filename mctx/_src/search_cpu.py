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
"""CPU-resident MCTS with accelerator-offloaded recurrent_fn.

The tree lives in CPU device memory so that random-access gather/scatter
operations (which are slow on TPU) run at CPU speed.  Only recurrent_fn —
the neural-net forward pass — is dispatched to the accelerator.

Because jax.lax.while_loop is a single-device XLA primitive, we replace it
with a plain Python for-loop.  The per-iteration Python overhead is negligible
relative to the neural-net call for the large simulation counts that motivate
this module (thousands of simulations per move).

Typical usage with data-parallel inference across all TPUs/GPUs::

    mesh = jax.sharding.Mesh(jax.devices(), 'x')
    acc_sh = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('x'))

    # params must already be on the accelerator with whatever sharding you use.
    tree = mctx.search_cpu(
        params, rng_key,
        root=root, recurrent_fn=recurrent_fn, ...,
        accelerator_sharding=acc_sh)
"""
from typing import Any, Optional

import chex
import jax
import jax.numpy as jnp

from mctx._src import action_selection
from mctx._src import base
from mctx._src import tree as tree_lib
from mctx._src.search import backward
from mctx._src.search import expand
from mctx._src.search import instantiate_tree_from_root
from mctx._src.search import simulate

Tree = tree_lib.Tree


def search_cpu(
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
    accelerator_sharding=None,
) -> Tree:
  """Like search(), but keeps the tree on CPU and runs recurrent_fn on the accelerator.

  All tree operations (simulate, expand tree-update, backward) run on CPU where
  random-access indexing is fast.  recurrent_fn is dispatched to the accelerator
  each simulation step.  Inputs (action, embedding) are transferred
  CPU→accelerator, and outputs are declared with out_shardings=cpu so XLA
  schedules the transfer back to CPU and can overlap it with computation.

  Args:
    params: model parameters.  Must already be placed on the accelerator with
      whatever sharding you use — search_cpu never re-shards them.
    rng_key: PRNG key; consumed.
    root: root node outputs (prior_logits, value, embedding).
    recurrent_fn: the model; called once per simulation on the accelerator.
    root_action_selection_fn: action selector at the root node.
    interior_action_selection_fn: action selector at interior nodes.
    num_simulations: number of MCTS simulations to run.
    max_depth: maximum search depth (defaults to num_simulations).
    invalid_actions: boolean mask of invalid actions at the root, shape [B, A].
    extra_data: arbitrary pytree stored in tree.extra_data; passed through as-is.
    accelerator_sharding: a jax.sharding.Sharding that describes how to place
      action/embedding on the accelerator(s).  Defaults to
      SingleDeviceSharding(jax.devices()[0]).  For data-parallel inference pass
      e.g. NamedSharding(Mesh(jax.devices(), 'x'), PartitionSpec('x')).

  Returns:
    The final Tree after all simulations, with arrays on the CPU device.
  """
  cpu = jax.devices('cpu')[0]
  # Regular CPU device memory.  'pinned_host' is a memory kind for accelerator
  # devices (data in CPU RAM, associated with GPU/TPU) and is not valid here;
  # for a CPU device, 'device' memory is just CPU RAM.
  cpu_sh = jax.sharding.SingleDeviceSharding(cpu)

  if accelerator_sharding is None:
    acc_sh = jax.sharding.SingleDeviceSharding(jax.devices()[0])
  else:
    acc_sh = accelerator_sharding

  # For rng_key: it has shape [2], not batch-shaped, so it cannot be sharded
  # along a batch axis.  When acc_sh is a NamedSharding, replicate rng_key
  # across the mesh instead.
  if isinstance(acc_sh, jax.sharding.NamedSharding):
    rng_sh = jax.sharding.NamedSharding(
        acc_sh.mesh, jax.sharding.PartitionSpec())
  else:
    rng_sh = acc_sh

  # Run the entire setup and loop under the CPU default device so that all
  # incidental jnp operations (arange, zeros_like, where, etc.) — including
  # those inside expand() and simulate() — land on CPU without per-call
  # device_put.
  with jax.default_device(cpu):
    # ---- Place root and auxiliary arrays on the CPU ----
    rng_key = jax.device_put(rng_key, cpu_sh)
    root_cpu = base.RootFnOutput(
        prior_logits=jax.device_put(root.prior_logits, cpu_sh),
        value=jax.device_put(root.value, cpu_sh),
        embedding=jax.tree.map(
            lambda x: jax.device_put(x, cpu_sh), root.embedding),
    )

    batch_size = root_cpu.value.shape[0]
    if max_depth is None:
      max_depth = num_simulations
    if invalid_actions is None:
      invalid_actions = jnp.zeros_like(root_cpu.prior_logits)
    else:
      invalid_actions = jax.device_put(invalid_actions, cpu_sh)
    if extra_data is not None:
      extra_data = jax.device_put(extra_data, cpu_sh)

    # ---- Initialise the tree on the CPU ----
    tree = instantiate_tree_from_root(
        root_cpu, num_simulations, invalid_actions, extra_data)

    # ---- Build combined action-selection function ----
    action_selection_fn = action_selection.switching_action_selection_wrapper(
        root_action_selection_fn=root_action_selection_fn,
        interior_action_selection_fn=interior_action_selection_fn,
    )

    # ---- Pre-compile simulate and backward for CPU ----
    # JAX infers the compilation device from input placement; since tree and
    # simulate_keys are on CPU these will compile for CPU on the first call.
    # action_selection_fn and max_depth are Python objects, not traced arrays.
    _simulate = jax.jit(
        simulate, static_argnames=('action_selection_fn', 'max_depth'))
    _backward = jax.jit(backward)

    # ---- Pre-compile recurrent_fn for the accelerator ----
    # out_shardings=cpu_sh tells XLA to place every output leaf back on the CPU,
    # so no explicit device_put loop is needed after the call.  XLA can also
    # overlap the transfer with other computation.
    _rf_acc = jax.jit(recurrent_fn, out_shardings=cpu_sh)

    def offloaded_recurrent_fn(params, rng_key, action, embedding):
      # Move the small per-step tensors to the accelerator.
      # params are left untouched — the caller is responsible for their placement.
      action_a = jax.device_put(action, acc_sh)
      emb_a = jax.tree.map(lambda x: jax.device_put(x, acc_sh), embedding)
      rng_a = jax.device_put(rng_key, rng_sh)
      # Outputs land in cpu_sh (pinned_host) automatically via out_shardings.
      return _rf_acc(params, rng_a, action_a, emb_a)

    batch_range = jnp.arange(batch_size)

    # ---- Python loop over simulations ----
    for i in range(num_simulations):
      rng_key, simulate_key, expand_key = jax.random.split(rng_key, 3)
      simulate_keys = jax.random.split(simulate_key, batch_size)

      parent_index, action = _simulate(
          simulate_keys, tree, action_selection_fn, max_depth)

      next_node_index = tree.children_index[batch_range, parent_index, action]
      next_node_index = jnp.where(
          next_node_index == Tree.UNVISITED, i + 1, next_node_index)

      # expand() calls offloaded_recurrent_fn, which dispatches to the
      # accelerator and returns pinned-host arrays.
      tree = expand(
          params, expand_key, tree, offloaded_recurrent_fn,
          parent_index, action, next_node_index)

      tree = _backward(tree, next_node_index)

  return tree
