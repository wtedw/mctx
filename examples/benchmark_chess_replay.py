"""Benchmark BNK search against optimized BNK search with action replay.

This benchmark expects https://github.com/wtedw/pgx1 at HEAD. For example:

  PYTHONPATH=/path/to/pgx1 JAX_PLATFORMS=cpu python \
    examples/benchmark_chess_replay.py

Compilation and opening-position generation are reported separately from the
steady-state measurements.
"""

import argparse
import functools
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import pgx1
from pgx1.chess import unpack_bitmask

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mctx


def _pgx1_metadata():
  package_root = Path(pgx1.__file__).resolve().parent.parent
  try:
    commit = subprocess.run(
        ["git", "-C", str(package_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True).stdout.strip()
  except (OSError, subprocess.CalledProcessError):
    commit = "unknown"
  try:
    version = importlib.metadata.version("pgx1")
  except importlib.metadata.PackageNotFoundError:
    version = "source checkout"
  return {"path": str(package_root), "commit": commit, "version": version}


def _percentile(samples, percentile):
  return float(np.percentile(np.asarray(samples), percentile))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch-size", type=int, default=256)
  parser.add_argument("--num-k-actions", type=int, default=16)
  parser.add_argument("--num-simulations", type=int, default=128)
  parser.add_argument("--opening-plies", type=int, default=8)
  parser.add_argument("--warmups", type=int, default=2)
  parser.add_argument("--runs", type=int, default=10)
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  env = pgx1.make(
      "chess", use_bitmask=True, return_observation=False)
  batch_size = args.batch_size

  def legal_action_mask(state):
    return unpack_bitmask(state.legal_action_bitmask)

  def embedding_step_fn(action, state):
    return env.step(state, action)

  def recurrent_fn(params, rng_key, action, embedding):
    del params, rng_key
    acting_player = embedding.current_player
    next_embedding = env.step_batch(embedding, action)
    next_legal_actions = legal_action_mask(next_embedding)
    reward = jnp.take_along_axis(
        next_embedding.rewards, acting_player[:, None], axis=1)[:, 0]
    is_terminal = next_embedding.terminated | next_embedding.truncated
    output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=(~is_terminal).astype(jnp.float32),
        prior_logits=jnp.where(next_legal_actions, 0.0, -1e9),
        value=jnp.zeros(action.shape, dtype=jnp.float32))
    return output, next_embedding

  def make_positions(key):
    key, init_key = jax.random.split(key)
    init_keys = jax.random.split(init_key, batch_size)
    states = jax.vmap(env.init)(init_keys)

    def play_ply(ply, states):
      legal_actions = legal_action_mask(states)
      action_key = jax.random.fold_in(key, ply)
      actions = jax.random.categorical(
          action_key, jnp.where(legal_actions, 0.0, -1e9))
      return env.step_batch(states, actions.astype(jnp.int32))

    return jax.lax.fori_loop(0, args.opening_plies, play_ply, states)

  position_key, search_key = jax.random.split(jax.random.PRNGKey(args.seed))
  position_start = time.perf_counter()
  states = jax.jit(make_positions)(position_key)
  states.current_player.block_until_ready()
  position_seconds = time.perf_counter() - position_start

  root_legal_actions = legal_action_mask(states)
  root = mctx.RootFnOutput(
      prior_logits=jnp.where(root_legal_actions, 0.0, -1e9),
      value=jnp.zeros((batch_size,), dtype=jnp.float32),
      embedding=states)
  invalid_actions = ~root_legal_actions

  common_policy_args = dict(
      params=(),
      root=root,
      recurrent_fn=recurrent_fn,
      num_k_actions=args.num_k_actions,
      num_simulations=args.num_simulations,
      invalid_actions=invalid_actions,
      max_depth=args.num_simulations,
      max_num_considered_actions=args.num_k_actions,
      gumbel_scale=0.0)

  def run_bnk(key):
    return mctx.gumbel_muzero_policy_bnk(
        rng_key=key, **common_policy_args)

  def run_opt_replay(key):
    return mctx.gumbel_muzero_policy_opt(
        rng_key=key,
        use_opt_replay_actions=True,
        embedding_step_fn=embedding_step_fn,
        return_search_tree=True,
        **common_policy_args)

  compiled = {
      "bnk": jax.jit(run_bnk),
      "opt_bnk_replay": jax.jit(run_opt_replay),
  }
  compile_seconds = {}
  first_outputs = {}
  for index, (name, function) in enumerate(compiled.items()):
    start = time.perf_counter()
    output = function(jax.random.fold_in(search_key, index))
    output.action.block_until_ready()
    compile_seconds[name] = time.perf_counter() - start
    first_outputs[name] = output

  # Both policies must at least propose legal actions for every input state.
  batch_range = jnp.arange(batch_size)
  for name, output in first_outputs.items():
    if not bool(jnp.all(root_legal_actions[batch_range, output.action])):
      raise AssertionError(f"{name} proposed an illegal chess action")

  for warmup in range(args.warmups):
    for index, function in enumerate(compiled.values()):
      output = function(jax.random.fold_in(search_key, 100 + warmup * 2 + index))
      output.action.block_until_ready()

  samples = {name: [] for name in compiled}
  names = tuple(compiled)
  for run in range(args.runs):
    order = names if run % 2 == 0 else tuple(reversed(names))
    for index, name in enumerate(order):
      key = jax.random.fold_in(search_key, 1000 + run * 2 + index)
      start = time.perf_counter()
      output = compiled[name](key)
      output.action.block_until_ready()
      samples[name].append(time.perf_counter() - start)

  results = {}
  for name, timings in samples.items():
    median = float(np.median(timings))
    results[name] = {
        "compile_seconds": compile_seconds[name],
        "median_seconds": median,
        "p10_seconds": _percentile(timings, 10),
        "p90_seconds": _percentile(timings, 90),
        "searches_per_second": batch_size / median,
        "simulations_per_second": (
            batch_size * args.num_simulations / median),
        "samples_seconds": timings,
    }

  results["opt_bnk_replay"]["speedup_vs_bnk"] = (
      results["bnk"]["median_seconds"] /
      results["opt_bnk_replay"]["median_seconds"])
  report = {
      "configuration": vars(args),
      "jax": {
          "version": jax.__version__,
          "backend": jax.default_backend(),
          "devices": [str(device) for device in jax.devices()],
      },
      "pgx1": _pgx1_metadata(),
      "position_generation_seconds": position_seconds,
      "results": results,
  }
  print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
