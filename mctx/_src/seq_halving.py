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
"""Functions for Sequential Halving."""

import math

import chex
import jax.numpy as jnp


def score_considered(considered_visit, gumbel, logits, normalized_qvalues,
                     visit_counts):
  """Returns a score usable for an argmax."""
  # We allow to visit a child, if it is the only considered child.
  low_logit = -1e9
  logits = logits - jnp.max(logits, keepdims=True, axis=-1)
  penalty = jnp.where(
      visit_counts == considered_visit,
      0, -jnp.inf)
  chex.assert_equal_shape([gumbel, logits, normalized_qvalues, penalty])
  return jnp.maximum(low_logit, gumbel + logits + normalized_qvalues) + penalty


def get_sequence_of_considered_visits(max_num_considered_actions,
                                      num_simulations):
  """Returns a sequence of visit counts considered by Sequential Halving.

  Sequential Halving is a "pure exploration" algorithm for bandits, introduced
  in "Almost Optimal Exploration in Multi-Armed Bandits":
  http://proceedings.mlr.press/v28/karnin13.pdf

  The visit counts allows to implement Sequential Halving by selecting the best
  action from the actions with the currently considered visit count.

  Args:
   max_num_considered_actions: The maximum number of considered actions.
     The `max_num_considered_actions` can be smaller than the number of
     actions.
   num_simulations: The total simulation budget.

  Returns:
    A tuple with visit counts. Length `num_simulations`.
  """
  if max_num_considered_actions <= 1:
    return tuple(range(num_simulations))
  log2max = int(math.ceil(math.log2(max_num_considered_actions)))
  sequence = []
  visits = [0] * max_num_considered_actions
  num_considered = max_num_considered_actions
  while len(sequence) < num_simulations:
    num_extra_visits = max(1, int(num_simulations / (log2max * num_considered)))
    for _ in range(num_extra_visits):
      sequence.extend(visits[:num_considered])
      for i in range(num_considered):
        visits[i] += 1
    # Halving the number of considered actions.
    num_considered = max(2, num_considered // 2)
  return tuple(sequence[:num_simulations])


def get_table_of_considered_visits(max_num_considered_actions, num_simulations):
  """Returns a table of sequences of visit counts.

  Args:
   max_num_considered_actions: The maximum number of considered actions.
     The `max_num_considered_actions` can be smaller than the number of
     actions.
   num_simulations: The total simulation budget.

  Returns:
    A tuple of sequences of visit counts.
    Shape [max_num_considered_actions + 1, num_simulations].
  """
  return tuple(
      get_sequence_of_considered_visits(m, num_simulations)
      for m in range(max_num_considered_actions + 1))


def get_active_explorer_table_original(max_num_considered_actions: int,

                       num_simulations: int,
                       p: int # num of explorers
                      #  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
                       ) -> jnp.ndarray:
    """
    Parameters
    ----------
    max_num_considered_actions : int
    num_simulations            : int
    p                          : int
        Bucket width (default 16).

    Returns
    -------
    considered : ⟨max_m + 1, num_simulations⟩ int32
    active     : ⟨max_m + 1, num_simulations, p⟩ int32
        Each slice `active[m, g]` contains the *g-th* run-length group from
        `considered[m]`, left-padded with that value, right-padded with –1.
    """
    # original DeepMind helper
    considered = jnp.asarray(
        get_table_of_considered_visits(max_num_considered_actions,
                                       num_simulations),
        dtype=jnp.int32)

    # start everything at −1
    active = -jnp.ones((max_num_considered_actions + 1,
                        num_simulations, p),
                       dtype=jnp.int32)

    for m in range(max_num_considered_actions + 1):
        row = considered[m]                         # (S,)
        # positions where the value changes
        change_pts = jnp.nonzero(jnp.diff(row))[0] + 1
        # prepend 0 and append S to get full segment boundaries
        boundaries = jnp.concatenate(
            [jnp.array([0], dtype=jnp.int32),
             change_pts,
             jnp.array([num_simulations], dtype=jnp.int32)]
        )

        # fill one bucket per group
        for g in range(len(boundaries) - 1):
            if g >= num_simulations:        # guard: table only has S buckets
                break
            start, end = int(boundaries[g]), int(boundaries[g + 1])
            length = end - start
            if length == 0:
                continue
            active = active.at[m, g, :length].set(row[start])

    return active
    # return considered, active


import numpy as np
import jax.numpy as jnp
from functools import lru_cache


# @lru_cache(maxsize=None)        # memoise across JIT recompiles
# def get_active_explorer_table(max_num_considered_actions, num_simulations, p):
#     """
#     Returns  active  ∈ ℤ^{max_m+1 × num_simulations × p}.
#     active[m, g, k] = visit-count assigned to the kth explorer bucket
#                       (or –1 if unused).
#     """
#     # 1. “considered” table – original helper already pure-Python
#     considered = np.asarray(
#         get_table_of_considered_visits(max_num_considered_actions,
#                                        num_simulations),
#         dtype=np.int32)                         # shape (max_m+1, S)

#     # 2. Allocate result
#     active = -np.ones((max_num_considered_actions + 1,
#                        num_simulations,
#                        p), dtype=np.int32)

#     # 3. Fill each row with run-length segments
#     for m in range(max_num_considered_actions + 1):
#         row = considered[m]                    # (S,)
#         change_pts = np.nonzero(np.diff(row))[0] + 1
#         boundaries = np.concatenate(([0], change_pts,
#                                      [num_simulations]))
#         for g in range(len(boundaries) - 1):
#             start, end = boundaries[g], boundaries[g + 1]
#             if end > start:
#                 active[m, g, : end - start] = row[start]

#     # 4. Hand back a JAX constant
#     return jnp.asarray(active)

# seq_halving.py
import jax.numpy as jnp
from typing import Tuple

def get_active_explorer_table(
    max_m: int,
    num_sim: int,
    p: int
) -> jnp.ndarray:
  """
  Pure-Python / tuple implementation (no JAX ops, no tracers).
  Returns   active[m, g, k]  ∈ ℤ  with shape  (max_m+1, num_sim, p).
  """
  # ---- 1. table of considered visits (tuple of tuples) -------------------
  considered: Tuple[Tuple[int, ...], ...] = get_table_of_considered_visits(
      max_m, num_sim)                       # shape (max_m+1, num_sim)

  # ---- 2. Build the active-explorer tensor as nested Python lists --------
  active = []
  for m in range(max_m + 1):
    row = list(considered[m])              # Python list length = num_sim
    segments = []                          # will hold num_sim sub-lists

    # run-length encode
    start = 0
    for i in range(1, num_sim):
      if row[i] != row[i - 1]:
        segments.append(row[start:i])
        start = i
    segments.append(row[start:])           # final segment

    # pad / truncate each segment to length p
    padded = [
        seg[:p] + [-1] * (p - len(seg)) if len(seg) < p else seg[:p]
        for seg in segments
    ]
    # if fewer than num_sim segments, pad with all −1 rows
    padded += [[-1] * p] * (num_sim - len(padded))

    active.append(padded)                  # (num_sim, p)

  # ---- 3. Convert to DeviceArray (constant in the compiled graph) --------
  return jnp.asarray(active, dtype=jnp.int32)


def get_num_active_explorers_table(
    max_m: int,
    num_sim: int,
    p: int
) -> jnp.ndarray:
  """
  Pure-Python / tuple implementation (no JAX ops, no tracers).
  Returns   active[m, g]  ∈ ℤ  with shape  (max_m+1, num_sim).
  """
  # ---- 1. table of considered visits (tuple of tuples) -------------------
  considered: Tuple[Tuple[int, ...], ...] = get_table_of_considered_visits(
      max_m, num_sim)                       # shape (max_m+1, num_sim)

  # ---- 2. Build the active-explorer tensor as nested Python lists --------
  active = []
  for m in range(max_m + 1):
    row = list(considered[m])              # Python list length = num_sim
    segments = []                          # will hold num_sim sub-lists

    # run-length encode
    start = 0
    for i in range(1, num_sim):
      if row[i] != row[i - 1]:
        segments.append(row[start:i])
        start = i
    segments.append(row[start:])           # final segment

    # pad / truncate each segment to length p
    padded = [
        seg[:p] + [-1] * (p - len(seg)) if len(seg) < p else seg[:p]
        for seg in segments
    ]
    # if fewer than num_sim segments, pad with all −1 rows
    padded += [[-1] * p] * (num_sim - len(padded))

    active.append(padded)                  # (num_sim, p)

  # ---- 3. Convert to DeviceArray (constant in the compiled graph) --------
  table = jnp.asarray(active, dtype=jnp.int32)
  table = jnp.sum(table, axis=-1)
