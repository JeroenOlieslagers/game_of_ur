//! Cooperative Finkel policy that maximises expected game length.
//!
//! A state is observed before the dice roll, exactly as in the win solver.
//! Every legal move earns one reward, including a move that wins the game.
//! Zero rolls and rolls with no legal move earn zero; terminal states have
//! value zero. Both players maximise the same expected remaining-move count.

use super::*;

const ACTION_FLAG: u32 = 1 << 31;
const INDEX_MASK: u32 = ACTION_FLAG - 1;
const TERMINAL_INDEX: u32 = INDEX_MASK;
const CHECK_SEED: u64 = 0x62a9_7d31_ef48_b50c;
const POLICY_MAGIC: &[u8; 8] = b"RGULPOL1";
const NO_ACTION: u8 = u8::MAX;
const ROLL_COUNT: usize = 5;

struct DurationSuccessors {
    active_rolls: Vec<(u8, f64)>,
    offsets: Vec<u64>,
    entries: Vec<u32>,
}

impl DurationSuccessors {
    #[inline]
    fn resolve(entry: u32, values: &[f64]) -> f64 {
        let reward = if entry & ACTION_FLAG != 0 { 1.0 } else { 0.0 };
        let index = entry & INDEX_MASK;
        reward
            + if index == TERMINAL_INDEX {
                0.0
            } else {
                values[index as usize]
            }
    }

    #[inline]
    fn bellman(&self, position: usize, values: &[f64]) -> f64 {
        let rolls = self.active_rolls.len();
        let mut total = 0.0;
        for (roll_slot, &(_, probability)) in self.active_rolls.iter().enumerate() {
            let start = self.offsets[position * rolls + roll_slot] as usize;
            let end = self.offsets[position * rolls + roll_slot + 1] as usize;
            let mut best = f64::NEG_INFINITY;
            for &entry in &self.entries[start..end] {
                best = best.max(Self::resolve(entry, values));
            }
            total += probability * best;
        }
        total
    }

    fn bytes(&self) -> usize {
        self.offsets.len() * 8 + self.entries.len() * 4
    }
}

#[inline]
fn remaining_training(lut: &TrainingLut, game: &Game, values: &[f64]) -> f64 {
    if game.finished {
        0.0
    } else {
        values[lut.lookup_index(lut.encoding.encode_symmetrical(game))]
    }
}

fn bellman_key(lut: &TrainingLut, key: u64, values: &[f64]) -> f64 {
    let game = lut.encoding.decode(key);
    debug_assert!(!game.finished);
    let mut moves = [0i8; 8];
    let mut total = 0.0;
    for (roll, &probability) in lut.rules.roll_probabilities().iter().enumerate() {
        if probability == 0.0 {
            continue;
        }
        let mut rolled = game.clone();
        let count = rolled.apply_roll(roll as u8, &mut moves);
        let best = if count == 0 {
            remaining_training(lut, &rolled, values)
        } else {
            moves[..count]
                .iter()
                .map(|&source| {
                    let mut next = rolled.clone();
                    next.apply_move(source, lut.rules);
                    1.0 + remaining_training(lut, &next, values)
                })
                .fold(f64::NEG_INFINITY, f64::max)
        };
        total += probability * best;
    }
    total
}

fn successor_entry(lut: &TrainingLut, game: &Game, action: bool) -> u32 {
    let index = if game.finished {
        TERMINAL_INDEX
    } else {
        let index = lut.lookup_index(lut.encoding.encode_symmetrical(game));
        assert!(
            (index as u32) < TERMINAL_INDEX,
            "state index does not fit in 31 bits: {index}"
        );
        index as u32
    };
    index | if action { ACTION_FLAG } else { 0 }
}

fn push_successors(lut: &TrainingLut, key: u64, roll: u8, entries: &mut Vec<u32>) -> usize {
    let game = lut.encoding.decode(key);
    let mut moves = [0i8; 8];
    let mut rolled = game;
    let count = rolled.apply_roll(roll, &mut moves);
    if count == 0 {
        entries.push(successor_entry(lut, &rolled, false));
        return 1;
    }
    for &source in &moves[..count] {
        let mut next = rolled.clone();
        next.apply_move(source, lut.rules);
        entries.push(successor_entry(lut, &next, true));
    }
    count
}

fn build_successors(lut: &TrainingLut, indices: &[u32]) -> DurationSuccessors {
    let active_rolls = lut
        .rules
        .roll_probabilities()
        .iter()
        .enumerate()
        .filter(|(_, &probability)| probability > 0.0)
        .map(|(roll, &probability)| (roll as u8, probability))
        .collect::<Vec<_>>();
    let rolls = active_rolls.len();
    let threads = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .max(1);
    let chunk_size = ((indices.len() + threads - 1) / threads).max(1);

    let chunks: Vec<(Vec<u32>, Vec<u32>)> = thread::scope(|scope| {
        let mut handles = Vec::new();
        for index_chunk in indices.chunks(chunk_size) {
            let active_rolls = &active_rolls;
            handles.push(scope.spawn(move || {
                let mut entries = Vec::with_capacity(index_chunk.len() * rolls * 3);
                let mut counts = Vec::with_capacity(index_chunk.len() * rolls);
                for &global in index_chunk {
                    let key = lut.key_at_global(global as usize);
                    for &(roll, _) in active_rolls {
                        counts.push(push_successors(lut, key, roll, &mut entries) as u32);
                    }
                }
                (counts, entries)
            }));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect()
    });

    let total_entries = chunks.iter().map(|(_, entries)| entries.len()).sum();
    let mut offsets = Vec::with_capacity(indices.len() * rolls + 1);
    let mut entries = Vec::with_capacity(total_entries);
    let mut running = 0u64;
    offsets.push(0);
    for (counts, chunk_entries) in chunks {
        for count in counts {
            running += count as u64;
            offsets.push(running);
        }
        entries.extend(chunk_entries);
    }
    assert_eq!(offsets.len(), indices.len() * rolls + 1);
    DurationSuccessors {
        active_rolls,
        offsets,
        entries,
    }
}

fn residual(successors: &DurationSuccessors, indices: &[u32], values: &[f64]) -> f64 {
    let threads = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .max(1);
    let chunk_size = ((indices.len() + threads - 1) / threads).max(1);
    thread::scope(|scope| {
        let mut handles = Vec::new();
        let mut base = 0usize;
        for index_chunk in indices.chunks(chunk_size) {
            let start = base;
            base += index_chunk.len();
            handles.push(scope.spawn(move || {
                let mut delta = 0.0f64;
                for (offset, &global) in index_chunk.iter().enumerate() {
                    let value = successors.bellman(start + offset, values);
                    delta = delta.max((value - values[global as usize]).abs());
                }
                delta
            }));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .fold(0.0, f64::max)
    })
}

fn residual_vector(
    successors: &DurationSuccessors,
    indices: &[u32],
    values: &[f64],
) -> (Vec<f64>, f64) {
    let threads = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .max(1);
    let chunk_size = ((indices.len() + threads - 1) / threads).max(1);
    let mut residuals = vec![0.0; indices.len()];
    let max_abs = thread::scope(|scope| {
        let mut handles = Vec::new();
        let mut base = 0usize;
        for (index_chunk, residual_chunk) in indices
            .chunks(chunk_size)
            .zip(residuals.chunks_mut(chunk_size))
        {
            let start = base;
            base += index_chunk.len();
            handles.push(scope.spawn(move || {
                let mut maximum = 0.0f64;
                for (offset, (&global, slot)) in index_chunk.iter().zip(residual_chunk).enumerate()
                {
                    *slot = successors.bellman(start + offset, values) - values[global as usize];
                    maximum = maximum.max(slot.abs());
                }
                maximum
            }));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .fold(0.0, f64::max)
    });
    (residuals, max_abs)
}

fn extrapolation_factor(previous: &[f64], current: &[f64], interval: usize) -> Option<f64> {
    let mut dot = 0.0;
    let mut previous_sq = 0.0;
    let mut current_sq = 0.0;
    for (&before, &now) in previous.iter().zip(current) {
        dot += before * now;
        previous_sq += before * before;
        current_sq += now * now;
    }
    if previous_sq == 0.0 || current_sq == 0.0 || dot <= 0.0 {
        return None;
    }
    let cosine = dot / (previous_sq * current_sq).sqrt();
    let interval_ratio = (current_sq / previous_sq).sqrt();
    // Multiple transient modes keep early residuals from being parallel. A
    // positive correlation plus a shrinking L2 norm is enough to justify a
    // trial because the actual correction is guarded by a full max-residual
    // pass and backtracking.
    if cosine < 0.5 || !(0.0..1.0).contains(&interval_ratio) {
        return None;
    }
    let per_sweep = interval_ratio.powf(1.0 / interval as f64);
    if !(0.0..1.0).contains(&per_sweep) {
        return None;
    }
    // Stay below the scalar fixed-point estimate; backtracking below adds a
    // second guard for non-scalar/nonlinear layers.
    Some((0.8 / (1.0 - per_sweep)).clamp(1.0, 5_000.0))
}

fn extrapolation_corrections(
    previous: &[f64],
    current: &[f64],
    interval: usize,
) -> Option<(Vec<f64>, f64)> {
    let fallback = extrapolation_factor(previous, current, interval)?;
    let mut largest_factor = fallback;
    let corrections = previous
        .iter()
        .zip(current)
        .map(|(&before, &now)| {
            let ratio = if before != 0.0 && before.signum() == now.signum() {
                (now / before).abs()
            } else {
                f64::NAN
            };
            let factor = if (0.0..1.0).contains(&ratio) {
                let per_sweep = ratio.powf(1.0 / interval as f64);
                (0.8 / (1.0 - per_sweep)).clamp(1.0, 5_000.0)
            } else {
                fallback
            };
            largest_factor = largest_factor.max(factor);
            factor * now
        })
        .collect();
    Some((corrections, largest_factor))
}

#[inline]
fn bellman_atomic(successors: &DurationSuccessors, position: usize, values: &[AtomicU64]) -> f64 {
    let rolls = successors.active_rolls.len();
    let mut total = 0.0;
    for (roll_slot, &(_, probability)) in successors.active_rolls.iter().enumerate() {
        let start = successors.offsets[position * rolls + roll_slot] as usize;
        let end = successors.offsets[position * rolls + roll_slot + 1] as usize;
        let mut best = f64::NEG_INFINITY;
        for &entry in &successors.entries[start..end] {
            let reward = if entry & ACTION_FLAG != 0 { 1.0 } else { 0.0 };
            let index = entry & INDEX_MASK;
            let continuation = if index == TERMINAL_INDEX {
                0.0
            } else {
                f64::from_bits(values[index as usize].load(AtomicOrdering::Relaxed))
            };
            best = best.max(reward + continuation);
        }
        total += probability * best;
    }
    total
}

fn gauss_seidel(successors: &DurationSuccessors, indices: &[u32], values: &mut [f64]) -> f64 {
    let threads = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .max(1);
    let chunk_size = ((indices.len() + threads - 1) / threads).max(1);
    let pointer = values.as_mut_ptr();
    let atomics: &[AtomicU64] = unsafe {
        // f64 and AtomicU64 share size/alignment. During this scope every
        // concurrent access is atomic, as in the original f64 solver.
        std::slice::from_raw_parts(pointer as *const AtomicU64, values.len())
    };
    thread::scope(|scope| {
        let mut handles = Vec::new();
        let mut base = 0usize;
        for index_chunk in indices.chunks(chunk_size) {
            let start = base;
            base += index_chunk.len();
            handles.push(scope.spawn(move || {
                let mut delta = 0.0f64;
                for (offset, &global) in index_chunk.iter().enumerate() {
                    let value = bellman_atomic(successors, start + offset, atomics);
                    let previous =
                        f64::from_bits(atomics[global as usize].load(AtomicOrdering::Relaxed));
                    delta = delta.max((value - previous).abs());
                    atomics[global as usize].store(value.to_bits(), AtomicOrdering::Relaxed);
                }
                delta
            }));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .fold(0.0, f64::max)
    })
}


/// Modified policy iteration for the duration objective.
///
/// Value iteration is the wrong algorithm for this problem. Cooperating players
/// avoid scoring, so the per-step probability of leaving a score layer is tiny,
/// expected durations run to hundreds or thousands of actions, and the
/// iteration contracts at a rate set by that escape probability: the observed
/// per-sweep factor on layer (3,3) is 0.99998, needing ~800,000 sweeps for 1e-8
/// on 0.9% of the state space.
///
/// Policy iteration attacks the cause. With the maximising action frozen, the
/// evaluation step is *linear*, so its error is dominated by a single mode and
/// Aitken extrapolation of the vector sequence sums the remaining geometric
/// tail in one step. That is exactly what cannot be done to the raw Bellman
/// iteration, where the argmax keeps shifting and the dominant mode with it --
/// which is the likely reason the earlier guarded extrapolation did not help.
///
/// Returns `(bellman_residual, outer_iterations, inner_sweeps)`.
/// Guarded policy-evaluation acceleration on top of value iteration.
///
/// Two facts set the design. First, plain value iteration is *safe* here: this
/// is an undiscounted positive-reward model, so iterating from V=0 increases
/// monotonically to V* -- but it contracts at ~0.99998 per sweep because
/// cooperating players avoid scoring, needing ~800k sweeps per layer.
///
/// Second, policy iteration is *not* safe here. Greedy improvement can produce
/// an **improper** policy, one under which the players cycle forever and never
/// score; for such a policy `(I - P_pi)` is singular and `V_pi = +infinity`, so
/// the inner solve chases an infinite fixed point and the outer Bellman residual
/// gets worse rather than better. That was observed directly: residual rising
/// 161 -> 663 -> 1650 over three outer iterations.
///
/// So value iteration drives, and policy evaluation is only ever a *proposal*.
/// After each block of Bellman sweeps the greedy policy is evaluated with Aitken
/// extrapolation -- linear, hence one dominant mode, hence the geometric tail
/// sums in a step -- and the result is adopted only if it strictly lowers the
/// deterministic Bellman residual. A divergent or improper evaluation is
/// detected and discarded, so the worst case is value iteration's own rate.
///
/// Returns `(bellman_residual, accepted_proposals, sweeps)`.
fn solve_layer_accelerated(
    successors: &DurationSuccessors,
    indices: &[u32],
    values: &mut [f64],
    tolerance: f64,
    max_blocks: usize,
    label: &str,
) -> (f64, usize, usize) {
    let rolls = successors.active_rolls.len();
    let positions = indices.len();
    let mut policy = vec![0u32; positions * rolls];
    let mut sweeps = 0usize;
    let mut accepted = 0usize;

    let bellman_sweep = |values: &mut [f64]| -> f64 {
        let mut delta = 0.0f64;
        for position in 0..positions {
            let updated = successors.bellman(position, values);
            let global = indices[position] as usize;
            delta = delta.max((updated - values[global]).abs());
            values[global] = updated;
        }
        delta
    };
    let extract = |values: &[f64], policy: &mut [u32]| {
        for position in 0..positions {
            for slot in 0..rolls {
                let start = successors.offsets[position * rolls + slot] as usize;
                let end = successors.offsets[position * rolls + slot + 1] as usize;
                let mut best = f64::NEG_INFINITY;
                let mut choice = start as u32;
                for index in start..end {
                    let value = DurationSuccessors::resolve(successors.entries[index], values);
                    if value > best {
                        best = value;
                        choice = index as u32;
                    }
                }
                policy[position * rolls + slot] = choice;
            }
        }
    };
    let policy_sweep = |values: &mut [f64], policy: &[u32]| -> f64 {
        let mut delta = 0.0f64;
        for position in 0..positions {
            let mut total = 0.0;
            for (slot, &(_, probability)) in successors.active_rolls.iter().enumerate() {
                let entry = successors.entries[policy[position * rolls + slot] as usize];
                total += probability * DurationSuccessors::resolve(entry, values);
            }
            let global = indices[position] as usize;
            delta = delta.max((total - values[global]).abs());
            values[global] = total;
        }
        delta
    };
    let gather = |values: &[f64]| -> Vec<f64> {
        indices.iter().map(|&g| values[g as usize]).collect()
    };
    let scatter = |values: &mut [f64], layer: &[f64]| {
        for (position, &global) in indices.iter().enumerate() {
            values[global as usize] = layer[position];
        }
    };

    let mut best_residual = successors_residual(successors, indices, values);
    for block in 1..=max_blocks {
        for _ in 0..200 {
            bellman_sweep(values);
            sweeps += 1;
        }
        let baseline = gather(values);
        let baseline_residual = successors_residual(successors, indices, values);
        if baseline_residual <= tolerance {
            best_residual = baseline_residual;
            break;
        }

        // Propose: evaluate the greedy policy with extrapolation.
        extract(values, &mut policy);
        let mut previous_step: Option<Vec<f64>> = None;
        let mut snapshot = gather(values);
        let mut diverged = false;
        let mut smallest = f64::INFINITY;
        for _round in 0..300 {
            let mut delta = 0.0;
            for _ in 0..8 {
                delta = policy_sweep(values, &policy);
                sweeps += 1;
            }
            if !delta.is_finite() || delta > smallest * 100.0 {
                // An improper policy has no finite value; stop chasing it.
                diverged = true;
                break;
            }
            smallest = smallest.min(delta);
            if delta < tolerance * 0.1 {
                break;
            }
            let current = gather(values);
            let step: Vec<f64> = current.iter().zip(&snapshot).map(|(a, b)| a - b).collect();
            if let Some(earlier) = &previous_step {
                let numerator = step.iter().map(|v| v * v).sum::<f64>().sqrt();
                let denominator = earlier.iter().map(|v: &f64| v * v).sum::<f64>().sqrt();
                if denominator > 0.0 {
                    let rho = numerator / denominator;
                    if rho > 0.0 && rho < 0.999_999 {
                        let scale = rho / (1.0 - rho);
                        for (position, &global) in indices.iter().enumerate() {
                            values[global as usize] = current[position] + step[position] * scale;
                        }
                    }
                }
            }
            previous_step = Some(step);
            snapshot = gather(values);
        }

        let proposed = successors_residual(successors, indices, values);
        if diverged || !proposed.is_finite() || proposed >= baseline_residual {
            scatter(values, &baseline);
            best_residual = baseline_residual;
        } else {
            accepted += 1;
            best_residual = proposed;
        }
        eprintln!(
            "{label} block={block} sweeps={sweeps} vi_residual={baseline_residual:.6e} \
             proposed={proposed:.6e} accepted={accepted} diverged={diverged}"
        );
        if best_residual <= tolerance {
            break;
        }
    }
    (best_residual, accepted, sweeps)
}

/// Maximum Bellman residual over a layer, single-threaded and deterministic.
fn successors_residual(successors: &DurationSuccessors, indices: &[u32], values: &[f64]) -> f64 {
    let mut worst = 0.0f64;
    for position in 0..indices.len() {
        let global = indices[position] as usize;
        worst = worst.max((successors.bellman(position, values) - values[global]).abs());
    }
    worst
}

fn solve_layer(
    lut: &mut TrainingLut,
    indices: &[u32],
    pair: (u8, u8),
    tolerance: f64,
    max_sweeps: usize,
    all_started: &Instant,
) -> f64 {
    let build_started = Instant::now();
    let successors = build_successors(lut, indices);
    eprintln!(
        "long-game scores=[{},{}] successor_entries={} bytes={:.3}GB build_seconds={:.2}",
        pair.0,
        pair.1,
        successors.entries.len(),
        successors.bytes() as f64 / 1e9,
        build_started.elapsed().as_secs_f64()
    );

    let checks = indices.len().min(2_000);
    let mut rng = SplitMix64::new(CHECK_SEED ^ ((pair.0 as u64) << 8) ^ pair.1 as u64);
    let mut worst = 0.0f64;
    for _ in 0..checks {
        let position = rng.index(indices.len());
        let global = indices[position] as usize;
        let expected = bellman_key(lut, lut.key_at_global(global), &lut.values);
        let actual = successors.bellman(position, &lut.values);
        worst = worst.max((expected - actual).abs());
    }
    assert!(
        worst == 0.0,
        "precomputed duration Bellman mismatch in layer {pair:?}: {worst:.3e}"
    );

    // Policy iteration first. Value iteration is retained below only as a
    // fallback if the policy loop fails to certify, since it is correct but
    // impractically slow on this objective.
    if std::env::var("UR_LONG_VALUE_ITERATION").is_err() {
        let label = format!("long-game scores=[{},{}]", pair.0, pair.1);
        let started = Instant::now();
        let (residual, outer, sweeps) = solve_layer_accelerated(
            &successors, indices, &mut lut.values, tolerance, 4_000, &label);
        if residual <= tolerance {
            eprintln!(
                "{label} certified_residual={residual:.12e} outer={outer} sweeps={sweeps} \
                 elapsed_seconds={:.1} total_seconds={:.1}",
                started.elapsed().as_secs_f64(),
                all_started.elapsed().as_secs_f64()
            );
            return residual;
        }
        eprintln!("{label} policy iteration did not certify ({residual:.6e}); falling back");
    }

    let mut certified = f64::INFINITY;
    let acceleration_interval = 500usize;
    let mut previous_residual: Option<(usize, Vec<f64>)> = None;
    for sweep in 1..=max_sweeps {
        let sweep_started = Instant::now();
        let delta = gauss_seidel(&successors, indices, &mut lut.values);
        if sweep <= 5 || (sweep <= 100 && sweep % 10 == 0) || sweep % 100 == 0 || delta <= tolerance
        {
            eprintln!(
                "long-game scores=[{},{}] sweep={} max_delta={:.12e} seconds={:.2}",
                pair.0,
                pair.1,
                sweep,
                delta,
                sweep_started.elapsed().as_secs_f64()
            );
        }
        if sweep % acceleration_interval == 0 {
            let (current, current_max) = residual_vector(&successors, indices, &lut.values);
            certified = current_max;
            if certified <= tolerance {
                return certified;
            }
            if let Some((previous_sweep, previous)) = previous_residual.take() {
                if let Some((corrections, nominal_factor)) =
                    extrapolation_corrections(&previous, &current, sweep - previous_sweep)
                {
                    let originals = indices
                        .iter()
                        .map(|&global| lut.values[global as usize])
                        .collect::<Vec<_>>();
                    let mut scale = 1.0;
                    let mut accepted = None;
                    for _ in 0..12 {
                        for ((&global, &original), &correction) in
                            indices.iter().zip(&originals).zip(&corrections)
                        {
                            lut.values[global as usize] = original + scale * correction;
                        }
                        let (trial, trial_max) = residual_vector(&successors, indices, &lut.values);
                        if trial_max.is_finite() && trial_max < current_max {
                            accepted = Some((scale, trial, trial_max));
                            break;
                        }
                        scale *= 0.5;
                    }
                    if let Some((scale, trial, trial_max)) = accepted {
                        certified = trial_max;
                        eprintln!(
                            "long-game scores=[{},{}] extrapolate scale={:.4} nominal_max={:.1} residual={:.12e}->{:.12e}",
                            pair.0, pair.1, scale, nominal_factor, current_max, trial_max
                        );
                        previous_residual = Some((sweep, trial));
                    } else {
                        for (&global, original) in indices.iter().zip(originals) {
                            lut.values[global as usize] = original;
                        }
                        previous_residual = Some((sweep, current));
                    }
                } else {
                    previous_residual = Some((sweep, current));
                }
            } else {
                previous_residual = Some((sweep, current));
            }
        } else if delta <= tolerance || sweep % 100 == 0 {
            certified = residual(&successors, indices, &lut.values);
            if certified <= tolerance {
                eprintln!(
                    "long-game scores=[{},{}] certified_residual={:.12e} sweeps={} elapsed_seconds={:.1}",
                    pair.0,
                    pair.1,
                    certified,
                    sweep,
                    all_started.elapsed().as_secs_f64()
                );
                return certified;
            }
        }
    }
    if certified > tolerance {
        certified = residual(&successors, indices, &lut.values);
    }
    certified
}

fn write_map(lut: &TrainingLut, output: &Path, target_precision: f64, achieved_precision: f64) {
    let partial = output.with_extension("rgu.partial");
    let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, File::create(&partial).unwrap());
    let metadata = format!(
        "{{\"value_type\":\"f64\",\"objective\":\"max_expected_legal_moves\",\"move_reward\":1,\"pass_reward\":0,\"terminal_reward\":0,\"target_precision\":{target_precision:.17e},\"training_precision\":{achieved_precision:.17e},{}",
        &lut.metadata[1..]
    );
    writer.write_all(b"RGU\0").unwrap();
    writer
        .write_all(&(metadata.len() as u32).to_be_bytes())
        .unwrap();
    writer.write_all(metadata.as_bytes()).unwrap();
    writer
        .write_all(&(lut.maps.len() as u32).to_be_bytes())
        .unwrap();
    for map in &lut.maps {
        writer.write_all(&(map.count as u32).to_be_bytes()).unwrap();
    }
    let mut buffer = Vec::with_capacity(8 * 1_048_576);
    for chunk in lut.keys.chunks(1_048_576) {
        buffer.clear();
        for &key in chunk {
            buffer.extend_from_slice(&key.to_be_bytes());
        }
        writer.write_all(&buffer).unwrap();
    }
    for chunk in lut.values.chunks(1_048_576) {
        buffer.clear();
        for &value in chunk {
            buffer.extend_from_slice(&value.to_be_bytes());
        }
        writer.write_all(&buffer).unwrap();
    }
    writer.flush().unwrap();
    drop(writer);
    fs::rename(&partial, output).unwrap();
}

pub(super) fn train(input: &Path, output: &Path, tolerance: f64, max_sweeps: usize) {
    assert!(tolerance > 0.0 && tolerance.is_finite());
    assert!(max_sweeps > 0);
    let mut lut = TrainingLut::read_percent16(input);
    assert_eq!(
        lut.rules,
        RuleSet::Finkel,
        "long-game experiment is Finkel-only"
    );
    lut.values.fill(0.0);
    eprintln!("long-game init: all values zero; reward=1/legal-move, 0/pass-or-terminal");

    let layer_dir = output.with_extension("layers");
    let pairs = build_layer_files(&lut, &layer_dir);
    let checkpoint = output.with_extension("checkpoint");
    let (completed, mut precisions) =
        load_checkpoint(&checkpoint, &mut lut, &layer_dir, &pairs, tolerance);
    let started = Instant::now();
    for (layer_index, &pair) in pairs.iter().enumerate() {
        if completed[layer_index] {
            continue;
        }
        let indices = read_layer_indices(&layer_file(&layer_dir, pair));
        eprintln!(
            "long-game scores=[{},{}] states={} start elapsed_seconds={:.1}",
            pair.0,
            pair.1,
            indices.len(),
            started.elapsed().as_secs_f64()
        );
        let precision = solve_layer(&mut lut, &indices, pair, tolerance, max_sweeps, &started);
        assert!(
            precision <= tolerance,
            "layer {pair:?} did not converge within {max_sweeps} sweeps: {precision:.3e}"
        );
        precisions[layer_index] = precision;
        append_checkpoint(&checkpoint, pair, precision, &indices, &lut.values);
    }
    let achieved = precisions.into_iter().fold(0.0f64, f64::max);
    let initial = Game::initial(lut.rules);
    let initial_value = remaining_training(&lut, &initial, &lut.values);
    println!("starting_state_value_actions={initial_value:.12}");
    println!("training_max_residual={achieved:.12e}");
    eprintln!("writing duration LUT {}", output.display());
    write_map(&lut, output, tolerance, achieved);
}

fn assert_duration_map(lut: &Lut) {
    assert_eq!(
        lut.rules,
        RuleSet::Finkel,
        "duration policy must use Finkel rules"
    );
    assert!(
        lut.metadata
            .contains("\"objective\":\"max_expected_legal_moves\""),
        "map is not a long-game duration LUT"
    );
}

#[inline]
fn remaining(lut: &Lut, game: &Game) -> f64 {
    if game.finished {
        0.0
    } else {
        lut.lookup_key(lut.encoding.encode_symmetrical(game))
    }
}

fn bellman_lut(lut: &Lut, game: &Game) -> f64 {
    let mut moves = [0i8; 8];
    let mut total = 0.0;
    for (roll, &probability) in lut.rules.roll_probabilities().iter().enumerate() {
        if probability == 0.0 {
            continue;
        }
        let mut rolled = game.clone();
        let count = rolled.apply_roll(roll as u8, &mut moves);
        let best = if count == 0 {
            remaining(lut, &rolled)
        } else {
            moves[..count]
                .iter()
                .map(|&source| {
                    let mut next = rolled.clone();
                    next.apply_move(source, lut.rules);
                    1.0 + remaining(lut, &next)
                })
                .fold(f64::NEG_INFINITY, f64::max)
        };
        total += probability * best;
    }
    total
}

pub(super) fn verify(model: &Path, samples: usize) {
    let lut = Lut::read(model);
    assert_duration_map(&lut);
    let initial = Game::initial(lut.rules);
    let initial_value = remaining(&lut, &initial);
    let mut rng = SplitMix64::new(CHECK_SEED);
    let mut checked = 0usize;
    let mut max_residual = 0.0f64;
    while checked < samples {
        let (key, value) = lut.key_value_at_global(rng.index(lut.total));
        let game = lut.encoding.decode(key);
        assert_eq!(lut.encoding.encode_light_turn(&game), key);
        if game.finished {
            continue;
        }
        max_residual = max_residual.max((bellman_lut(&lut, &game) - value).abs());
        checked += 1;
    }
    println!("ruleset=finkel");
    println!("objective=max_expected_legal_moves");
    println!("entries={}", lut.total);
    println!("starting_state_value_actions={initial_value:.12}");
    println!("sampled_states={checked}");
    println!("sample_max_bellman_residual={max_residual:.12e}");
}

fn choose_move(lut: &Lut, game: &Game, moves: &[i8]) -> i8 {
    let mut choice = moves[0];
    let mut best = f64::NEG_INFINITY;
    for &source in moves {
        let mut next = game.clone();
        next.apply_move(source, lut.rules);
        let value = remaining(lut, &next);
        if value > best {
            best = value;
            choice = source;
        }
    }
    choice
}

fn encode_source(source: i8) -> u8 {
    assert!((-1..14).contains(&source));
    (source + 1) as u8
}

fn decode_source(encoded: u8) -> i8 {
    assert_ne!(encoded, NO_ACTION);
    encoded as i8 - 1
}

fn build_policy_chunk(lut: &Lut, start: usize, count: usize) -> Vec<u8> {
    let mut policy = vec![NO_ACTION; count * ROLL_COUNT];
    let mut moves = [0i8; 8];
    for local in 0..count {
        let (key, _) = lut.key_value_at_global(start + local);
        let game = lut.encoding.decode(key);
        if game.finished {
            continue;
        }
        for roll in 0..ROLL_COUNT {
            let mut rolled = game.clone();
            let available = rolled.apply_roll(roll as u8, &mut moves);
            if available > 0 {
                policy[local * ROLL_COUNT + roll] =
                    encode_source(choose_move(lut, &rolled, &moves[..available]));
            }
        }
    }
    policy
}

fn write_policy(lut: &Lut, path: &Path) {
    let partial = path.with_extension("policy.partial");
    let mut output = BufWriter::with_capacity(16 * 1024 * 1024, File::create(&partial).unwrap());
    output.write_all(POLICY_MAGIC).unwrap();
    output.write_all(&(lut.total as u64).to_le_bytes()).unwrap();
    output
        .write_all(&(ROLL_COUNT as u32).to_le_bytes())
        .unwrap();
    let threads = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .max(1);
    let states_per_thread = 1_048_576usize;
    let batch_size = states_per_thread * threads;
    let started = Instant::now();
    for base in (0..lut.total).step_by(batch_size) {
        let chunks = thread::scope(|scope| {
            let mut handles = Vec::new();
            for offset in (0..(lut.total - base).min(batch_size)).step_by(states_per_thread) {
                let count = (lut.total - base - offset).min(states_per_thread);
                handles.push(scope.spawn(move || build_policy_chunk(lut, base + offset, count)));
            }
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        for chunk in chunks {
            output.write_all(&chunk).unwrap();
        }
        eprintln!(
            "long-game policy states={}/{} elapsed_seconds={:.1}",
            (base + batch_size).min(lut.total),
            lut.total,
            started.elapsed().as_secs_f64()
        );
    }
    output.flush().unwrap();
    drop(output);
    fs::rename(&partial, path).unwrap();
    eprintln!(
        "long-game policy complete states={} bytes={} elapsed_seconds={:.1}",
        lut.total,
        20 + lut.total * ROLL_COUNT,
        started.elapsed().as_secs_f64()
    );
}

fn read_or_create_policy(lut: &Lut, path: &Path) -> Vec<u8> {
    if !path.exists() {
        write_policy(lut, path);
    }
    let mut input = BufReader::with_capacity(16 * 1024 * 1024, File::open(path).unwrap());
    let mut magic = [0u8; 8];
    input.read_exact(&mut magic).unwrap();
    assert_eq!(&magic, POLICY_MAGIC, "not a long-game policy table");
    let mut count_bytes = [0u8; 8];
    input.read_exact(&mut count_bytes).unwrap();
    assert_eq!(u64::from_le_bytes(count_bytes) as usize, lut.total);
    let mut rolls_bytes = [0u8; 4];
    input.read_exact(&mut rolls_bytes).unwrap();
    assert_eq!(u32::from_le_bytes(rolls_bytes) as usize, ROLL_COUNT);
    let mut policy = Vec::with_capacity(lut.total * ROLL_COUNT);
    input.read_to_end(&mut policy).unwrap();
    assert_eq!(policy.len(), lut.total * ROLL_COUNT);
    policy
}

#[derive(Default)]
struct SimResult {
    histogram: Vec<u64>,
    light_wins: u64,
    rolls: u128,
}

fn one_game(lut: &Lut, policy: &[u8], rng: &mut SplitMix64) -> (usize, usize, bool) {
    let mut game = Game::initial(lut.rules);
    let mut moves = [0i8; 8];
    let mut actions = 0usize;
    let mut rolls = 0usize;
    while !game.finished {
        rolls += 1;
        let roll = lut.rules.roll(rng);
        let count = game.apply_roll(roll, &mut moves);
        if count == 0 {
            continue;
        }
        // apply_roll leaves board and turn unchanged when a legal move exists;
        // the encoder ignores the transient roll field.
        let state_index = lut.lookup_index(lut.encoding.encode_symmetrical(&game));
        let source = decode_source(policy[state_index * ROLL_COUNT + roll as usize]);
        debug_assert!(moves[..count].contains(&source));
        game.apply_move(source, lut.rules);
        actions += 1;
    }
    (actions, rolls, game.light_score >= lut.rules.pieces())
}

fn quantile(histogram: &[u64], games: u64, probability: f64) -> usize {
    let target = (probability * games as f64).ceil() as u64;
    let mut cumulative = 0u64;
    for (length, &count) in histogram.iter().enumerate() {
        cumulative += count;
        if cumulative >= target.max(1) {
            return length;
        }
    }
    histogram.len().saturating_sub(1)
}

pub(super) fn simulate(model: &Path, output_dir: &Path, games: usize, seed: u64) {
    assert!(games > 0);
    let lut = Lut::read(model);
    assert_duration_map(&lut);
    fs::create_dir_all(output_dir).unwrap();
    let policy_path = output_dir.join("finkel_longest.policy");
    let policy = read_or_create_policy(&lut, &policy_path);
    let started = Instant::now();
    let threads = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(games);
    let results = thread::scope(|scope| {
        let mut handles = Vec::new();
        let mut first = 0usize;
        let lut_ref = &lut;
        let policy_ref = &policy;
        for thread_index in 0..threads {
            let count = games / threads + usize::from(thread_index < games % threads);
            let first_game = first;
            first += count;
            handles.push(scope.spawn(move || {
                let mut result = SimResult::default();
                for local in 0..count {
                    let game_index = first_game + local;
                    let mut rng = SplitMix64::new(
                        seed ^ (game_index as u64).wrapping_mul(0xd6e8_feb8_6659_fd93),
                    );
                    let (actions, rolls, light_won) = one_game(lut_ref, policy_ref, &mut rng);
                    if result.histogram.len() <= actions {
                        result.histogram.resize(actions + 1, 0);
                    }
                    result.histogram[actions] += 1;
                    result.light_wins += light_won as u64;
                    result.rolls += rolls as u128;
                }
                result
            }));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });

    let mut combined = SimResult::default();
    for result in results {
        if combined.histogram.len() < result.histogram.len() {
            combined.histogram.resize(result.histogram.len(), 0);
        }
        for (length, count) in result.histogram.into_iter().enumerate() {
            combined.histogram[length] += count;
        }
        combined.light_wins += result.light_wins;
        combined.rolls += result.rolls;
    }
    let game_count = games as u64;
    assert_eq!(combined.histogram.iter().sum::<u64>(), game_count);
    let sum = combined
        .histogram
        .iter()
        .enumerate()
        .map(|(length, &count)| length as f64 * count as f64)
        .sum::<f64>();
    let mean = sum / games as f64;
    let variance = combined
        .histogram
        .iter()
        .enumerate()
        .map(|(length, &count)| {
            let difference = length as f64 - mean;
            difference * difference * count as f64
        })
        .sum::<f64>()
        / (games.saturating_sub(1).max(1) as f64);
    let stddev = variance.sqrt();
    let standard_error = stddev / (games as f64).sqrt();
    let start_value = remaining(&lut, &Game::initial(lut.rules));
    let min = combined
        .histogram
        .iter()
        .position(|&count| count > 0)
        .unwrap();
    let max = combined
        .histogram
        .iter()
        .rposition(|&count| count > 0)
        .unwrap();

    let histogram_path = output_dir.join("length_histogram.csv");
    let mut histogram = BufWriter::new(File::create(&histogram_path).unwrap());
    writeln!(histogram, "actions,count,probability,cdf").unwrap();
    let mut cumulative = 0u64;
    for (actions, &count) in combined.histogram.iter().enumerate() {
        if count == 0 {
            continue;
        }
        cumulative += count;
        writeln!(
            histogram,
            "{actions},{count},{:.12e},{:.12e}",
            count as f64 / games as f64,
            cumulative as f64 / games as f64
        )
        .unwrap();
    }
    histogram.flush().unwrap();

    let q05 = quantile(&combined.histogram, game_count, 0.05);
    let q25 = quantile(&combined.histogram, game_count, 0.25);
    let q50 = quantile(&combined.histogram, game_count, 0.50);
    let q75 = quantile(&combined.histogram, game_count, 0.75);
    let q95 = quantile(&combined.histogram, game_count, 0.95);
    let q99 = quantile(&combined.histogram, game_count, 0.99);
    let seconds = started.elapsed().as_secs_f64();
    let summary_path = output_dir.join("summary.json");
    let mut summary = BufWriter::new(File::create(&summary_path).unwrap());
    writeln!(summary, "{{").unwrap();
    writeln!(summary, "  \"objective\": \"max_expected_legal_moves\",").unwrap();
    writeln!(summary, "  \"games\": {games},").unwrap();
    writeln!(summary, "  \"seed\": {seed},").unwrap();
    writeln!(
        summary,
        "  \"starting_state_value_actions\": {start_value:.12},"
    )
    .unwrap();
    writeln!(summary, "  \"simulation_mean_actions\": {mean:.12},").unwrap();
    writeln!(summary, "  \"simulation_stddev_actions\": {stddev:.12},").unwrap();
    writeln!(
        summary,
        "  \"simulation_standard_error\": {standard_error:.12},"
    )
    .unwrap();
    writeln!(
        summary,
        "  \"mean_minus_value\": {:.12},",
        mean - start_value
    )
    .unwrap();
    writeln!(summary, "  \"min_actions\": {min},").unwrap();
    writeln!(summary, "  \"p05_actions\": {q05},").unwrap();
    writeln!(summary, "  \"p25_actions\": {q25},").unwrap();
    writeln!(summary, "  \"median_actions\": {q50},").unwrap();
    writeln!(summary, "  \"p75_actions\": {q75},").unwrap();
    writeln!(summary, "  \"p95_actions\": {q95},").unwrap();
    writeln!(summary, "  \"p99_actions\": {q99},").unwrap();
    writeln!(summary, "  \"max_actions\": {max},").unwrap();
    writeln!(
        summary,
        "  \"mean_rolls\": {:.12},",
        combined.rolls as f64 / games as f64
    )
    .unwrap();
    writeln!(
        summary,
        "  \"light_win_fraction\": {:.12},",
        combined.light_wins as f64 / games as f64
    )
    .unwrap();
    writeln!(summary, "  \"seconds\": {seconds:.6}").unwrap();
    writeln!(summary, "}}").unwrap();
    summary.flush().unwrap();

    println!("games={games}");
    println!("starting_state_value_actions={start_value:.12}");
    println!("simulation_mean_actions={mean:.12}");
    println!("simulation_standard_error={standard_error:.12}");
    println!("median_actions={q50}");
    println!("p95_actions={q95}");
    println!("p99_actions={q99}");
    println!("max_actions={max}");
    println!("histogram={}", histogram_path.display());
    println!("policy={}", policy_path.display());
    println!("summary={}", summary_path.display());
    println!("simulation_seconds={seconds:.3}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_entries_add_reward_only_for_actions() {
        let values = [2.5, 8.0];
        assert_eq!(DurationSuccessors::resolve(0, &values), 2.5);
        assert_eq!(DurationSuccessors::resolve(ACTION_FLAG | 1, &values), 9.0);
        assert_eq!(DurationSuccessors::resolve(TERMINAL_INDEX, &values), 0.0);
        assert_eq!(
            DurationSuccessors::resolve(ACTION_FLAG | TERMINAL_INDEX, &values),
            1.0
        );
    }

    #[test]
    fn duration_bellman_maximises_each_roll_then_averages() {
        let successors = DurationSuccessors {
            active_rolls: vec![(0, 0.25), (1, 0.75)],
            offsets: vec![0, 1, 3],
            entries: vec![0, ACTION_FLAG | 0, ACTION_FLAG | 1],
        };
        let values = [4.0, 7.0];
        assert_eq!(successors.bellman(0, &values), 0.25 * 4.0 + 0.75 * 8.0);
    }

    #[test]
    fn extrapolation_recovers_a_collinear_slow_mode() {
        let previous = [2.0, 4.0, 6.0];
        let current = [1.0, 2.0, 3.0];
        let factor = extrapolation_factor(&previous, &current, 10).unwrap();
        let expected = 0.8 / (1.0 - 0.5f64.powf(0.1));
        assert!((factor - expected).abs() < 1e-12);
        assert!(extrapolation_factor(&previous, &[-1.0, -2.0, -3.0], 10).is_none());
        let (corrections, largest) = extrapolation_corrections(&previous, &current, 10).unwrap();
        assert!((largest - expected).abs() < 1e-12);
        for (&correction, &residual) in corrections.iter().zip(&current) {
            assert!((correction - expected * residual).abs() < 1e-12);
        }
    }
}
