#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

/*
 * batched_entropy_codex.cu
 *
 * Codex v1 two-guess GPU search kernel:
 *   - Anti-diagonal task decode from linear task id
 *   - Per-candidate bound gate using live floor
 *   - Compact per-thread histogram (open-addressing hash)
 *   - Two-layer staging: block-local shared buffer -> global append buffer
 */

#define MAX_BINS 59049
#define TABLE_SIZE 4096
#define TABLE_MASK (TABLE_SIZE - 1)
#define EMPTY_KEY 0xFFFFFFFFu
#define ENTROPY_SCALE 1000000.0f
#define BLOCK_THREADS 128

__device__ __forceinline__ uint32_t hash_key(uint32_t key) {
    return (key * 2654435761u) & TABLE_MASK;
}

__device__ __forceinline__ int float_to_entropy_q(float v) {
    return __float2int_rn(v * ENTROPY_SCALE);
}

__device__ __forceinline__ long long comb2_ll(long long x) {
    if (x < 2) {
        return 0;
    }
    return (x * (x - 1)) / 2;
}

__device__ __forceinline__ long long comb3_ll(long long x) {
    if (x < 3) {
        return 0;
    }
    return (x * (x - 1) * (x - 2)) / 6;
}

__device__ __forceinline__ long long antidiag_prefix_count(int s, int n_allowed) {
    // Number of upper-triangle pairs with i+j <= s (where i<j, 0<=i<j<n).
    int m = n_allowed - 1;
    if (s <= 0) {
        return 0;
    }
    if (s <= m) {
        // Sum_{u=1..s} floor((u+1)/2) = floor((s+1)^2 / 4)
        long long a = (long long)(s + 1) * (long long)(s + 1);
        return a / 4;
    }

    long long total = (long long)n_allowed * (long long)(n_allowed - 1) / 2;
    int tail = (2 * m - 1) - s;  // number of remaining anti-diagonals after s
    long long tail_sum = (long long)(tail + 1) * (long long)(tail + 1) / 4;
    return total - tail_sum;
}

__device__ __forceinline__ int antidiag_find_s_for_t(long long t, int n_allowed) {
    // Find smallest s with antidiag_prefix_count(s, n_allowed) > t.
    int m = n_allowed - 1;
    long long first_half = (long long)(m + 1) * (long long)(m + 1) / 4;

    if (t < first_half) {
        int s = (int)(2.0 * sqrt((double)t + 0.25) - 0.5);
        if (s < 1) {
            s = 1;
        }
        if (s > m) {
            s = m;
        }
        while (antidiag_prefix_count(s, n_allowed) <= t) {
            s++;
        }
        while (s > 1 && antidiag_prefix_count(s - 1, n_allowed) > t) {
            s--;
        }
        return s;
    }

    long long total = (long long)n_allowed * (long long)(n_allowed - 1) / 2;
    long long rem = (total - 1) - t;  // reverse index from the end

    int q = (int)(2.0 * sqrt((double)rem + 0.25) - 0.5);
    if (q < 1) {
        q = 1;
    }
    if (q > m) {
        q = m;
    }
    long long q_pref = (long long)(q + 1) * (long long)(q + 1) / 4;
    while (q_pref <= rem) {
        q++;
        q_pref = (long long)(q + 1) * (long long)(q + 1) / 4;
    }
    while (q > 1) {
        long long prev = (long long)q * (long long)q / 4;
        if (prev <= rem) {
            break;
        }
        q--;
    }

    int s = 2 * m - q;
    if (s < m + 1) {
        s = m + 1;
    }
    if (s > 2 * m - 1) {
        s = 2 * m - 1;
    }
    return s;
}

__device__ __forceinline__ void decode_antidiag_pair_from_task(
    long long t,
    int n_allowed,
    int* pos_i,
    int* pos_j
) {
    int s = antidiag_find_s_for_t(t, n_allowed);
    long long prev = antidiag_prefix_count(s - 1, n_allowed);
    int offset = (int)(t - prev);

    int i_min = s - (n_allowed - 1);
    if (i_min < 0) {
        i_min = 0;
    }

    int i = i_min + offset;
    int j = s - i;
    *pos_i = i;
    *pos_j = j;
}

__device__ __forceinline__ void decode_row_pair_from_task(
    long long pair_id,
    int n_allowed,
    int* pos_i,
    int* pos_j
) {
    double nd = (double)n_allowed;
    double disc = (2.0 * nd - 1.0) * (2.0 * nd - 1.0) - 8.0 * (double)pair_id;
    int i = (int)((2.0 * nd - 1.0 - sqrt(disc)) / 2.0);

    long long base = (long long)i * n_allowed - (long long)i * (i + 1) / 2;
    if (pair_id < base) {
        i--;
        base = (long long)i * n_allowed - (long long)i * (i + 1) / 2;
    } else {
        long long next_base =
            (long long)(i + 1) * n_allowed - (long long)(i + 1) * (i + 2) / 2;
        if (pair_id >= next_base) {
            i++;
            base = next_base;
        }
    }

    *pos_i = i;
    *pos_j = (int)(pair_id - base) + i + 1;
}

__device__ __forceinline__ void decode_row_triple_from_task(
    long long t,
    int n_allowed,
    int* pos_i,
    int* pos_j,
    int* pos_k
) {
    // Lexicographic row order: i outer, j middle, k inner (i < j < k).
    long long total = comb3_ll((long long)n_allowed);
    if (t < 0 || t >= total) {
        *pos_i = -1;
        *pos_j = -1;
        *pos_k = -1;
        return;
    }

    // Find i where prefix_i(i) <= t < prefix_i(i+1),
    // prefix_i(i) = C(n,3) - C(n-i,3).
    int lo = 0;
    int hi = n_allowed - 3;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        long long pref_mid = total - comb3_ll((long long)n_allowed - mid);
        if (pref_mid <= t) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    int i = lo;
    long long pref_i = total - comb3_ll((long long)n_allowed - i);
    long long rem = t - pref_i;

    // For fixed i, j spans [i+1, n-2], with suffix count (n-j-1).
    // prefix_j(j) = C(n-i-1,2) - C(n-j,2).
    long long span_i = comb2_ll((long long)n_allowed - i - 1);
    int j_lo = i + 1;
    int j_hi = n_allowed - 2;
    while (j_lo < j_hi) {
        int mid = (j_lo + j_hi + 1) >> 1;
        long long pref_j = span_i - comb2_ll((long long)n_allowed - mid);
        if (pref_j <= rem) {
            j_lo = mid;
        } else {
            j_hi = mid - 1;
        }
    }
    int j = j_lo;
    long long pref_j = span_i - comb2_ll((long long)n_allowed - j);
    long long rem_k = rem - pref_j;
    int k = j + 1 + (int)rem_k;

    *pos_i = i;
    *pos_j = j;
    *pos_k = k;
}

__global__ void compute_two_guess_entropy_q_kernel_codex(
    const uint8_t* matrix_u8,
    int n_allowed,
    int n_answers,
    const int* sorted_indices,
    long long task_start,
    int n_tasks,
    int* out_entropy_q
) {
    int local_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_tid >= n_tasks) {
        return;
    }

    long long t = task_start + (long long)local_tid;
    int i_sorted = -1;
    int j_sorted = -1;
    decode_row_pair_from_task(t, n_allowed, &i_sorted, &j_sorted);
    if (!(i_sorted >= 0 && i_sorted < n_allowed && j_sorted >= 0 && j_sorted < n_allowed && i_sorted < j_sorted)) {
        return;
    }

    int i_idx = sorted_indices[i_sorted];
    int j_idx = sorted_indices[j_sorted];
    const uint8_t* row_i = matrix_u8 + ((long long)i_idx * n_answers);
    const uint8_t* row_j = matrix_u8 + ((long long)j_idx * n_answers);

    uint32_t keys[TABLE_SIZE];
    uint16_t counts[TABLE_SIZE];
    uint16_t touched_slots[2400];
    int touched_n = 0;

    for (int slot = 0; slot < TABLE_SIZE; slot++) {
        keys[slot] = EMPTY_KEY;
        counts[slot] = 0;
    }

    for (int a = 0; a < n_answers; a++) {
        uint32_t joint = (uint32_t)row_i[a] * 243u + (uint32_t)row_j[a];
        uint32_t slot = hash_key(joint);
        while (true) {
            uint32_t key = keys[slot];
            if (key == joint) {
                counts[slot]++;
                break;
            }
            if (key == EMPTY_KEY) {
                keys[slot] = joint;
                counts[slot] = 1;
                if (touched_n < 2400) {
                    touched_slots[touched_n++] = (uint16_t)slot;
                }
                break;
            }
            slot = (slot + 1) & TABLE_MASK;
        }
    }

    float entropy = 0.0f;
    const float total_f = (float)n_answers;
    if (touched_n < 2400) {
        for (int q = 0; q < touched_n; q++) {
            uint16_t slot = touched_slots[q];
            int count = (int)counts[slot];
            float prob = (float)count / total_f;
            entropy -= prob * log2f(prob);
        }
    } else {
        for (int slot = 0; slot < TABLE_SIZE; slot++) {
            int count = (int)counts[slot];
            if (count == 0) {
                continue;
            }
            float prob = (float)count / total_f;
            entropy -= prob * log2f(prob);
        }
    }

    out_entropy_q[t] = float_to_entropy_q(entropy);
}

__global__ void two_guess_search_kernel_codex(
    const uint8_t* matrix_u8,
    int n_allowed,
    int n_answers,
    const int* sorted_indices,
    const float* sorted_entropies,
    long long task_start,
    int n_tasks,
    const int* diag_offsets,
    int n_diags,
    int decode_mode,
    int* floor_q,
    int* out_count,
    int out_capacity,
    int* out_overflow_flag,
    int* out_overflow_dropped,
    int* out_full_prune_blocks,
    unsigned long long* out_active_threads,
    unsigned long long* out_bound_pass_threads,
    int* out_entropy_q,
    int* out_i,
    int* out_j
) {
    __shared__ int s_count;
    __shared__ int s_emitted;
    __shared__ int s_base;
    __shared__ int s_active;
    __shared__ int s_need_compute;
    __shared__ int s_antidiag_i[BLOCK_THREADS];
    __shared__ int s_antidiag_j[BLOCK_THREADS];
    __shared__ int s_entropy_q[BLOCK_THREADS];
    __shared__ int s_i[BLOCK_THREADS];
    __shared__ int s_j[BLOCK_THREADS];

    if (threadIdx.x == 0) {
        s_count = 0;
        s_emitted = 0;
        s_base = 0;
        s_active = 0;
        s_need_compute = 0;
    }
    __syncthreads();

    int local_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int emit_entropy_q = -1;
    int emit_i = -1;
    int emit_j = -1;
    int is_active = 0;
    int passes_bound_gate = 0;
    int i_sorted = -1;
    int j_sorted = -1;

    if (decode_mode == 1) {
        // Block-seeded antidiagonal decode:
        // thread 0 decodes first task in block once, then walks lanes with
        // cheap integer updates instead of per-thread sqrt-based decode.
        if (threadIdx.x == 0) {
            long long block_t0 = task_start + (long long)blockIdx.x * blockDim.x;
            long long block_t_end = task_start + (long long)n_tasks;

            for (int lane = 0; lane < BLOCK_THREADS; lane++) {
                s_antidiag_i[lane] = -1;
                s_antidiag_j[lane] = -1;
            }

            if (block_t0 < block_t_end) {
                int cur_i = -1;
                int cur_j = -1;
                decode_antidiag_pair_from_task(block_t0, n_allowed, &cur_i, &cur_j);

                for (int lane = 0; lane < BLOCK_THREADS; lane++) {
                    long long tt = block_t0 + (long long)lane;
                    if (tt >= block_t_end) {
                        break;
                    }

                    s_antidiag_i[lane] = cur_i;
                    s_antidiag_j[lane] = cur_j;

                    int s = cur_i + cur_j;
                    int i_max = (s - 1) >> 1;
                    int cap = n_allowed - 2;
                    if (i_max > cap) {
                        i_max = cap;
                    }

                    if (cur_i < i_max) {
                        cur_i += 1;
                        cur_j -= 1;
                    } else {
                        int s_next = s + 1;
                        int i_min = s_next - (n_allowed - 1);
                        if (i_min < 0) {
                            i_min = 0;
                        }
                        cur_i = i_min;
                        cur_j = s_next - cur_i;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (local_tid >= n_tasks) {
        // No work for this thread in this launch.
    } else {
        long long t = task_start + (long long)local_tid;
        if (decode_mode == 0) {
            decode_row_pair_from_task(t, n_allowed, &i_sorted, &j_sorted);
        } else {
            i_sorted = s_antidiag_i[threadIdx.x];
            j_sorted = s_antidiag_j[threadIdx.x];
        }

        if (i_sorted >= 0 && i_sorted < n_allowed && j_sorted >= 0 && j_sorted < n_allowed && i_sorted < j_sorted) {
            is_active = 1;
            float bound = sorted_entropies[i_sorted] + sorted_entropies[j_sorted];
            int bound_q = float_to_entropy_q(bound);
            int floor_snapshot = *floor_q;

            if (bound_q > floor_snapshot) {
                passes_bound_gate = 1;
                int i_idx = sorted_indices[i_sorted];
                int j_idx = sorted_indices[j_sorted];

                const uint8_t* row_i = matrix_u8 + ((long long)i_idx * n_answers);
                const uint8_t* row_j = matrix_u8 + ((long long)j_idx * n_answers);

                uint32_t keys[TABLE_SIZE];
                uint16_t counts[TABLE_SIZE];
                uint16_t touched_slots[2400];
                int touched_n = 0;

                for (int slot = 0; slot < TABLE_SIZE; slot++) {
                    keys[slot] = EMPTY_KEY;
                    counts[slot] = 0;
                }

                for (int a = 0; a < n_answers; a++) {
                    uint32_t joint = (uint32_t)row_i[a] * 243u + (uint32_t)row_j[a];
                    uint32_t slot = hash_key(joint);

                    while (true) {
                        uint32_t key = keys[slot];
                        if (key == joint) {
                            counts[slot]++;
                            break;
                        }
                        if (key == EMPTY_KEY) {
                            keys[slot] = joint;
                            counts[slot] = 1;
                            if (touched_n < 2400) {
                                touched_slots[touched_n++] = (uint16_t)slot;
                            }
                            break;
                        }
                        slot = (slot + 1) & TABLE_MASK;
                    }
                }

                float entropy = 0.0f;
                const float total = (float)n_answers;

                if (touched_n < 2400) {
                    for (int i = 0; i < touched_n; i++) {
                        uint16_t slot = touched_slots[i];
                        int count = (int)counts[slot];
                        float prob = (float)count / total;
                        entropy -= prob * log2f(prob);
                    }
                } else {
                    for (int slot = 0; slot < TABLE_SIZE; slot++) {
                        int count = (int)counts[slot];
                        if (count == 0) {
                            continue;
                        }
                        float prob = (float)count / total;
                        entropy -= prob * log2f(prob);
                    }
                }

                int entropy_q = float_to_entropy_q(entropy);
                if (entropy_q > floor_snapshot) {
                    emit_entropy_q = entropy_q;
                    emit_i = i_idx;
                    emit_j = j_idx;
                }
            }
        }
    }

    if (is_active) {
        atomicAdd(&s_active, 1);
    }
    if (passes_bound_gate) {
        atomicAdd(&s_need_compute, 1);
    }

    if (emit_entropy_q >= 0) {
        int pos = atomicAdd(&s_count, 1);
        if (pos < BLOCK_THREADS) {
            s_entropy_q[pos] = emit_entropy_q;
            s_i[pos] = emit_i;
            s_j[pos] = emit_j;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && s_count > 0) {
        s_emitted = s_count;
        int reserve_base = atomicAdd(out_count, s_emitted);
        int remaining = out_capacity - reserve_base;
        if (remaining < 0) {
            remaining = 0;
        }
        int kept = s_emitted;
        if (kept > remaining) {
            kept = remaining;
        }
        s_base = reserve_base;
        s_count = kept;

        int block_dropped = s_emitted - kept;
        if (block_dropped > 0) {
            atomicExch(out_overflow_flag, 1);
            atomicAdd(out_overflow_dropped, block_dropped);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && s_active > 0 && s_need_compute == 0) {
        atomicAdd(out_full_prune_blocks, 1);
    }
    if (threadIdx.x == 0) {
        atomicAdd(out_active_threads, (unsigned long long)s_active);
        atomicAdd(out_bound_pass_threads, (unsigned long long)s_need_compute);
    }

    if (threadIdx.x < s_count) {
        int out_pos = s_base + threadIdx.x;
        if (out_pos < out_capacity) {
            out_entropy_q[out_pos] = s_entropy_q[threadIdx.x];
            out_i[out_pos] = s_i[threadIdx.x];
            out_j[out_pos] = s_j[threadIdx.x];
        }
    }
}

__global__ void three_guess_search_kernel_codex(
    const uint8_t* matrix_u8,
    int n_allowed,
    int n_answers,
    const int* sorted_indices,
    const float* sorted_entropies,
    long long task_start,
    int n_tasks,
    int* floor_q,
    int* out_count,
    int out_capacity,
    int* out_overflow_flag,
    int* out_overflow_dropped,
    int* out_full_prune_blocks,
    unsigned long long* out_active_threads,
    unsigned long long* out_bound_pass_threads,
    int* out_entropy_q,
    int* out_i,
    int* out_j,
    int* out_k
) {
    __shared__ int s_count;
    __shared__ int s_emitted;
    __shared__ int s_base;
    __shared__ int s_active;
    __shared__ int s_need_compute;
    __shared__ int s_entropy_q[BLOCK_THREADS];
    __shared__ int s_i[BLOCK_THREADS];
    __shared__ int s_j[BLOCK_THREADS];
    __shared__ int s_k[BLOCK_THREADS];

    if (threadIdx.x == 0) {
        s_count = 0;
        s_emitted = 0;
        s_base = 0;
        s_active = 0;
        s_need_compute = 0;
    }
    __syncthreads();

    int local_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int emit_entropy_q = -1;
    int emit_i = -1;
    int emit_j = -1;
    int emit_k = -1;
    int is_active = 0;
    int passes_bound_gate = 0;

    if (local_tid < n_tasks) {
        long long t = task_start + (long long)local_tid;
        int i_sorted = -1;
        int j_sorted = -1;
        int k_sorted = -1;
        decode_row_triple_from_task(t, n_allowed, &i_sorted, &j_sorted, &k_sorted);

        if (
            i_sorted >= 0 && j_sorted >= 0 && k_sorted >= 0 &&
            i_sorted < j_sorted && j_sorted < k_sorted &&
            k_sorted < n_allowed
        ) {
            is_active = 1;
            float bound =
                sorted_entropies[i_sorted] +
                sorted_entropies[j_sorted] +
                sorted_entropies[k_sorted];
            int bound_q = float_to_entropy_q(bound);
            int floor_snapshot = *floor_q;

            if (bound_q > floor_snapshot) {
                passes_bound_gate = 1;
                int i_idx = sorted_indices[i_sorted];
                int j_idx = sorted_indices[j_sorted];
                int k_idx = sorted_indices[k_sorted];

                const uint8_t* row_i = matrix_u8 + ((long long)i_idx * n_answers);
                const uint8_t* row_j = matrix_u8 + ((long long)j_idx * n_answers);
                const uint8_t* row_k = matrix_u8 + ((long long)k_idx * n_answers);

                uint32_t keys[TABLE_SIZE];
                uint16_t counts[TABLE_SIZE];
                uint16_t touched_slots[2400];
                int touched_n = 0;

                for (int slot = 0; slot < TABLE_SIZE; slot++) {
                    keys[slot] = EMPTY_KEY;
                    counts[slot] = 0;
                }

                for (int a = 0; a < n_answers; a++) {
                    uint32_t joint =
                        (uint32_t)row_i[a] * 59049u +
                        (uint32_t)row_j[a] * 243u +
                        (uint32_t)row_k[a];
                    uint32_t slot = hash_key(joint);

                    while (true) {
                        uint32_t key = keys[slot];
                        if (key == joint) {
                            counts[slot]++;
                            break;
                        }
                        if (key == EMPTY_KEY) {
                            keys[slot] = joint;
                            counts[slot] = 1;
                            if (touched_n < 2400) {
                                touched_slots[touched_n++] = (uint16_t)slot;
                            }
                            break;
                        }
                        slot = (slot + 1) & TABLE_MASK;
                    }
                }

                float entropy = 0.0f;
                const float total_f = (float)n_answers;
                if (touched_n < 2400) {
                    for (int q = 0; q < touched_n; q++) {
                        uint16_t slot = touched_slots[q];
                        int count = (int)counts[slot];
                        float prob = (float)count / total_f;
                        entropy -= prob * log2f(prob);
                    }
                } else {
                    for (int slot = 0; slot < TABLE_SIZE; slot++) {
                        int count = (int)counts[slot];
                        if (count == 0) {
                            continue;
                        }
                        float prob = (float)count / total_f;
                        entropy -= prob * log2f(prob);
                    }
                }

                int entropy_q = float_to_entropy_q(entropy);
                if (entropy_q > floor_snapshot) {
                    emit_entropy_q = entropy_q;
                    emit_i = i_idx;
                    emit_j = j_idx;
                    emit_k = k_idx;
                }
            }
        }
    }

    if (is_active) {
        atomicAdd(&s_active, 1);
    }
    if (passes_bound_gate) {
        atomicAdd(&s_need_compute, 1);
    }

    if (emit_entropy_q >= 0) {
        int pos = atomicAdd(&s_count, 1);
        if (pos < BLOCK_THREADS) {
            s_entropy_q[pos] = emit_entropy_q;
            s_i[pos] = emit_i;
            s_j[pos] = emit_j;
            s_k[pos] = emit_k;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && s_count > 0) {
        s_emitted = s_count;
        int reserve_base = atomicAdd(out_count, s_emitted);
        int remaining = out_capacity - reserve_base;
        if (remaining < 0) {
            remaining = 0;
        }
        int kept = s_emitted;
        if (kept > remaining) {
            kept = remaining;
        }
        s_base = reserve_base;
        s_count = kept;

        int block_dropped = s_emitted - kept;
        if (block_dropped > 0) {
            atomicExch(out_overflow_flag, 1);
            atomicAdd(out_overflow_dropped, block_dropped);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && s_active > 0 && s_need_compute == 0) {
        atomicAdd(out_full_prune_blocks, 1);
    }
    if (threadIdx.x == 0) {
        atomicAdd(out_active_threads, (unsigned long long)s_active);
        atomicAdd(out_bound_pass_threads, (unsigned long long)s_need_compute);
    }

    if (threadIdx.x < s_count) {
        int out_pos = s_base + threadIdx.x;
        if (out_pos < out_capacity) {
            out_entropy_q[out_pos] = s_entropy_q[threadIdx.x];
            out_i[out_pos] = s_i[threadIdx.x];
            out_j[out_pos] = s_j[threadIdx.x];
            out_k[out_pos] = s_k[threadIdx.x];
        }
    }
}

__global__ void three_guess_search_kernel_indexed_codex(
    const uint8_t* matrix_u8,
    int n_allowed,
    int n_answers,
    const int* sorted_indices,
    const int* task_i_sorted,
    const int* task_j_sorted,
    const int* task_k_sorted,
    int n_tasks,
    int* floor_q,
    int* out_count,
    int out_capacity,
    int* out_overflow_flag,
    int* out_overflow_dropped,
    int* out_full_prune_blocks,
    unsigned long long* out_active_threads,
    unsigned long long* out_bound_pass_threads,
    int* out_entropy_q,
    int* out_i,
    int* out_j,
    int* out_k
) {
    __shared__ int s_count;
    __shared__ int s_emitted;
    __shared__ int s_base;
    __shared__ int s_active;
    __shared__ int s_entropy_q[BLOCK_THREADS];
    __shared__ int s_i[BLOCK_THREADS];
    __shared__ int s_j[BLOCK_THREADS];
    __shared__ int s_k[BLOCK_THREADS];

    if (threadIdx.x == 0) {
        s_count = 0;
        s_emitted = 0;
        s_base = 0;
        s_active = 0;
    }
    __syncthreads();

    int local_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int emit_entropy_q = -1;
    int emit_i = -1;
    int emit_j = -1;
    int emit_k = -1;
    int is_active = 0;

    if (local_tid < n_tasks) {
        int i_sorted = task_i_sorted[local_tid];
        int j_sorted = task_j_sorted[local_tid];
        int k_sorted = task_k_sorted[local_tid];

        if (
            i_sorted >= 0 && j_sorted >= 0 && k_sorted >= 0 &&
            i_sorted < j_sorted && j_sorted < k_sorted &&
            k_sorted < n_allowed
        ) {
            is_active = 1;
            int i_idx = sorted_indices[i_sorted];
            int j_idx = sorted_indices[j_sorted];
            int k_idx = sorted_indices[k_sorted];

            const uint8_t* row_i = matrix_u8 + ((long long)i_idx * n_answers);
            const uint8_t* row_j = matrix_u8 + ((long long)j_idx * n_answers);
            const uint8_t* row_k = matrix_u8 + ((long long)k_idx * n_answers);

            uint32_t keys[TABLE_SIZE];
            uint16_t counts[TABLE_SIZE];
            uint16_t touched_slots[2400];
            int touched_n = 0;

            for (int slot = 0; slot < TABLE_SIZE; slot++) {
                keys[slot] = EMPTY_KEY;
                counts[slot] = 0;
            }

            for (int a = 0; a < n_answers; a++) {
                uint32_t joint =
                    (uint32_t)row_i[a] * 59049u +
                    (uint32_t)row_j[a] * 243u +
                    (uint32_t)row_k[a];
                uint32_t slot = hash_key(joint);

                while (true) {
                    uint32_t key = keys[slot];
                    if (key == joint) {
                        counts[slot]++;
                        break;
                    }
                    if (key == EMPTY_KEY) {
                        keys[slot] = joint;
                        counts[slot] = 1;
                        if (touched_n < 2400) {
                            touched_slots[touched_n++] = (uint16_t)slot;
                        }
                        break;
                    }
                    slot = (slot + 1) & TABLE_MASK;
                }
            }

            float entropy = 0.0f;
            const float total_f = (float)n_answers;
            if (touched_n < 2400) {
                for (int q = 0; q < touched_n; q++) {
                    uint16_t slot = touched_slots[q];
                    int count = (int)counts[slot];
                    float prob = (float)count / total_f;
                    entropy -= prob * log2f(prob);
                }
            } else {
                for (int slot = 0; slot < TABLE_SIZE; slot++) {
                    int count = (int)counts[slot];
                    if (count == 0) {
                        continue;
                    }
                    float prob = (float)count / total_f;
                    entropy -= prob * log2f(prob);
                }
            }

            int entropy_q = float_to_entropy_q(entropy);
            if (entropy_q > *floor_q) {
                emit_entropy_q = entropy_q;
                emit_i = i_idx;
                emit_j = j_idx;
                emit_k = k_idx;
            }
        }
    }

    if (is_active) {
        atomicAdd(&s_active, 1);
    }

    if (emit_entropy_q >= 0) {
        int pos = atomicAdd(&s_count, 1);
        if (pos < BLOCK_THREADS) {
            s_entropy_q[pos] = emit_entropy_q;
            s_i[pos] = emit_i;
            s_j[pos] = emit_j;
            s_k[pos] = emit_k;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && s_count > 0) {
        s_emitted = s_count;
        int reserve_base = atomicAdd(out_count, s_emitted);
        int remaining = out_capacity - reserve_base;
        if (remaining < 0) {
            remaining = 0;
        }
        int kept = s_emitted;
        if (kept > remaining) {
            kept = remaining;
        }
        s_base = reserve_base;
        s_count = kept;

        int block_dropped = s_emitted - kept;
        if (block_dropped > 0) {
            atomicExch(out_overflow_flag, 1);
            atomicAdd(out_overflow_dropped, block_dropped);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        if (s_active == 0) {
            atomicAdd(out_full_prune_blocks, 1);
        }
        atomicAdd(out_active_threads, (unsigned long long)s_active);
        atomicAdd(out_bound_pass_threads, (unsigned long long)s_active);
    }

    if (threadIdx.x < s_count) {
        int out_pos = s_base + threadIdx.x;
        if (out_pos < out_capacity) {
            out_entropy_q[out_pos] = s_entropy_q[threadIdx.x];
            out_i[out_pos] = s_i[threadIdx.x];
            out_j[out_pos] = s_j[threadIdx.x];
            out_k[out_pos] = s_k[threadIdx.x];
        }
    }
}

extern "C" {

void launch_two_guess_search_codex(
    const uint8_t* d_matrix_u8,
    int n_allowed,
    int n_answers,
    const int* d_sorted_indices,
    const float* d_sorted_entropies,
    long long task_start,
    int n_tasks,
    const int* d_diag_offsets,
    int n_diags,
    int decode_mode,
    int* d_floor_q,
    int* d_out_count,
    int out_capacity,
    int* d_out_overflow_flag,
    int* d_out_overflow_dropped,
    int* d_out_full_prune_blocks,
    unsigned long long* d_out_active_threads,
    unsigned long long* d_out_bound_pass_threads,
    int* d_out_entropy_q,
    int* d_out_i,
    int* d_out_j
) {
    int threads_per_block = BLOCK_THREADS;
    int num_blocks = (n_tasks + threads_per_block - 1) / threads_per_block;

    two_guess_search_kernel_codex<<<num_blocks, threads_per_block>>>(
        d_matrix_u8,
        n_allowed,
        n_answers,
        d_sorted_indices,
        d_sorted_entropies,
        task_start,
        n_tasks,
        d_diag_offsets,
        n_diags,
        decode_mode,
        d_floor_q,
        d_out_count,
        out_capacity,
        d_out_overflow_flag,
        d_out_overflow_dropped,
        d_out_full_prune_blocks,
        d_out_active_threads,
        d_out_bound_pass_threads,
        d_out_entropy_q,
        d_out_i,
        d_out_j
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

void launch_three_guess_search_codex(
    const uint8_t* d_matrix_u8,
    int n_allowed,
    int n_answers,
    const int* d_sorted_indices,
    const float* d_sorted_entropies,
    long long task_start,
    int n_tasks,
    int* d_floor_q,
    int* d_out_count,
    int out_capacity,
    int* d_out_overflow_flag,
    int* d_out_overflow_dropped,
    int* d_out_full_prune_blocks,
    unsigned long long* d_out_active_threads,
    unsigned long long* d_out_bound_pass_threads,
    int* d_out_entropy_q,
    int* d_out_i,
    int* d_out_j,
    int* d_out_k
) {
    int threads_per_block = BLOCK_THREADS;
    int num_blocks = (n_tasks + threads_per_block - 1) / threads_per_block;

    three_guess_search_kernel_codex<<<num_blocks, threads_per_block>>>(
        d_matrix_u8,
        n_allowed,
        n_answers,
        d_sorted_indices,
        d_sorted_entropies,
        task_start,
        n_tasks,
        d_floor_q,
        d_out_count,
        out_capacity,
        d_out_overflow_flag,
        d_out_overflow_dropped,
        d_out_full_prune_blocks,
        d_out_active_threads,
        d_out_bound_pass_threads,
        d_out_entropy_q,
        d_out_i,
        d_out_j,
        d_out_k
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

void launch_three_guess_search_indexed_codex(
    const uint8_t* d_matrix_u8,
    int n_allowed,
    int n_answers,
    const int* d_sorted_indices,
    const int* d_task_i_sorted,
    const int* d_task_j_sorted,
    const int* d_task_k_sorted,
    int n_tasks,
    int* d_floor_q,
    int* d_out_count,
    int out_capacity,
    int* d_out_overflow_flag,
    int* d_out_overflow_dropped,
    int* d_out_full_prune_blocks,
    unsigned long long* d_out_active_threads,
    unsigned long long* d_out_bound_pass_threads,
    int* d_out_entropy_q,
    int* d_out_i,
    int* d_out_j,
    int* d_out_k
) {
    int threads_per_block = BLOCK_THREADS;
    int num_blocks = (n_tasks + threads_per_block - 1) / threads_per_block;

    three_guess_search_kernel_indexed_codex<<<num_blocks, threads_per_block>>>(
        d_matrix_u8,
        n_allowed,
        n_answers,
        d_sorted_indices,
        d_task_i_sorted,
        d_task_j_sorted,
        d_task_k_sorted,
        n_tasks,
        d_floor_q,
        d_out_count,
        out_capacity,
        d_out_overflow_flag,
        d_out_overflow_dropped,
        d_out_full_prune_blocks,
        d_out_active_threads,
        d_out_bound_pass_threads,
        d_out_entropy_q,
        d_out_i,
        d_out_j,
        d_out_k
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

void launch_compute_two_guess_entropy_q_codex(
    const uint8_t* d_matrix_u8,
    int n_allowed,
    int n_answers,
    const int* d_sorted_indices,
    long long task_start,
    int n_tasks,
    int* d_out_entropy_q
) {
    int threads_per_block = BLOCK_THREADS;
    int num_blocks = (n_tasks + threads_per_block - 1) / threads_per_block;

    compute_two_guess_entropy_q_kernel_codex<<<num_blocks, threads_per_block>>>(
        d_matrix_u8,
        n_allowed,
        n_answers,
        d_sorted_indices,
        task_start,
        n_tasks,
        d_out_entropy_q
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

}  // extern "C"
