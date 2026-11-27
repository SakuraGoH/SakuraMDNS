// mdns/samplers/cpp/samplers.cpp
//
// Fast CPU MCMC samplers for 2D Ising / q-state Potts models with periodic
// boundary conditions.  Single translation unit, flat C ABI, no dependencies
// beyond the C++17 standard library + OpenMP (optional but recommended).
//
// Build (Windows, MSVC):
//   scripts/build_cpp.ps1        -> mdns/samplers/build/mdns_samplers.dll
//
// Conventions (must match the python reference implementations):
//   Ising : spin storage is 0/1 (token convention), mapped to +-1 as 2*s-1.
//           H = -J * sum_{<ij>} s_i s_j - h * sum_i s_i
//   Potts : state storage is 0..q-1.
//           H = -J * (#edges with equal endpoints)
//
// All kernels are parallelised over independent chains (OpenMP), which is the
// only source of parallelism available for these Markov chains without
// changing their law.

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

#if defined(_OPENMP)
#include <omp.h>
#endif

#ifdef _WIN32
#define MDNS_API extern "C" __declspec(dllexport)
#else
#define MDNS_API extern "C" __attribute__((visibility("default")))
#endif

namespace {

// ---------------------------------------------------------------- RNG ------
// xoshiro128+ seeded through splitmix64.  Deterministic per (seed, chain).
struct Rng {
    uint32_t s[4];

    explicit Rng(uint64_t seed) {
        uint64_t z = seed + 0x9E3779B97F4A7C15ull;
        for (int i = 0; i < 4; ++i) {
            z += 0x9E3779B97F4A7C15ull;
            uint64_t x = z;
            x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
            x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
            x = x ^ (x >> 31);
            s[i] = static_cast<uint32_t>(x);
        }
    }

    static inline uint32_t rotl(uint32_t x, int k) {
        return (x << k) | (x >> (32 - k));
    }

    inline uint32_t next() {
        const uint32_t r = rotl(s[1] * 5u, 7) * 9u;
        const uint32_t t = s[1] << 9;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 11);
        return r;
    }

    // [0, 1) with 24 bits of mantissa, plenty for Metropolis.
    inline double uniform() { return (next() >> 8) * (1.0 / 16777216.0); }
    // Unbiased-enough multiply-shift; avoids an integer division per call.
    inline int below(int n) {
        return static_cast<int>((static_cast<uint64_t>(next()) * static_cast<uint64_t>(n)) >> 32);
    }
};

// Neighbour indices without division: callers only ever step by one lattice
// unit, so a comparison + add/substitute is enough (measurably faster than %).
inline int idx_up(int i, int j, int L) { return (i == 0 ? L - 1 : i - 1) * L + j; }
inline int idx_down(int i, int j, int L) { return (i + 1 == L ? 0 : i + 1) * L + j; }
inline int idx_left(int i, int j, int L) { return i * L + (j == 0 ? L - 1 : j - 1); }
inline int idx_right(int i, int j, int L) { return i * L + (j + 1 == L ? 0 : j + 1); }

inline int wrap(int x, int L) {
    x %= L;
    return x < 0 ? x + L : x;
}

inline uint64_t chain_seed(uint64_t seed, int b) {
    return seed * 0x9E3779B97F4A7C15ull + static_cast<uint64_t>(b) * 0x1D8E4E27C47D124Full + 1;
}

// ------------------------------------------------------------- union-find --
inline int find_root(std::vector<int>& parent, int x) {
    int r = x;
    while (parent[r] != r) r = parent[r];
    while (parent[x] != r) {
        const int nx = parent[x];
        parent[x] = r;
        x = nx;
    }
    return r;
}

inline void unite(std::vector<int>& parent, std::vector<int>& rnk, int a, int b) {
    int ra = find_root(parent, a);
    int rb = find_root(parent, b);
    if (ra == rb) return;
    if (rnk[ra] < rnk[rb]) std::swap(ra, rb);
    parent[rb] = ra;
    if (rnk[ra] == rnk[rb]) ++rnk[ra];
}

// ------------------------------------------------------------ observables --
// Total energy of one Ising configuration (same convention as ising2d_ham).
inline double ising_energy(const int8_t* s, int L, double J, double h) {
    double e = 0.0;
    for (int i = 0; i < L; ++i) {
        const int row = i * L;
        const int rowd = wrap(i + 1, L) * L;
        for (int j = 0; j < L; ++j) {
            const int a = 2 * static_cast<int>(s[row + j]) - 1;
            const int r = 2 * static_cast<int>(s[row + wrap(j + 1, L)]) - 1;
            const int d = 2 * static_cast<int>(s[rowd + j]) - 1;
            e -= J * (a * r + a * d);
            e -= h * a;
        }
    }
    return e;
}

inline int ising_neighbour_sum(const int8_t* s, int i, int j, int L) {
    return (2 * static_cast<int>(s[idx_up(i, j, L)]) - 1)
         + (2 * static_cast<int>(s[idx_down(i, j, L)]) - 1)
         + (2 * static_cast<int>(s[idx_left(i, j, L)]) - 1)
         + (2 * static_cast<int>(s[idx_right(i, j, L)]) - 1);
}

inline double potts_energy(const int32_t* s, int L, double J) {
    long long eq = 0;
    for (int i = 0; i < L; ++i) {
        const int row = i * L;
        const int rowd = wrap(i + 1, L) * L;
        for (int j = 0; j < L; ++j) {
            const int32_t a = s[row + j];
            if (a == s[row + wrap(j + 1, L)]) ++eq;
            if (a == s[rowd + j]) ++eq;
        }
    }
    return -J * static_cast<double>(eq);
}

inline int potts_matches(const int32_t* s, int i, int j, int L, int32_t state) {
    return static_cast<int>(s[idx_up(i, j, L)] == state)
         + static_cast<int>(s[idx_down(i, j, L)] == state)
         + static_cast<int>(s[idx_left(i, j, L)] == state)
         + static_cast<int>(s[idx_right(i, j, L)] == state);
}

// --------------------------------------------------------------- kernels --
// Every kernel signature is identical so the ctypes layer stays trivial.
//   S        : in/out configuration buffer, [B, L*L]
//   q        : number of states (unused for Ising)
//   e_series : optional out buffer [B, sweeps], per-chain energy after each sweep
//   seed     : base RNG seed

void ising_mh_kernel(int8_t* S, int B, int L, double beta, double J, double h,
                     int sweeps, double* e_series, uint64_t seed) {
    const int D = L * L;
    const double invT = beta;
#pragma omp parallel for schedule(static)
    for (int b = 0; b < B; ++b) {
        Rng rng(chain_seed(seed, b));
        int8_t* s = S + static_cast<size_t>(b) * D;
        double* es = e_series ? e_series + static_cast<size_t>(b) * sweeps : nullptr;
        for (int sw = 0; sw < sweeps; ++sw) {
            for (int n = 0; n < D; ++n) {
                const int i = rng.below(L);
                const int j = rng.below(L);
                const int spin = 2 * static_cast<int>(s[i * L + j]) - 1;
                const double dH = 2.0 * J * spin * ising_neighbour_sum(s, i, j, L)
                                + 2.0 * h * spin;
                if (dH <= 0.0 || rng.uniform() < std::exp(-invT * dH)) {
                    s[i * L + j] = static_cast<int8_t>(1 - s[i * L + j]);
                }
            }
            if (es) es[sw] = ising_energy(s, L, J, h);
        }
    }
}

void ising_sw_kernel(int8_t* S, int B, int L, double beta, double J, double h,
                     int sweeps, double* e_series, uint64_t seed) {
    const int D = L * L;
    const double p = 1.0 - std::exp(-2.0 * beta * J);
#pragma omp parallel
    {
        std::vector<int> parent(D), rnk(D);
        std::vector<char> flip(D);  // 0 unset, 1 flip, 2 keep
#pragma omp for schedule(static)
        for (int b = 0; b < B; ++b) {
            Rng rng(chain_seed(seed, b));
            int8_t* s = S + static_cast<size_t>(b) * D;
            double* es = e_series ? e_series + static_cast<size_t>(b) * sweeps : nullptr;
            for (int sw = 0; sw < sweeps; ++sw) {
                for (int k = 0; k < D; ++k) { parent[k] = k; rnk[k] = 0; }
                for (int i = 0; i < L; ++i) {
                    const int row = i * L;
                    const int rowd = wrap(i + 1, L) * L;
                    for (int j = 0; j < L; ++j) {
                        const int k = row + j;
                        const int kr = row + wrap(j + 1, L);
                        const int kd = rowd + j;
                        if (s[k] == s[kr] && rng.uniform() < p) unite(parent, rnk, k, kr);
                        if (s[k] == s[kd] && rng.uniform() < p) unite(parent, rnk, k, kd);
                    }
                }
                std::fill(flip.begin(), flip.end(), static_cast<char>(0));
                for (int k = 0; k < D; ++k) {
                    const int r = find_root(parent, k);
                    if (flip[r] == 0) flip[r] = (rng.uniform() < 0.5) ? 1 : 2;
                    if (flip[r] == 1) s[k] = static_cast<int8_t>(1 - s[k]);
                }
                if (es) es[sw] = ising_energy(s, L, J, h);
            }
        }
    }
}

void potts_mh_kernel(int32_t* S, int B, int L, int q, double beta, double J,
                     int sweeps, double* e_series, uint64_t seed) {
    const int D = L * L;
#pragma omp parallel for schedule(static)
    for (int b = 0; b < B; ++b) {
        Rng rng(chain_seed(seed, b));
        int32_t* s = S + static_cast<size_t>(b) * D;
        double* es = e_series ? e_series + static_cast<size_t>(b) * sweeps : nullptr;
        for (int sw = 0; sw < sweeps; ++sw) {
            for (int n = 0; n < D; ++n) {
                const int i = rng.below(L);
                const int j = rng.below(L);
                const int32_t cur = s[i * L + j];
                // Uniform over the q-1 states != cur.  Mapping a uniform
                // draw t in [0,q-1) to the t-th state other than cur keeps the
                // proposal symmetric, which plain Metropolis relies on.  (The
                // naive `draw; if equal then (nw+1)%q` gives cur+1 twice the
                // mass and silently biases the stationary distribution.)
                int32_t nw = rng.below(q - 1);
                if (nw >= cur) ++nw;
                const int m_old = potts_matches(s, i, j, L, cur);
                const int m_new = potts_matches(s, i, j, L, nw);
                const double dE = -beta * J * static_cast<double>(m_new - m_old);
                if (dE <= 0.0 || rng.uniform() < std::exp(-dE)) s[i * L + j] = nw;
            }
            if (es) es[sw] = potts_energy(s, L, J);
        }
    }
}

void potts_glauber_kernel(int32_t* S, int B, int L, int q, double beta, double J,
                          int sweeps, double* e_series, uint64_t seed) {
    const int D = L * L;
#pragma omp parallel
    {
        std::vector<double> w(q > 0 ? q : 1);
#pragma omp for schedule(static)
        for (int b = 0; b < B; ++b) {
            Rng rng(chain_seed(seed, b));
            int32_t* s = S + static_cast<size_t>(b) * D;
            double* es = e_series ? e_series + static_cast<size_t>(b) * sweeps : nullptr;
            for (int sw = 0; sw < sweeps; ++sw) {
                for (int n = 0; n < D; ++n) {
                    const int i = rng.below(L);
                    const int j = rng.below(L);
                    double total = 0.0;
                    for (int c = 0; c < q; ++c) {
                        const int m = potts_matches(s, i, j, L, c);
                        w[c] = std::exp(beta * J * static_cast<double>(m));
                        total += w[c];
                    }
                    double u = rng.uniform() * total;
                    int chosen = 0;
                    while (chosen < q - 1 && w[chosen] < u) { u -= w[chosen]; ++chosen; }
                    s[i * L + j] = chosen;
                }
                if (es) es[sw] = potts_energy(s, L, J);
            }
        }
    }
}

void potts_sw_kernel(int32_t* S, int B, int L, int q, double beta, double J,
                     int sweeps, double* e_series, uint64_t seed) {
    const int D = L * L;
    const double p = 1.0 - std::exp(-beta * J);
#pragma omp parallel
    {
        std::vector<int> parent(D), rnk(D), new_state(D);
#pragma omp for schedule(static)
        for (int b = 0; b < B; ++b) {
            Rng rng(chain_seed(seed, b));
            int32_t* s = S + static_cast<size_t>(b) * D;
            double* es = e_series ? e_series + static_cast<size_t>(b) * sweeps : nullptr;
            for (int sw = 0; sw < sweeps; ++sw) {
                for (int k = 0; k < D; ++k) { parent[k] = k; rnk[k] = 0; }
                for (int i = 0; i < L; ++i) {
                    const int row = i * L;
                    const int rowd = wrap(i + 1, L) * L;
                    for (int j = 0; j < L; ++j) {
                        const int k = row + j;
                        const int kr = row + wrap(j + 1, L);
                        const int kd = rowd + j;
                        if (s[k] == s[kr] && rng.uniform() < p) unite(parent, rnk, k, kr);
                        if (s[k] == s[kd] && rng.uniform() < p) unite(parent, rnk, k, kd);
                    }
                }
                std::fill(new_state.begin(), new_state.end(), -1);
                for (int k = 0; k < D; ++k) {
                    const int r = find_root(parent, k);
                    if (new_state[r] < 0) new_state[r] = rng.below(q);
                    s[k] = new_state[r];
                }
                if (es) es[sw] = potts_energy(s, L, J);
            }
        }
    }
}

}  // namespace

// ------------------------------------------------------------- C ABI -------
// Unified signature: (S, B, L, q, beta, J, h, sweeps, e_series, seed)

MDNS_API void mdns_ising_mh(void* S, int B, int L, int /*q*/, double beta, double J,
                            double h, int sweeps, void* e_series, uint64_t seed) {
    ising_mh_kernel(static_cast<int8_t*>(S), B, L, beta, J, h, sweeps,
                    static_cast<double*>(e_series), seed);
}

MDNS_API void mdns_ising_sw(void* S, int B, int L, int /*q*/, double beta, double J,
                            double h, int sweeps, void* e_series, uint64_t seed) {
    ising_sw_kernel(static_cast<int8_t*>(S), B, L, beta, J, h, sweeps,
                    static_cast<double*>(e_series), seed);
}

MDNS_API void mdns_potts_mh(void* S, int B, int L, int q, double beta, double J,
                            double /*h*/, int sweeps, void* e_series, uint64_t seed) {
    potts_mh_kernel(static_cast<int32_t*>(S), B, L, q, beta, J, sweeps,
                    static_cast<double*>(e_series), seed);
}

MDNS_API void mdns_potts_glauber(void* S, int B, int L, int q, double beta, double J,
                                 double /*h*/, int sweeps, void* e_series, uint64_t seed) {
    potts_glauber_kernel(static_cast<int32_t*>(S), B, L, q, beta, J, sweeps,
                         static_cast<double*>(e_series), seed);
}

MDNS_API void mdns_potts_sw(void* S, int B, int L, int q, double beta, double J,
                            double /*h*/, int sweeps, void* e_series, uint64_t seed) {
    potts_sw_kernel(static_cast<int32_t*>(S), B, L, q, beta, J, sweeps,
                    static_cast<double*>(e_series), seed);
}

MDNS_API void mdns_set_threads(int n) {
#if defined(_OPENMP)
    omp_set_num_threads(n);
#else
    (void)n;
#endif
}

MDNS_API int mdns_threads() {
#if defined(_OPENMP)
    return omp_get_max_threads();
#else
    return 1;
#endif
}

MDNS_API const char* mdns_version() {
#if defined(_OPENMP)
    return "mdns-samplers-1.0 openmp=on";
#else
    return "mdns-samplers-1.0 openmp=off";
#endif
}
