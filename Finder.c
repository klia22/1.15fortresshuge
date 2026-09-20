/*
Requirement
rx has exactly 2 trailing ones	rx ≡ 3 (mod 8): 3, 11, 19, …
or exactly 3 trailing ones	rx ≡ 7 (mod 16): 7, 23, …
rz has at least 5 trailing ones	rz ≡ 31 (mod 32): 31, 63, 95, …

Preset rx and rz do not meet these.
*/


#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <signal.h>
#include <stdatomic.h>
#include "fortress.h"

#define MULT   0x5DEECE66DULL
#define ADD    0xBULL

#define TOP_N 10
#define REGION_COUNT 4
#define LUT_BUCKETS 3          // residue classes of Q2 mod 3
#define CELLS_PER_DIM 16       // LUT grid resolution along Q3 (x pin) and Q4 (z pin)
#define MIN_FORTRESS_PIECES 160
#define CHUNK_BATCH 4096ULL
#define H_BATCH 512ULL
#define AUTO_RX_OPTIONS 2
#define AUTO_RX_2TRAIL  3   // rx = 3 (mod 8): rx ^ (rx+1) == 0x07
#define AUTO_RX_3TRAIL  7   // rx = 7 (mod 16): rx ^ (rx+1) == 0x0f
#define AUTO_RZ_MIN_TRAIL 5 // rz may have any number of trailing ones >= 5

static uint64_t g_feed_deltas[REGION_COUNT] = {0, 1, 48, 49};
static int g_corner_rx = 0;
static int g_corner_rz = 1;
static uint64_t g_base_feed_delta = 16;
static int g_test_mode;

// ----- Required fortress start offsets (the nextInt(8) x/z draws) -----
// g_pin_x[d] / g_pin_z[d] in 0..7 pins that region's offset to exactly that value
// and it is enforced INSIDE the MITM LUT.  -1 means "any value".
// Region index d: 0=(rx,rz) "lo", 1=(rx+1,rz), 2=(rx,rz+1), 3=(rx+1,rz+1) "hi".
// Default: all four regions pinned to their outer-most (corner) offsets:
//   region 0 (rx,rz)     -> offset (0,0)
//   region 1 (rx+1,rz)   -> offset (7,0)
//   region 2 (rx,rz+1)   -> offset (0,7)
//   region 3 (rx+1,rz+1) -> offset (7,7)
static int g_pin_x[REGION_COUNT] = {7, 0, 7, 0};
static int g_pin_z[REGION_COUNT] = {7, 7, 0, 0};

static int      g_b;
static uint64_t g_lower_size;
static uint64_t g_upper_size;
static uint64_t g_lower_mask;
static uint64_t g_upper_mask;
static uint64_t g_W;            // upper_size / 8 : width of one "top 3 bits" bin
static int      g_sc;
static int      g_sc_inv;
static int      g_pc;
static uint64_t g_Ak[5], g_Bk[5];
static int      g_use_x, g_use_z;
static int      g_G3, g_G4;      // grid cells along Q3 / Q4 (1 if that axis is unpinned)
static uint64_t g_cellw3, g_cellw4;
static size_t   g_nb;            // total LUT buckets = 3 * G3 * G4

typedef struct {
    uint32_t L;
    uint32_t rlo;   // valid Q2 range (carry pattern + residue class)
    uint32_t rhi;
    uint32_t s3;    // valid Q3 arc: (Q3 - s3) mod U < len3   (x-offset pins)
    uint32_t len3;
    uint32_t s4;    // valid Q4 arc: (Q4 - s4) mod U < len4   (z-offset pins)
    uint32_t len4;
} LEntry;

typedef struct {
    LEntry *data;
    size_t count;
    size_t cap;
} EntryVec;

typedef struct {
    LEntry *data;
    uint32_t *prefix_max_rhi;
    size_t count;
} LutBucket;

static LutBucket *g_lut;
static uint64_t g_lut_entries = 0;
static uint64_t g_lut_bytes = 0;

// ----- Global progress -----
static atomic_int g_phase; // 0 init, 1 phase1, 2 sort, 3 phase2, 4 done
static atomic_uint_fast64_t g_p1_next;
static atomic_uint_fast64_t g_p2_next;
static atomic_uint_fast64_t g_p1_done;
static atomic_uint_fast64_t g_p2_done;
static atomic_uint_fast64_t g_mitm_seeds;
static atomic_uint_fast64_t g_total_evals;
static atomic_uint_fast64_t g_selfcheck_fail;
static atomic_uint_fast64_t g_test_queries;
static atomic_uint_fast64_t g_test_matches;
static atomic_uint_fast64_t g_test_lookup_ns;
static atomic_uint_fast64_t g_test_fortress_ns;
static volatile sig_atomic_t g_stop_requested;

static void free_lut(void)
{
    if (!g_lut)
        return;
    for (size_t r = 0; r < g_nb; ++r) {
        free(g_lut[r].data);
        free(g_lut[r].prefix_max_rhi);
    }
    free(g_lut);
    g_lut = NULL;
    g_lut_entries = 0;
    g_lut_bytes = 0;
}

// ----- Automatic corner enumeration -----
//
// With the default four pins, valid corners have either
//   rx = 3 (mod 8)   -> rx ^ (rx+1) = 0x07
//   rx = 7 (mod 16)  -> rx ^ (rx+1) = 0x0f
// and rz with at least five trailing one bits.  For a fixed trailing-one
// count t, every rz = (2^t - 1) + k*2^(t+1) preserves the same z delta.
// Likewise, stepping rx by 8 or 16 preserves its x delta class.
//
// There are only a limited number of distinct RNG-delta signatures, but there
// are vastly more distinct absolute corners.  Automatic mode therefore uses a
// round-robin enumeration: the first round contains one corner for every
// distinct signature, and later rounds continue with fresh absolute corners.
// This makes --regions 100, 1000, etc. useful instead of imposing an arbitrary
// 44-iteration ceiling.
static uint64_t automatic_max_trailing(void)
{
    const uint64_t max_region = (uint64_t)(INT_MAX / 32);
    uint64_t coord_max = 0;
    while (coord_max < 62 &&
           ((UINT64_C(1) << (coord_max + 1)) - 1) <= max_region)
        ++coord_max;

    uint64_t lut_max = (uint64_t)g_b - 5;
    return lut_max < coord_max ? lut_max : coord_max;
}

static uint64_t automatic_signature_count(void)
{
    const uint64_t max_trailing = automatic_max_trailing();
    if (max_trailing < AUTO_RZ_MIN_TRAIL)
        return 0;
    return 2 * (max_trailing - AUTO_RZ_MIN_TRAIL + 1);
}

static uint64_t automatic_signature_capacity(uint64_t sig)
{
    const uint64_t max_region = (uint64_t)(INT_MAX / 32);
    const uint64_t trail = AUTO_RZ_MIN_TRAIL + sig / 2;
    const uint64_t rx_base = (sig & 1) ? AUTO_RX_3TRAIL : AUTO_RX_2TRAIL;
    const uint64_t rx_step = (rx_base == AUTO_RX_2TRAIL) ? 8 : 16;
    const uint64_t rz_base = (UINT64_C(1) << trail) - 1;
    const uint64_t rz_step = UINT64_C(1) << (trail + 1);

    if (rx_base > max_region || rz_base > max_region)
        return 0;

    const uint64_t rx_count = (max_region - rx_base) / rx_step + 1;
    const uint64_t rz_count = (max_region - rz_base) / rz_step + 1;
    if (rx_count != 0 && rz_count > UINT64_MAX / rx_count)
        return UINT64_MAX;
    return rx_count * rz_count;
}

static uint64_t automatic_corner_capacity(void)
{
    const uint64_t sig_count = automatic_signature_count();
    uint64_t total = 0;
    for (uint64_t sig = 0; sig < sig_count; ++sig) {
        const uint64_t n = automatic_signature_capacity(sig);
        if (UINT64_MAX - total < n)
            return UINT64_MAX;
        total += n;
    }
    return total;
}

// Number of corners contributed by all signatures in rounds [0, round).
static uint64_t automatic_count_before_round(uint64_t round, uint64_t sig_count)
{
    uint64_t total = 0;
    for (uint64_t sig = 0; sig < sig_count; ++sig) {
        const uint64_t cap = automatic_signature_capacity(sig);
        const uint64_t add = round < cap ? round : cap;
        if (UINT64_MAX - total < add)
            return UINT64_MAX;
        total += add;
    }
    return total;
}

static int automatic_corner(uint64_t iteration, int *rx_out, int *rz_out)
{
    const uint64_t max_region = (uint64_t)(INT_MAX / 32);
    const uint64_t sig_count = automatic_signature_count();
    const uint64_t capacity = automatic_corner_capacity();
    if (sig_count == 0 || iteration >= capacity)
        return 0;

    // Find the round containing this iteration.  At round r, every signature
    // with capacity > r contributes exactly one corner.
    uint64_t lo = 0;
    uint64_t hi = 1;
    while (hi < capacity && automatic_count_before_round(hi, sig_count) <= iteration) {
        if (hi > UINT64_MAX / 2)
            break;
        hi *= 2;
    }
    if (hi > capacity)
        hi = capacity;

    while (lo < hi) {
        const uint64_t mid = lo + (hi - lo) / 2;
        if (automatic_count_before_round(mid, sig_count) <= iteration)
            lo = mid + 1;
        else
            hi = mid;
    }

    const uint64_t round = lo == 0 ? 0 : lo - 1;
    const uint64_t before = automatic_count_before_round(round, sig_count);
    uint64_t pos = iteration - before;

    uint64_t sig = 0;
    for (; sig < sig_count; ++sig) {
        if (automatic_signature_capacity(sig) > round) {
            if (pos == 0)
                break;
            --pos;
        }
    }
    if (sig >= sig_count)
        return 0;

    const uint64_t trail = AUTO_RZ_MIN_TRAIL + sig / 2;
    const uint64_t rx_base = (sig & 1) ? AUTO_RX_3TRAIL : AUTO_RX_2TRAIL;
    const uint64_t rx_step = (rx_base == AUTO_RX_2TRAIL) ? 8 : 16;
    const uint64_t rz_base = (UINT64_C(1) << trail) - 1;
    const uint64_t rz_step = UINT64_C(1) << (trail + 1);
    const uint64_t rx_count = (max_region - rx_base) / rx_step + 1;

    const uint64_t rx_index = round % rx_count;
    const uint64_t rz_index = round / rx_count;
    const uint64_t rx = rx_base + rx_index * rx_step;
    const uint64_t rz = rz_base + rz_index * rz_step;

    if (rx > max_region || rz > max_region)
        return 0;

    *rx_out = (int)rx;
    *rz_out = (int)rz;
    return 1;
}

static void handle_sigint(int signal_number)
{
    (void)signal_number;
    g_stop_requested = 1;
    signal(SIGINT, handle_sigint);
}

// ----- Detailed result types -----
typedef struct {
    char name[32];
    int type;
    int x, y, z;
    int rot;
    int depth;
} PieceBrief;

typedef struct {
    int region_x;
    int region_z;
    int chunk_x;
    int chunk_z;
    uint64_t feed_seed;
    uint64_t seeded_state;
    uint64_t after_set_attempt;
    uint64_t rng_s0;
    uint64_t rotation_state;
    int rotation;
    int piece_count;
    uint16_t type_counts[FORTRESS_PIECE_COUNT];
    int brief_count;
    PieceBrief brief[16];
} RegionDetail;

typedef struct {
    uint64_t world_seed;
    int corner_rx;
    int corner_rz;
    int four_piece_count;
    int total_pieces;
    RegionDetail region[REGION_COUNT];
} SearchResult;

typedef struct {
    uint64_t world_seed;
    int four_piece_count;
    int total_pieces;
    int piece_count[REGION_COUNT];
    int chunk_x[REGION_COUNT];
    int chunk_z[REGION_COUNT];
} CandidateSummary;

typedef struct {
    EntryVec *v;               // g_nb vectors, one per LUT bucket
    SearchResult top[TOP_N];
    int top_n;
    uint64_t local_evals;
} WorkerCtx;

// ----- Utility -----
static void die(const char *msg)
{
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(1);
}

static void *xmalloc(size_t n)
{
    void *p = malloc(n);
    if (!p) {
        fprintf(stderr, "ERROR: malloc(%zu) failed: %s\n", n, strerror(errno));
        exit(1);
    }
    return p;
}

static void *xcalloc(size_t n, size_t sz)
{
    if (sz != 0 && n > SIZE_MAX / sz)
        die("calloc size overflow");
    void *p = calloc(n, sz);
    if (!p) {
        fprintf(stderr, "ERROR: calloc(%zu,%zu) failed: %s\n", n, sz, strerror(errno));
        exit(1);
    }
    return p;
}

static void *xrealloc(void *old, size_t n)
{
    void *p = realloc(old, n);
    if (!p) {
        fprintf(stderr, "ERROR: realloc(%zu) failed: %s\n", n, strerror(errno));
        exit(1);
    }
    return p;
}

static inline uint64_t atomic_load_u64(const atomic_uint_fast64_t *p)
{
    return atomic_load_explicit(p, memory_order_relaxed);
}

static inline uint64_t lcg(uint64_t s)
{
    return (s * MULT + ADD) & MASK48;
}

static uint64_t monotonic_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

static void vec_push(EntryVec *v, LEntry e)
{
    if (v->count == v->cap) {
        // Small start: there are hundreds of buckets per thread now.
        size_t nc = v->cap ? v->cap * 2 : 64;
        if (nc < v->cap || nc > SIZE_MAX / sizeof(LEntry))
            die("LUT vector size overflow");
        v->data = (LEntry *)xrealloc(v->data, nc * sizeof(LEntry));
        v->cap = nc;
    }
    v->data[v->count++] = e;
}

// ----- Setup and exact L-side math -----
// Set the base corner region (rx,rz).  The four regions are:
//   (rx,rz), (rx+1,rz), (rx,rz+1), (rx+1,rz+1).
// The base feed is k = worldSeed ^ (rx ^ (rz<<4)).  Relative feed deltas are
// computed exactly from those region indices, so odd-coordinate corners are
// handled instead of assuming {0,1,16,17}.
static void setup_corner(int rx, int rz)
{
    // Negative regions are fine (cx>>4 is an arithmetic shift, so the region index is
    // sign-extended exactly like the game does).  The one exception is a 2x2 block that
    // straddles 0 (rx == -1 or rz == -1): the feed delta then flips ALL high bits, which
    // the shared-H MITM cannot represent, and setup() rejects it.
    if (rx == INT_MAX || rx < -(INT_MAX / 32) || rx > INT_MAX / 32 ||
        rz == INT_MAX || rz < -(INT_MAX / 32) || rz > INT_MAX / 32)
        die("corner region coordinates are too large for the chunk-coordinate int math");

    g_corner_rx = rx;
    g_corner_rz = rz;
    uint64_t a = (uint64_t)rx;
    uint64_t b = (uint64_t)rz;
    uint64_t a1 = (uint64_t)(rx + 1);
    uint64_t b1 = (uint64_t)(rz + 1);
    g_base_feed_delta = (a ^ (b << 4)) & MASK48;
    g_feed_deltas[0] = 0;
    g_feed_deltas[1] = (a ^ a1) & MASK48;
    g_feed_deltas[2] = ((b ^ b1) << 4) & MASK48;
    g_feed_deltas[3] = g_feed_deltas[1] ^ g_feed_deltas[2];
}

static void setup(int bits)
{
    if (bits < 20 || bits > 32)
        die("--lut-bits must be in [20,32]");

    g_b = bits;
    g_lower_size = 1ULL << g_b;
    g_upper_size = 1ULL << (48 - g_b);
    g_lower_mask = g_lower_size - 1;
    g_upper_mask = g_upper_size - 1;
    g_W = g_upper_size >> 3;

    if (g_lower_size == 0 || g_upper_size == 0)
        die("internal split size became zero");

    // eval assumes the region deltas only touch the LOW b bits (the H side is
    // shared by all four regions).  Make that explicit instead of silently wrong.
    for (int d = 0; d < REGION_COUNT; ++d)
        if (g_feed_deltas[d] >= g_lower_size)
            die("corner too large for this --lut-bits: feed deltas must fit in the low bits");

    g_Ak[0] = 1;
    g_Bk[0] = 0;
    for (int i = 1; i <= 4; ++i) {
        g_Ak[i] = (g_Ak[i-1] * MULT) & MASK48;
        g_Bk[i] = (g_Bk[i-1] * MULT + ADD) & MASK48;
    }

    int t = 1;
    for (int i = 0; i < g_b - 17; ++i)
        t = (t * 2) % 3;
    g_sc = t;
    g_sc_inv = (t == 1) ? 1 : 2;

    t = 1;
    for (int i = 0; i < 48 - g_b; ++i)
        t = (t * 2) % 3;
    g_pc = t;

    // Offset pins -> extra LUT dimensions.
    g_use_x = g_use_z = 0;
    for (int d = 0; d < REGION_COUNT; ++d) {
        if (g_pin_x[d] >= 0) g_use_x = 1;
        if (g_pin_z[d] >= 0) g_use_z = 1;
    }
    g_G3 = g_use_x ? CELLS_PER_DIM : 1;
    g_G4 = g_use_z ? CELLS_PER_DIM : 1;
    g_cellw3 = g_upper_size / (uint64_t)g_G3;
    g_cellw4 = g_upper_size / (uint64_t)g_G4;
    g_nb = (size_t)LUT_BUCKETS * (size_t)g_G3 * (size_t)g_G4;
}

// ----- Cyclic arcs in Z_U (U = upper_size), used for the offset pins -----
// A pin "top 3 bits of state_k == v" for region d becomes, on the shared H-side
// value Qk = (A_k * Hx) mod U, the arc  Qk in [v*W - phi_k,d , v*W - phi_k,d + W)  (mod U),
// where phi_k,d is the high part of the L-side contribution.  Several regions'
// arcs are intersected here so the LUT stores ONE arc per L.
typedef struct {
    uint64_t start;
    uint64_t len;
} Arc;

// Intersect *cur with [start, start+len) (mod U). Valid because each arc is
// shorter than U/2, so the intersection is a single piece. Returns 0 if empty.
static int arc_and(Arc *cur, int *have, uint64_t start, uint64_t len)
{
    if (!*have) {
        cur->start = start;
        cur->len = len;
        *have = 1;
        return 1;
    }
    uint64_t delta = (start - cur->start) & g_upper_mask;
    int64_t cl = (int64_t)cur->len;
    int64_t nl = (int64_t)len;
    for (int k = 0; k < 2; ++k) {
        int64_t s = (int64_t)delta - (k ? (int64_t)g_upper_size : 0);
        int64_t lo = s > 0 ? s : 0;
        int64_t hi = (s + nl - 1 < cl - 1) ? (s + nl - 1) : (cl - 1);
        if (lo <= hi) {
            cur->start = (cur->start + (uint64_t)lo) & g_upper_mask;
            cur->len = (uint64_t)(hi - lo + 1);
            return 1;
        }
    }
    return 0;
}

// Which grid cells (of G equal cells over [0,U)) does the arc touch?
static void arc_cells(const Arc *a, int G, int *first, int *count)
{
    if (G == 1) {
        *first = 0;
        *count = 1;
        return;
    }
    uint64_t w = g_upper_size / (uint64_t)G;
    uint64_t c0 = a->start / w;
    uint64_t c1 = ((a->start + a->len - 1) & g_upper_mask) / w;
    *first = (int)c0;
    *count = (int)((c1 + (uint64_t)G - c0) % (uint64_t)G) + 1;
}

static inline int entry_arcs_ok(const LEntry *e, uint64_t Q3, uint64_t Q4)
{
    return ((Q3 - e->s3) & g_upper_mask) < e->len3 &&
           ((Q4 - e->s4) & g_upper_mask) < e->len4;
}

// L-side work for one L: applies the nextInt(3)==0 constraints (via carry
// patterns) AND the x/z offset pins, then stores entries in the LUT buckets.
static void emit_L(uint64_t L, EntryVec *vecs)
{
    uint64_t P2[REGION_COUNT];
    Arc ax = {0, g_upper_size}, az = {0, g_upper_size};
    int have_x = 0, have_z = 0;

    for (int d = 0; d < REGION_COUNT; ++d) {
        uint64_t Lx = L ^ (uint64_t)g_feed_deltas[d] ^ (MULT & g_lower_mask);
        P2[d] = (g_Ak[2] * Lx + g_Bk[2]) & MASK48;

        if (g_pin_x[d] >= 0) {
            // state after 3 LCG steps; its top 3 bits are the x offset
            uint64_t phi3 = ((g_Ak[3] * Lx + g_Bk[3]) & MASK48) >> g_b;
            uint64_t start = ((uint64_t)g_pin_x[d] * g_W + g_upper_size - phi3) & g_upper_mask;
            if (!arc_and(&ax, &have_x, start, g_W))
                return;
        }
        if (g_pin_z[d] >= 0) {
            // state after 4 LCG steps; its top 3 bits are the z offset
            uint64_t phi4 = ((g_Ak[4] * Lx + g_Bk[4]) & MASK48) >> g_b;
            uint64_t start = ((uint64_t)g_pin_z[d] * g_W + g_upper_size - phi4) & g_upper_mask;
            if (!arc_and(&az, &have_z, start, g_W))
                return;
        }
    }

    int c3first, c3n, c4first, c4n;
    arc_cells(&ax, g_G3, &c3first, &c3n);
    arc_cells(&az, g_G4, &c4first, &c4n);

    for (int cp = 0; cp < 16; ++cp) {
        int req_d[REGION_COUNT];
        for (int d = 0; d < REGION_COUNT; ++d) {
            uint64_t lo = P2[d] & g_lower_mask;
            uint64_t hi = P2[d] >> g_b;
            int Lc = (int)(((lo >> 17) % 3 + (uint64_t)g_sc * (hi % 3)) % 3);
            int cd = (cp >> d) & 1;
            int r = (-g_sc_inv * Lc + g_pc * cd) % 3;
            if (r < 0)
                r += 3;
            req_d[d] = r;
        }

        int same = 1;
        for (int d = 1; d < REGION_COUNT; ++d) {
            if (req_d[d] != req_d[0]) { same = 0; break; }
        }
        if (!same)
            continue;

        // Carry c_d = 1 iff Q2 >= T_d, where T_d = upper_size - P2_hi.
        uint64_t lo_q = 0;
        uint64_t hi_q = g_upper_size - 1;

        for (int d = 0; d < REGION_COUNT; ++d) {
            uint64_t phi = P2[d] >> g_b;
            uint64_t T = g_upper_size - phi;
            if ((cp >> d) & 1) {
                if (T > lo_q)
                    lo_q = T;
            } else {
                // T is in [1, upper_size] because phi < upper_size.
                if (T - 1 < hi_q)
                    hi_q = T - 1;
            }
        }

        if (lo_q > hi_q)
            continue;

        LEntry en;
        en.L = (uint32_t)L;
        en.rlo = (uint32_t)lo_q;
        en.rhi = (uint32_t)hi_q;
        en.s3 = (uint32_t)ax.start;
        en.len3 = (uint32_t)ax.len;
        en.s4 = (uint32_t)az.start;
        en.len4 = (uint32_t)az.len;

        int req = req_d[0];
        for (int i3 = 0; i3 < c3n; ++i3) {
            int c3 = (c3first + i3) % g_G3;
            for (int i4 = 0; i4 < c4n; ++i4) {
                int c4 = (c4first + i4) % g_G4;
                size_t idx = ((size_t)req * (size_t)g_G3 + (size_t)c3) * (size_t)g_G4 + (size_t)c4;
                vec_push(&vecs[idx], en);
            }
        }
    }
}

static void *phase1_worker(void *arg)
{
    WorkerCtx *ctx = (WorkerCtx *)arg;

    for (;;) {
        if (g_stop_requested)
            break;
        uint64_t s = atomic_fetch_add_explicit(&g_p1_next, CHUNK_BATCH, memory_order_relaxed);
        if (s >= g_lower_size)
            break;
        uint64_t e = s + CHUNK_BATCH;
        if (e > g_lower_size)
            e = g_lower_size;

        for (uint64_t L = s; L < e; ++L) {
            if (g_stop_requested)
                break;
            emit_L(L, ctx->v);
            atomic_fetch_add_explicit(&g_p1_done, 1, memory_order_relaxed);
        }
    }

    return NULL;
}

static int cmp_lut_entry(const void *pa, const void *pb)
{
    const LEntry *a = (const LEntry *)pa;
    const LEntry *b = (const LEntry *)pb;
    if (a->rlo < b->rlo) return -1;
    if (a->rlo > b->rlo) return 1;
    if (a->rhi < b->rhi) return -1;
    if (a->rhi > b->rhi) return 1;
    if (a->L < b->L) return -1;
    if (a->L > b->L) return 1;
    return 0;
}

static void build_lut(WorkerCtx *ctxs, int nthr)
{
    atomic_store_explicit(&g_phase, 2, memory_order_relaxed);
    g_lut = (LutBucket *)xcalloc(g_nb, sizeof(LutBucket));
    g_lut_entries = 0;
    g_lut_bytes = 0;

    for (size_t r = 0; r < g_nb; ++r) {
        uint64_t total = 0;
        for (int t = 0; t < nthr; ++t)
            total += (uint64_t)ctxs[t].v[r].count;

        if (total > SIZE_MAX / sizeof(LEntry))
            die("global LUT size exceeds size_t");
        g_lut[r].count = (size_t)total;
        if (total != 0)
            g_lut[r].data = (LEntry *)xmalloc((size_t)total * sizeof(LEntry));

        size_t off = 0;
        for (int t = 0; t < nthr; ++t) {
            size_t n = ctxs[t].v[r].count;
            if (n != 0) {
                memcpy(g_lut[r].data + off, ctxs[t].v[r].data, n * sizeof(LEntry));
                off += n;
            }
            free(ctxs[t].v[r].data);
            ctxs[t].v[r].data = NULL;
            ctxs[t].v[r].count = 0;
            ctxs[t].v[r].cap = 0;
        }

        if (g_lut[r].count != 0)
            qsort(g_lut[r].data, g_lut[r].count, sizeof(LEntry), cmp_lut_entry);

        if (g_lut[r].count != 0) {
            g_lut[r].prefix_max_rhi = (uint32_t *)xmalloc(g_lut[r].count * sizeof(uint32_t));
            uint32_t mx = 0;
            for (size_t i = 0; i < g_lut[r].count; ++i) {
                if (g_lut[r].data[i].rhi > mx)
                    mx = g_lut[r].data[i].rhi;
                g_lut[r].prefix_max_rhi[i] = mx;
            }
        }

        g_lut_entries += (uint64_t)g_lut[r].count;
        g_lut_bytes += (uint64_t)g_lut[r].count * (sizeof(LEntry) + sizeof(uint32_t));
    }
}

static size_t upper_bound_rlo(const LutBucket *b, uint32_t q)
{
    size_t lo = 0, hi = b->count;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (b->data[mid].rlo <= q)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

typedef void (*lut_match_cb)(const LEntry *e, void *opaque);

// Calls cb for every entry in the bucket whose Q2 range contains q.
// (The Q3/Q4 arc test is done by the callback via entry_arcs_ok().)
static void lut_query(const LutBucket *b, uint32_t q, lut_match_cb cb, void *opaque)
{
    if (b->count == 0)
        return;

    uint64_t query_start = g_test_mode ? monotonic_ns() : 0;
    if (g_test_mode)
        atomic_fetch_add_explicit(&g_test_queries, 1, memory_order_relaxed);

    size_t n = upper_bound_rlo(b, q);
    if (n == 0)
        return;

    size_t i = n - 1;
    for (;;) {
        if (g_stop_requested)
            break;
        if (b->data[i].rhi >= q) {
            if (g_test_mode)
                atomic_fetch_add_explicit(&g_test_matches, 1, memory_order_relaxed);
            cb(&b->data[i], opaque);
        }
        if (g_stop_requested)
            break;
        if (i == 0)
            break;
        if (b->prefix_max_rhi[i - 1] < q)
            break;
        --i;
    }

    if (g_test_mode)
        atomic_fetch_add_explicit(&g_test_lookup_ns,
                monotonic_ns() - query_start, memory_order_relaxed);
}

// H-side values for one H: Q2/Q3/Q4 and the LUT bucket they select.
static inline size_t h_bucket(uint64_t H, uint64_t *Q3, uint64_t *Q4, uint64_t *Q2out)
{
    uint64_t Hx = H ^ (MULT >> g_b);
    uint64_t Q2 = (g_Ak[2] * Hx) & g_upper_mask;
    *Q3 = (g_Ak[3] * Hx) & g_upper_mask;
    *Q4 = (g_Ak[4] * Hx) & g_upper_mask;
    *Q2out = Q2;
    size_t r = (size_t)(Q2 % 3);
    size_t c3 = g_G3 == 1 ? 0 : (size_t)(*Q3 / g_cellw3);
    size_t c4 = g_G4 == 1 ? 0 : (size_t)(*Q4 / g_cellw4);
    return (r * (size_t)g_G3 + c3) * (size_t)g_G4 + c4;
}

// ----- Fortress evaluation -----
// Full (exact) check of nextInt(3)==0 and the pinned offsets for one candidate,
// then the expensive fortress generation.  With the pins folded into the LUT the
// two self-checks below should never fire; they only guard against LUT bugs.
static int summarize_candidate(uint64_t base_feed, CandidateSummary *out)
{
    memset(out, 0, sizeof(*out));
    uint64_t ws = (base_feed ^ g_base_feed_delta) & MASK48;
    out->world_seed = ws;
    out->four_piece_count = 0;
    out->total_pieces = 0;

    for (int d = 0; d < REGION_COUNT; ++d) {
        uint64_t s = (base_feed ^ g_feed_deltas[d] ^ MULT) & MASK48;

        s = lcg(s);                         // setAttemptSeed(): next(31)
        s = lcg(s);                         // nextInt(3)
        if (((s >> 17) % 3) != 0) {
            atomic_fetch_add_explicit(&g_selfcheck_fail, 1, memory_order_relaxed);
            return 0;
        }

        s = lcg(s);                         // nextInt(8) for x
        int xv = (int)(s >> 45);
        s = lcg(s);                         // nextInt(8) for z
        int zv = (int)(s >> 45);

        if ((g_pin_x[d] >= 0 && xv != g_pin_x[d]) ||
            (g_pin_z[d] >= 0 && zv != g_pin_z[d])) {
            atomic_fetch_add_explicit(&g_selfcheck_fail, 1, memory_order_relaxed);
            return 0;
        }

        int rx = g_corner_rx + ((d == 1 || d == 3) ? 1 : 0);
        int rz = g_corner_rz + ((d == 2 || d == 3) ? 1 : 0);
        out->chunk_x[d] = rx * 16 + xv + 4;
        out->chunk_z[d] = rz * 16 + zv + 4;

        Piece list[400];
        int n = getFortressPieces(list, 400, MC_1_15, ws,
                                  out->chunk_x[d], out->chunk_z[d]);
        if (g_stop_requested)
            return 0;
        if (n < MIN_FORTRESS_PIECES)
            return 0;

        out->piece_count[d] = n;
        out->total_pieces += n;
        if (n == 4)
            ++out->four_piece_count;
    }

    return 1;
}

static void fill_region_detail(uint64_t ws, int d, int cx, int cz, RegionDetail *out)
{
    memset(out, 0, sizeof(*out));

    int rx = g_corner_rx + ((d == 1 || d == 3) ? 1 : 0);
    int rz = g_corner_rz + ((d == 2 || d == 3) ? 1 : 0);
    out->region_x = rx;
    out->region_z = rz;
    out->chunk_x = cx;
    out->chunk_z = cz;

    uint64_t region_delta = ((uint64_t)rx ^ ((uint64_t)rz << 4)) & MASK48;
    out->feed_seed = (ws ^ region_delta) & MASK48;
    out->seeded_state = (out->feed_seed ^ MULT) & MASK48;
    out->after_set_attempt = lcg(out->seeded_state);
    uint64_t s = out->after_set_attempt;
    s = lcg(s);
    s = lcg(s);
    s = lcg(s);
    out->rng_s0 = s;
    out->rotation_state = lcg(out->rng_s0);

    Piece list[400];
    int n = getFortressPieces(list, 400, MC_1_15, ws, cx, cz);
    out->piece_count = n;
    out->rotation = (n > 0) ? list[0].rot : -1;

    for (int i = 0; i < n; ++i) {
        int t = list[i].type;
        if (t >= 0 && t < FORTRESS_PIECE_COUNT)
            ++out->type_counts[t];
    }

    out->brief_count = n < 16 ? n : 16;
    for (int i = 0; i < out->brief_count; ++i) {
        PieceBrief *pb = &out->brief[i];
        const Piece *p = &list[i];
        snprintf(pb->name, sizeof(pb->name), "%s", p->name ? p->name : "?");
        pb->type = p->type;
        pb->x = p->pos.x;
        pb->y = p->pos.y;
        pb->z = p->pos.z;
        pb->rot = p->rot;
        pb->depth = p->depth;
    }
}

static void make_detailed_result(const CandidateSummary *sum, SearchResult *out)
{
    memset(out, 0, sizeof(*out));
    out->world_seed = sum->world_seed;
    out->corner_rx = g_corner_rx;
    out->corner_rz = g_corner_rz;
    out->four_piece_count = sum->four_piece_count;
    out->total_pieces = sum->total_pieces;

    for (int d = 0; d < REGION_COUNT; ++d)
        fill_region_detail(sum->world_seed, d, sum->chunk_x[d], sum->chunk_z[d], &out->region[d]);
}

static int summary_should_enter_top(const CandidateSummary *a, const SearchResult *b)
{
    if (a->total_pieces != b->total_pieces)
        return a->total_pieces > b->total_pieces;
    if (a->four_piece_count != b->four_piece_count)
        return a->four_piece_count > b->four_piece_count;
    return a->world_seed < b->world_seed;
}

static int result_equal_seed(const SearchResult *a, uint64_t ws)
{
    return a->world_seed == ws;
}

static int result_better(const SearchResult *a, const SearchResult *b)
{
    if (a->total_pieces != b->total_pieces)
        return a->total_pieces > b->total_pieces;
    if (a->four_piece_count != b->four_piece_count)
        return a->four_piece_count > b->four_piece_count;
    return a->world_seed < b->world_seed;
}

static void top_insert(SearchResult *top, int *top_n, const SearchResult *cand)
{
    for (int i = 0; i < *top_n; ++i) {
        if (result_equal_seed(&top[i], cand->world_seed))
            return;
    }

    int pos = *top_n;
    if (pos < TOP_N) {
        top[pos] = *cand;
        ++*top_n;
    } else {
        if (!result_better(cand, &top[TOP_N - 1]))
            return;
        top[TOP_N - 1] = *cand;
        pos = TOP_N - 1;
    }

    while (pos > 0 && result_better(&top[pos], &top[pos - 1])) {
        SearchResult tmp = top[pos - 1];
        top[pos - 1] = top[pos];
        top[pos] = tmp;
        --pos;
    }
}

// ----- Phase 2 -----
typedef struct {
    WorkerCtx *ctx;
    uint64_t H;
    uint64_t Q3;
    uint64_t Q4;
} CandidateCallbackCtx;

static void free_worker_vectors(WorkerCtx *ctxs, int nthr)
{
    for (int i = 0; i < nthr; ++i) {
        if (!ctxs[i].v)
            continue;
        for (size_t r = 0; r < g_nb; ++r) {
            free(ctxs[i].v[r].data);
            ctxs[i].v[r].data = NULL;
            ctxs[i].v[r].count = 0;
            ctxs[i].v[r].cap = 0;
        }
    }
}

static void reset_iteration_state(WorkerCtx *ctxs, int nthr)
{
    free_lut();
    free_worker_vectors(ctxs, nthr);

    atomic_store_explicit(&g_phase, 0, memory_order_relaxed);
    atomic_store_explicit(&g_p1_next, 0, memory_order_relaxed);
    atomic_store_explicit(&g_p2_next, 0, memory_order_relaxed);
    atomic_store_explicit(&g_p1_done, 0, memory_order_relaxed);
    atomic_store_explicit(&g_p2_done, 0, memory_order_relaxed);
    atomic_store_explicit(&g_mitm_seeds, 0, memory_order_relaxed);
    atomic_store_explicit(&g_total_evals, 0, memory_order_relaxed);
    atomic_store_explicit(&g_selfcheck_fail, 0, memory_order_relaxed);
    atomic_store_explicit(&g_test_queries, 0, memory_order_relaxed);
    atomic_store_explicit(&g_test_matches, 0, memory_order_relaxed);
    atomic_store_explicit(&g_test_lookup_ns, 0, memory_order_relaxed);
    atomic_store_explicit(&g_test_fortress_ns, 0, memory_order_relaxed);

    for (int i = 0; i < nthr; ++i) {
        ctxs[i].top_n = 0;
        ctxs[i].local_evals = 0;
    }
}

static void evaluate_one_L(const LEntry *e, void *opaque)
{
    if (g_stop_requested)
        return;

    CandidateCallbackCtx *cc = (CandidateCallbackCtx *)opaque;

    // Grid cells are coarse, so finish the exact x/z arc test here.  This is a
    // couple of integer ops on an entry that already passed the LUT, not a
    // per-seed LCG replay.
    if (!entry_arcs_ok(e, cc->Q3, cc->Q4))
        return;

    atomic_fetch_add_explicit(&g_mitm_seeds, 1, memory_order_relaxed);

    WorkerCtx *ctx = cc->ctx;
    uint64_t base_feed = (cc->H << g_b) | (uint64_t)e->L;

    CandidateSummary sum;
    uint64_t fortress_start = g_test_mode ? monotonic_ns() : 0;
    int valid = summarize_candidate(base_feed, &sum);
    if (g_test_mode)
        atomic_fetch_add_explicit(&g_test_fortress_ns,
                monotonic_ns() - fortress_start, memory_order_relaxed);
    if (!valid)
        return;

    ++ctx->local_evals;
    atomic_fetch_add_explicit(&g_total_evals, 1, memory_order_relaxed);

    if (ctx->top_n < TOP_N || summary_should_enter_top(&sum, &ctx->top[ctx->top_n - 1])) {
        SearchResult detailed;
        make_detailed_result(&sum, &detailed);
        top_insert(ctx->top, &ctx->top_n, &detailed);
    }
}

static void *phase2_worker(void *arg)
{
    WorkerCtx *ctx = (WorkerCtx *)arg;
    CandidateCallbackCtx cbctx;
    cbctx.ctx = ctx;

    while (1) {
        if (g_stop_requested)
            break;
        uint64_t s = atomic_fetch_add_explicit(&g_p2_next, H_BATCH, memory_order_relaxed);
        if (s >= g_upper_size)
            break;
        uint64_t e = s + H_BATCH;
        if (e > g_upper_size)
            e = g_upper_size;

        for (uint64_t H = s; H < e; ++H) {
            if (g_stop_requested)
                break;
            uint64_t Q2, Q3, Q4;
            size_t idx = h_bucket(H, &Q3, &Q4, &Q2);
            cbctx.H = H;
            cbctx.Q3 = Q3;
            cbctx.Q4 = Q4;
            lut_query(&g_lut[idx], (uint32_t)Q2, evaluate_one_L, &cbctx);
            atomic_fetch_add_explicit(&g_p2_done, 1, memory_order_relaxed);
        }
    }

    return NULL;
}

// ----- Progress monitor -----
static double elapsed_sec(const struct timespec *a, const struct timespec *b)
{
    return (double)(b->tv_sec - a->tv_sec) +
           (double)(b->tv_nsec - a->tv_nsec) / 1e9;
}

static void *monitor_thread(void *arg)
{
    (void)arg;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    uint64_t last_work = 0;
    double last_t = 0.0;
    int last_phase = -1;
    uint64_t last_evals = 0;
    double last_eval_t = 0.0;
    uint64_t last_mitm_seeds = 0;
    double last_mitm_t = 0.0;

    for (;;) {
        sleep(1);

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double now = elapsed_sec(&t0, &t1);
        uint64_t p1 = atomic_load_u64(&g_p1_done);
        uint64_t p2 = atomic_load_u64(&g_p2_done);
        uint64_t mitm_seeds = atomic_load_u64(&g_mitm_seeds);
        uint64_t evals = atomic_load_u64(&g_total_evals);
        int phase = atomic_load_explicit(&g_phase, memory_order_relaxed);

        uint64_t work = 0;
        uint64_t total = 1;
        const char *label = "init";
        if (phase == 1) {
            work = p1 > g_lower_size ? g_lower_size : p1;
            total = g_lower_size;
            label = "phase1-LUT-build";
        } else if (phase == 2) {
            label = "phase2-LUT-sort";
            work = g_lut_entries;
            total = g_lut_entries ? g_lut_entries : 1;
        } else if (phase == 3) {
            work = p2 > g_upper_size ? g_upper_size : p2;
            total = g_upper_size;
            label = "phase3-H-scan";
        } else if (phase == 4) {
            work = total = 1;
            label = "done";
        }

        double rate = 0.0;
        if (phase != last_phase) {
            last_work = work;
            last_t = now;
            last_phase = phase;
        } else {
            double dt = now - last_t;
            rate = dt > 0.0 && work >= last_work
                ? (double)(work - last_work) / dt
                : 0.0;
            last_work = work;
            last_t = now;
        }

        double eval_dt = now - last_eval_t;
        uint64_t eval_rate_count = evals - last_evals;
        double eval_rate = last_eval_t > 0.0 && eval_dt > 0.0
            ? (double)eval_rate_count / eval_dt : 0.0;
        last_evals = evals;
        last_eval_t = now;
        double mitm_dt = now - last_mitm_t;
        uint64_t mitm_rate_count = mitm_seeds - last_mitm_seeds;
        double mitm_rate = last_mitm_t > 0.0 && mitm_dt > 0.0
            ? (double)mitm_rate_count / mitm_dt : 0.0;
        last_mitm_seeds = mitm_seeds;
        last_mitm_t = now;

        printf("[%7.1fs] %-20s H=%12" PRIu64 "/%-12" PRIu64 " %6.2f%%  H/s=%10.1f  seeds/s=%10.1f  valid4/s=%10.1f  seeds=%" PRIu64 " valid4=%" PRIu64 "\n",
               now, label, work, total,
               total ? 100.0 * (double)work / (double)total : 100.0,
               rate, mitm_rate, eval_rate, mitm_seeds, evals);
        fflush(stdout);

        if (phase == 4)
            break;
    }

    return NULL;
}

// ----- Printing -----
static void print_usage(const char *argv0)
{
    printf("Usage: %s --regions N [--lut-bits N] [--threads N] [--test]\n", argv0);
    printf("       %s [--lut-bits N] [--threads N] [--corner-rx N] [--corner-rz N]\n", argv0);
    printf("          [--pin D X Z]... [--no-pin] [--test]\n");
    printf("       %s N THREADS        (legacy positional form)\n\n", argv0);
    printf("  --regions N         Run N automatic 2x2 region searches.  Corners are\n");
    printf("                     unique; the first round covers every distinct RNG/feed-delta\n");
    printf("                     signature, then later rounds continue with fresh coordinates.\n");
    printf("                     There is no small fixed iteration limit; only the safe int\n");
    printf("                     region-coordinate range limits the total number of corners.\n");
    printf("                     After all N searches, report the global top 10.\n");
    printf("  --lut-bits, -b N   Number of LOW bits placed in the MITM LUT [20..32].\n");
    printf("                     Larger N = larger RAM LUT, smaller H scan.\n");
    printf("  --threads, -t N    Worker thread count [1..256].\n");
    printf("  --corner-rx N      DEBUG: base region X (chunk>>4) for one legacy search.\n");
    printf("  --corner-rz N      DEBUG: base region Z (chunk>>4) for one legacy search.\n");
    printf("                     Only the number of trailing 1-bits of rx / rz matters for\n");
    printf("                     satisfiability. With the default 4 pins it needs rx = 3 mod 8\n");
    printf("                     or 7 mod 16, and rz = 31 mod 32 (negative values are allowed;\n");
    printf("                     rx/rz = -1 is not supported).\n");
    printf("  --pin D X Z        DEBUG: require region D (0=(rx,rz) lo, 1=(rx+1,rz), 2=(rx,rz+1),\n");
    printf("                     3=(rx+1,rz+1) hi) to have fortress offset (X,Z), each 0..7\n");
    printf("                     or -1 for any. Enforced inside the LUT. Default:\n");
    printf("                     --pin 0 0 0 --pin 1 7 0 --pin 2 0 7 --pin 3 7 7.\n");
    printf("  --no-pin           DEBUG: clear all offset pins (legacy behaviour).\n");
    printf("  --test             Measure MITM lookup time versus fortress evaluation time.\n");
    printf("                     Corner (0,0) gives deltas {0,1,16,17}; that corner's\n");
    printf("                     four-way nextInt(3)==0 filter is unsatisfiable, so it\n");
    printf("                     is useful as a sanity-check but not a search target.\n");
    printf("  --help, -h         Show this help.\n");
}

static void print_hex_seed(uint64_t x)
{
    printf("0x%012" PRIx64, (uint64_t)(x & MASK48));
}

static void print_result(const SearchResult *r, int rank)
{
    printf("#%d rngSeeds=0x%012" PRIx64 ",0x%012" PRIx64 ",0x%012" PRIx64 ",0x%012" PRIx64
        " rx=%d rz=%d pieces=%d,%d,%d,%d total=%d\n",
        rank,
        r->region[0].rng_s0,
        r->region[1].rng_s0,
        r->region[2].rng_s0,
        r->region[3].rng_s0,
        r->corner_rx,
        r->corner_rz,
        r->region[0].piece_count,
        r->region[1].piece_count,
        r->region[2].piece_count,
        r->region[3].piece_count,
        r->total_pieces);
}

static int parse_int_arg(const char *s, int *out)
{
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno || end == s || *end != '\0' || v < INT_MIN || v > INT_MAX)
        return 0;
    *out = (int)v;
    return 1;
}

static int parse_u64_arg(const char *s, uint64_t *out)
{
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno || end == s || *end != '\0')
        return 0;
    *out = (uint64_t)v;
    return 1;
}


int main(int argc, char **argv)
{
    int bits = 28;
    int nthr = 16;
    int corner_rx = 0;
    int corner_rz = 1;
    uint64_t regions = 0;
    int legacy_corner_used = 0;
    int legacy_pin_used = 0;

    if (argc == 3 && argv[1][0] != '-') {
        if (!parse_int_arg(argv[1], &bits) || !parse_int_arg(argv[2], &nthr)) {
            print_usage(argv[0]);
            return 1;
        }
    } else {
        for (int i = 1; i < argc; ++i) {
            if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
                print_usage(argv[0]);
                return 0;
            } else if (!strcmp(argv[i], "--regions")) {
                if (++i >= argc || !parse_u64_arg(argv[i], &regions))
                    die("invalid --regions value");
            } else if (!strcmp(argv[i], "-b") || !strcmp(argv[i], "--lut-bits")) {
                if (++i >= argc || !parse_int_arg(argv[i], &bits))
                    die("invalid --lut-bits value");
            } else if (!strcmp(argv[i], "-t") || !strcmp(argv[i], "--threads")) {
                if (++i >= argc || !parse_int_arg(argv[i], &nthr))
                    die("invalid --threads value");
            } else if (!strcmp(argv[i], "--corner-rx")) {
                if (++i >= argc || !parse_int_arg(argv[i], &corner_rx))
                    die("invalid --corner-rx value");
                legacy_corner_used = 1;
            } else if (!strcmp(argv[i], "--corner-rz")) {
                if (++i >= argc || !parse_int_arg(argv[i], &corner_rz))
                    die("invalid --corner-rz value");
                legacy_corner_used = 1;
            } else if (!strcmp(argv[i], "--pin")) {
                int d, px, pz;
                if (i + 3 >= argc ||
                    !parse_int_arg(argv[i + 1], &d) ||
                    !parse_int_arg(argv[i + 2], &px) ||
                    !parse_int_arg(argv[i + 3], &pz) ||
                    d < 0 || d >= REGION_COUNT || px < -1 || px > 7 || pz < -1 || pz > 7)
                    die("--pin needs: D in 0..3, X in -1..7, Z in -1..7");
                g_pin_x[d] = px;
                g_pin_z[d] = pz;
                legacy_pin_used = 1;
                i += 3;
            } else if (!strcmp(argv[i], "--no-pin")) {
                for (int d = 0; d < REGION_COUNT; ++d)
                    g_pin_x[d] = g_pin_z[d] = -1;
                legacy_pin_used = 1;
            } else if (!strcmp(argv[i], "--test")) {
                g_test_mode = 1;
            } else {
                fprintf(stderr, "Unknown argument: %s\n", argv[i]);
                print_usage(argv[0]);
                return 1;
            }
        }
    }

    if (bits < 20 || bits > 32)
        die("--lut-bits must be in [20,32]");
    if (nthr < 1) nthr = 1;
    if (nthr > 256) nthr = 256;

    if (regions > 0 && (legacy_corner_used || legacy_pin_used))
        die("--regions is automatic mode; do not combine it with the legacy corner/pin debug flags");

    if (signal(SIGINT, handle_sigint) == SIG_ERR)
        die("failed to bind SIGINT handler");

    setup(bits);

    uint64_t max_auto_regions = automatic_corner_capacity();
    if (regions > 0 && regions > max_auto_regions)
        die("--regions exceeds the safe automatic corner-coordinate capacity");

    long cpu_count = 16;
    if (cpu_count <= 0)
        cpu_count = 16;
    if (cpu_count > 0 && nthr > cpu_count * 4)
        fprintf(stderr, "Note: using %d threads on %ld online CPUs.\n", nthr, cpu_count);

    printf("MITM split: b=%d LOW bits in LUT, H=%d HIGH bits scanned\n", bits, 48 - bits);
    printf("L side: 2^%d = %" PRIu64 " values\n", bits, g_lower_size);
    printf("H side: 2^%d = %" PRIu64 " values\n", 48 - bits, g_upper_size);
    printf("LUT is built fully in RAM, then sorted in RAM; no on-disk table is used.\n");

    if (regions > 0) {
        printf("Automatic region mode: %" PRIu64 " iterations requested.\n", regions);
        printf("Automatic corners are unique; the first round uses every distinct RNG-delta signature,\n");
        printf("then later rounds continue with fresh absolute corners while preserving valid trailing-one classes.\n");
        printf("Distinct RNG-delta signatures available at b=%d: %" PRIu64 "\n",
               bits, automatic_signature_count());
        printf("Total safe unique automatic corners available: %" PRIu64 "\n", max_auto_regions);
    } else {
        printf("Legacy single-corner mode: use --corner-rx/--corner-rz and pin flags for debugging.\n");
    }
    fflush(stdout);

    atomic_init(&g_phase, 0);
    atomic_init(&g_p1_next, 0);
    atomic_init(&g_p2_next, 0);
    atomic_init(&g_p1_done, 0);
    atomic_init(&g_p2_done, 0);
    atomic_init(&g_mitm_seeds, 0);
    atomic_init(&g_total_evals, 0);
    atomic_init(&g_selfcheck_fail, 0);
    atomic_init(&g_test_queries, 0);
    atomic_init(&g_test_matches, 0);
    atomic_init(&g_test_lookup_ns, 0);
    atomic_init(&g_test_fortress_ns, 0);

    WorkerCtx *ctxs = (WorkerCtx *)xcalloc((size_t)nthr, sizeof(WorkerCtx));
    for (int i = 0; i < nthr; ++i)
        ctxs[i].v = (EntryVec *)xcalloc(g_nb, sizeof(EntryVec));
    pthread_t *threads = (pthread_t *)xmalloc((size_t)nthr * sizeof(pthread_t));

    SearchResult global_top[TOP_N];
    int global_top_n = 0;
    uint64_t aggregate_mitm = 0;
    uint64_t aggregate_evals = 0;
    uint64_t aggregate_selfcheck = 0;
    uint64_t start_all = monotonic_ns();
    uint64_t completed_iterations = 0;

    uint64_t iterations = regions > 0 ? regions : 1;
    for (uint64_t iter = 0; iter < iterations && !g_stop_requested; ++iter) {
        if (regions > 0) {
            if (!automatic_corner((uint64_t)iter, &corner_rx, &corner_rz))
                die("failed to generate automatic corner");
        }

        setup_corner(corner_rx, corner_rz);
        reset_iteration_state(ctxs, nthr);

        if (regions > 0) {
            printf("\n=== Iteration %" PRIu64 "/%" PRIu64 ": corner region=(%d,%d) ===\n",
                   iter + 1, regions, corner_rx, corner_rz);
        } else {
            printf("\n=== Single legacy search: corner region=(%d,%d) ===\n",
                   corner_rx, corner_rz);
        }
        printf("Corner feeds: deltas={0,0x%llx,0x%llx,0x%llx} baseDelta=0x%llx\n",
               (unsigned long long)g_feed_deltas[1],
               (unsigned long long)g_feed_deltas[2],
               (unsigned long long)g_feed_deltas[3],
               (unsigned long long)g_base_feed_delta);
        for (int d = 0; d < REGION_COUNT; ++d) {
            printf("  region %d pin: x=", d);
            if (g_pin_x[d] >= 0) printf("%d", g_pin_x[d]); else printf("any");
            printf(" z=");
            if (g_pin_z[d] >= 0) printf("%d", g_pin_z[d]); else printf("any");
            printf("\n");
        }
        printf("LUT grid: %d x %d cells x %d residues = %zu buckets\n",
               g_G3, g_G4, LUT_BUCKETS, g_nb);
        fflush(stdout);

        atomic_store_explicit(&g_phase, 1, memory_order_relaxed);
        pthread_t mon;
        if (pthread_create(&mon, NULL, monitor_thread, NULL) != 0)
            die("pthread_create(monitor) failed");

        printf("Phase 1: building LUT...\n");
        fflush(stdout);
        for (int i = 0; i < nthr; ++i) {
            if (pthread_create(&threads[i], NULL, phase1_worker, &ctxs[i]) != 0)
                die("pthread_create(phase1) failed");
        }
        for (int i = 0; i < nthr; ++i)
            pthread_join(threads[i], NULL);

        int iteration_has_lut = 1;
        uint64_t raw_counts[LUT_BUCKETS] = {0, 0, 0};
        for (size_t r = 0; r < g_nb; ++r) {
            size_t res = r / ((size_t)g_G3 * (size_t)g_G4);
            for (int i = 0; i < nthr; ++i)
                raw_counts[res] += (uint64_t)ctxs[i].v[r].count;
        }

        uint64_t raw_total = raw_counts[0] + raw_counts[1] + raw_counts[2];
        printf("Phase 1 complete: raw entries req0=%" PRIu64 " req1=%" PRIu64 " req2=%" PRIu64 " total=%" PRIu64 "\n",
               raw_counts[0], raw_counts[1], raw_counts[2], raw_total);
        fflush(stdout);

        g_lut_entries = raw_total;
        if (g_lut_entries == 0) {
            iteration_has_lut = 0;
            printf("ERROR: automatic corner produced an empty LUT; this indicates the default pin assumptions changed.\n");
        }

        SearchResult iteration_top[TOP_N];
        int iteration_top_n = 0;

        if (iteration_has_lut && !g_stop_requested) {
            build_lut(ctxs, nthr);
            if (!g_stop_requested) {
                printf("LUT loaded in RAM: %" PRIu64 " entries (incl. grid replication), %.2f MiB incl. prefix index\n",
                       g_lut_entries, (double)g_lut_bytes / (1024.0 * 1024.0));
                for (int res = 0; res < LUT_BUCKETS; ++res) {
                    uint64_t n = 0;
                    size_t per = (size_t)g_G3 * (size_t)g_G4;
                    for (size_t k = 0; k < per; ++k)
                        n += g_lut[(size_t)res * per + k].count;
                    printf("  residue %d: %" PRIu64 " entries over %zu cells\n", res, n, per);
                }
                fflush(stdout);

                for (size_t r = 0; r < g_nb; ++r) {
                    for (size_t i = 1; i < g_lut[r].count; ++i) {
                        if (cmp_lut_entry(&g_lut[r].data[i-1], &g_lut[r].data[i]) > 0)
                            die("internal error: LUT bucket is not sorted");
                    }
                }

                printf("LUT sort/ordering sanity check: OK\n");
                fflush(stdout);

                atomic_store_explicit(&g_phase, 3, memory_order_relaxed);
                printf("Phase 2: scanning H and evaluating candidates...\n");
                fflush(stdout);
                for (int i = 0; i < nthr; ++i) {
                    if (pthread_create(&threads[i], NULL, phase2_worker, &ctxs[i]) != 0)
                        die("pthread_create(phase2) failed");
                }
                for (int i = 0; i < nthr; ++i)
                    pthread_join(threads[i], NULL);
                atomic_store_explicit(&g_phase, 4, memory_order_relaxed);

                for (int i = 0; i < nthr; ++i) {
                    for (int j = 0; j < ctxs[i].top_n; ++j)
                        top_insert(iteration_top, &iteration_top_n, &ctxs[i].top[j]);
                }
            }
        }

        atomic_store_explicit(&g_phase, 4, memory_order_relaxed);
        pthread_join(mon, NULL);

        uint64_t iter_mitm = atomic_load_u64(&g_mitm_seeds);
        uint64_t iter_evals = atomic_load_u64(&g_total_evals);
        uint64_t iter_selfcheck = atomic_load_u64(&g_selfcheck_fail);
        aggregate_mitm += iter_mitm;
        aggregate_evals += iter_evals;
        aggregate_selfcheck += iter_selfcheck;

        if (iteration_top_n > 0) {
            printf("\nTop %d results for iteration %" PRIu64 " (corner rx=%d rz=%d):\n",
                   iteration_top_n, iter + 1, corner_rx, corner_rz);
            for (int i = 0; i < iteration_top_n; ++i) {
                print_result(&iteration_top[i], i + 1);
                top_insert(global_top, &global_top_n, &iteration_top[i]);
            }

        } else {
            printf("Iteration %" PRIu64 ": no valid fortress candidates.\n", iter + 1);
        }

        if (g_test_mode) {
            uint64_t queries = atomic_load_u64(&g_test_queries);
            uint64_t matches = atomic_load_u64(&g_test_matches);
            uint64_t lookup_ns = atomic_load_u64(&g_test_lookup_ns);
            uint64_t fortress_ns = atomic_load_u64(&g_test_fortress_ns);
            uint64_t mitm_ns = lookup_ns > fortress_ns ? lookup_ns - fortress_ns : 0;
            printf("TEST measurements: queries=%" PRIu64 " matches=%" PRIu64 "\n",
                   queries, matches);
            printf("  MITM + fortress total: %.3f s\n", (double)lookup_ns / 1e9);
            printf("  fortress evaluation:   %.3f s\n", (double)fortress_ns / 1e9);
            printf("  MITM overhead:         %.3f s\n", (double)mitm_ns / 1e9);
        }

        printf("Iteration %" PRIu64 ": MITM matches=%" PRIu64 " fortress-evaluated=%" PRIu64
               " selfcheck-failures=%" PRIu64 "\n",
               iter + 1, iter_mitm, iter_evals, iter_selfcheck);

        free_lut();
        free_worker_vectors(ctxs, nthr);
        ++completed_iterations;
    }

    if (g_stop_requested)
        printf("\nSIGINT received; stopping search and reporting results found so far.\n");

    double total_sec = (double)(monotonic_ns() - start_all) / 1e9;
    if (regions > 0) {
        printf("\nAutomatic region search complete: %" PRIu64 "/%" PRIu64 " iterations finished in %.3f s.\n",
               completed_iterations, regions, total_sec);
        printf("Aggregate MITM matches=%" PRIu64 " fortress-evaluated=%" PRIu64
               " selfcheck-failures=%" PRIu64 "\n",
               aggregate_mitm, aggregate_evals, aggregate_selfcheck);

        if (global_top_n == 0) {
            printf("No valid candidates survived any iteration.\n");
        } else {
            printf("\nTop %d results after %" PRIu64 " iterations:\n", global_top_n, completed_iterations);
            for (int i = 0; i < global_top_n; ++i)
                print_result(&global_top[i], i + 1);
        }
    } else {
        printf("MITM matches (all constraints satisfied): %" PRIu64 "   fortress-evaluated: %" PRIu64 "\n",
               aggregate_mitm, aggregate_evals);
        printf("LUT self-check failures (should be 0): %" PRIu64 "\n", aggregate_selfcheck);

        if (global_top_n == 0) {
            printf("\nTop 0 results:\n");
            printf("No candidates survived the MITM + placement/quadrant filter.\n");
        } else {
            printf("\nTop %d results:\n", global_top_n);
            for (int i = 0; i < global_top_n; ++i)
                print_result(&global_top[i], i + 1);
        }
    }

    free_lut();
    free_worker_vectors(ctxs, nthr);
    for (int i = 0; i < nthr; ++i)
        free(ctxs[i].v);
    free(threads);
    free(ctxs);
    return 0;
}
