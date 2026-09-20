#define _POSIX_C_SOURCE 200809L

// megafortress_revised.c — 1.15 megafortress seed search.
//
// Finds world seeds ws such that the four fortresses in the 2x2 block of
// regions {(0,0),(1,0),(0,1),(1,1)} each lie in the quadrant nearest the
// shared corner (16,16 in chunk coordinates), i.e. they "meet".
//
// MITM: ws = H * 2^b + L.
//   Phase 1 (L-side): for each L and each 4-bit carry pattern cp, compute
//     the required Q mod 3 (must be identical across all four feeds) and the
//     Q-range realizing cp. Entries are kept entirely in RAM.
//   Phase 2 (H-side): for each H compute Q = A^2 * (H ^ MULT_hi) mod 2^(48-b),
//     query the sorted LUT bucket by interval, and fully evaluate candidates.
//
// IMPORTANT SPLIT CONVENTION:
//   b is the number of LOW bits allocated to the LUT side.
//   Larger b => larger RAM LUT and smaller H scan.
//   This intentionally prioritizes the cheap low bits, as requested.
//
// CLI:
//   ./megafortress_revised --lut-bits 28 --threads 16
//   ./megafortress_revised -b 28 -t 16
//   Legacy positional form is also accepted: ./megafortress_revised 28 16
//
// Compile:
//   gcc -O3 -march=native megafortress_revised.c -o megafortress_revised
//       -lpthread -I<path-to-cubiomes>

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
#define LUT_BUCKETS 3
#define MIN_FORTRESS_PIECES 150
#define CHUNK_BATCH 4096ULL
#define H_BATCH 512ULL

static uint64_t g_feed_deltas[REGION_COUNT] = {0, 1, 48, 49};
static int g_corner_rx = 0;
static int g_corner_rz = 1;
static uint64_t g_base_feed_delta = 16;
static int g_test_mode;

// Desired quadrant, relative to the shared corner at chunk coordinate (16,16):
//   d=0 (0,0): x high, z high
//   d=1 (1,0): x low,  z high
//   d=2 (0,1): x high, z low
//   d=3 (1,1): x low,  z low
static const int X_HIGH[REGION_COUNT] = {1, 0, 1, 0};
static const int Z_HIGH[REGION_COUNT] = {1, 1, 0, 0};

static int      g_b;
static uint64_t g_lower_size;
static uint64_t g_upper_size;
static uint64_t g_lower_mask;
static uint64_t g_upper_mask;
static int      g_sc;
static int      g_sc_inv;
static int      g_pc;
static uint64_t g_Ak[4], g_Bk[4];

typedef struct {
    uint32_t L;
    uint32_t rlo;
    uint32_t rhi;
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

static LutBucket g_lut[LUT_BUCKETS];
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
static atomic_uint_fast64_t g_test_queries;
static atomic_uint_fast64_t g_test_matches;
static atomic_uint_fast64_t g_test_lookup_ns;
static atomic_uint_fast64_t g_test_fortress_ns;
static volatile sig_atomic_t g_stop_requested;

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
    EntryVec v[LUT_BUCKETS];
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
        size_t nc = v->cap ? v->cap * 2 : 65536;
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
    if (rx < 0 || rz < 0 || rx == INT_MAX || rz > (INT_MAX - 1) / 16)
        die("corner region coordinates must be non-negative and safely fit in signed int math");

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

    if (g_lower_size == 0 || g_upper_size == 0)
        die("internal split size became zero");

    g_Ak[0] = 1;
    g_Bk[0] = 0;
    for (int i = 1; i <= 3; ++i) {
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
}

// L-side test for one (L, cp).
// Returns 1 if valid and fills req/rlo/rhi.
static int eval_L(uint64_t L, int cp, int *req, uint32_t *rlo, uint32_t *rhi)
{
    uint64_t P2[REGION_COUNT];

    for (int d = 0; d < REGION_COUNT; ++d) {
        uint64_t Pl = L ^ (uint64_t)g_feed_deltas[d] ^ (MULT & g_lower_mask);
        P2[d] = (g_Ak[2] * Pl + g_Bk[2]) & MASK48;
    }

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

    for (int d = 1; d < REGION_COUNT; ++d) {
        if (req_d[d] != req_d[0])
            return 0;
    }

    // Carry c_d = 1 iff Q >= T_d, where T_d = upper_size - P2_hi.
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
        return 0;

    *req = req_d[0];
    *rlo = (uint32_t)lo_q;
    *rhi = (uint32_t)hi_q;
    return 1;
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
            for (int cp = 0; cp < 16; ++cp) {
                int req;
                uint32_t rlo, rhi;
                if (eval_L(L, cp, &req, &rlo, &rhi)) {
                    LEntry en;
                    en.L = (uint32_t)L;
                    en.rlo = rlo;
                    en.rhi = rhi;
                    vec_push(&ctx->v[req], en);
                }
            }
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
    for (int r = 0; r < LUT_BUCKETS; ++r) {
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
    }

    for (int r = 0; r < LUT_BUCKETS; ++r) {
        g_lut_bytes += (uint64_t)g_lut[r].count * sizeof(LEntry);
        g_lut_bytes += (uint64_t)g_lut[r].count * sizeof(uint32_t);
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

typedef void (*lut_match_cb)(uint32_t L, void *opaque);

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
            cb(b->data[i].L, opaque);
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

// ----- Fortress evaluation -----
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
        if (((s >> 17) % 3) != 0)
            return 0;

        s = lcg(s);                         // nextInt(8) for x
        int xv = (int)((s >> 45) & 7);
        s = lcg(s);                         // nextInt(8) for z
        int zv = (int)((s >> 45) & 7);

        if (X_HIGH[d] ? (xv < 4) : (xv >= 4))
            return 0;
        if (Z_HIGH[d] ? (zv < 4) : (zv >= 4))
            return 0;

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
// We do not use the generic callback above; phase2 keeps H in a local context.
typedef struct {
    WorkerCtx *ctx;
    uint64_t H;
} CandidateCallbackCtx;

static void evaluate_one_L(uint32_t L, void *opaque)
{
    if (g_stop_requested)
        return;

    atomic_fetch_add_explicit(&g_mitm_seeds, 1, memory_order_relaxed);

    CandidateCallbackCtx *cc = (CandidateCallbackCtx *)opaque;
    WorkerCtx *ctx = cc->ctx;
    uint64_t base_feed = (cc->H << g_b) | (uint64_t)L;

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
            uint64_t Q2 = (g_Ak[2] * (H ^ (MULT >> g_b))) & g_upper_mask;
            int r = (int)(Q2 % 3);
            const LutBucket *bucket = &g_lut[r];
            cbctx.H = H;
            lut_query(bucket, (uint32_t)Q2, evaluate_one_L, &cbctx);
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

        static uint64_t last_evals = 0;
        static double last_eval_t = 0.0;
         static uint64_t last_mitm_seeds = 0;
         static double last_mitm_t = 0.0;
        double eval_dt = now - last_eval_t;
        uint64_t eval_rate_count = evals - last_evals;
        double eval_rate = eval_dt > 0.0 ? (double)eval_rate_count / eval_dt : 0.0;
        last_evals = evals;
        last_eval_t = now;
         double mitm_dt = now - last_mitm_t;
         uint64_t mitm_rate_count = mitm_seeds - last_mitm_seeds;
         double mitm_rate = mitm_dt > 0.0 ? (double)mitm_rate_count / mitm_dt : 0.0;
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
    printf("Usage: %s [--lut-bits N] [--threads N] [--corner-rx N] [--corner-rz N] [--test]\n", argv0);
    printf("       %s N THREADS        (legacy positional form)\n\n", argv0);
    printf("  --lut-bits, -b N   Number of LOW bits placed in the MITM LUT [20..32].\n");
    printf("                     Larger N = larger RAM LUT, smaller H scan.\n");
    printf("  --threads, -t N    Worker thread count [1..256].\n");
    printf("  --corner-rx N      Base region X for the 2x2 corner. Default 0.\n");
    printf("  --corner-rz N      Base region Z for the 2x2 corner. Default 1.\n");
    printf("  --test              Measure MITM lookup time versus fortress evaluation time.\n");
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
    printf("#%d rngSeeds=%012" PRIx64 ",%012" PRIx64 ",%012" PRIx64 ",%012" PRIx64
        " pieces=%d,%d,%d,%d total=%d\n",
        rank,
        r->region[0].rng_s0,
        r->region[1].rng_s0,
        r->region[2].rng_s0,
        r->region[3].rng_s0,
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

int main(int argc, char **argv)
{
    int bits = 28;
    int nthr = 16;
    int corner_rx = 0;
    int corner_rz = 1;

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
            } else if (!strcmp(argv[i], "-b") || !strcmp(argv[i], "--lut-bits")) {
                if (++i >= argc || !parse_int_arg(argv[i], &bits))
                    die("invalid --lut-bits value");
            } else if (!strcmp(argv[i], "-t") || !strcmp(argv[i], "--threads")) {
                if (++i >= argc || !parse_int_arg(argv[i], &nthr))
                    die("invalid --threads value");
            } else if (!strcmp(argv[i], "--corner-rx")) {
                if (++i >= argc || !parse_int_arg(argv[i], &corner_rx))
                    die("invalid --corner-rx value");
            } else if (!strcmp(argv[i], "--corner-rz")) {
                if (++i >= argc || !parse_int_arg(argv[i], &corner_rz))
                    die("invalid --corner-rz value");
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

    if (signal(SIGINT, handle_sigint) == SIG_ERR)
        die("failed to bind SIGINT handler");

    setup_corner(corner_rx, corner_rz);
    setup(bits);

    printf("Corner region=(%d,%d)  feeds deltas={0,0x%llx,0x%llx,0x%llx}  baseDelta=0x%llx\n",
           g_corner_rx, g_corner_rz,
           (unsigned long long)g_feed_deltas[1],
           (unsigned long long)g_feed_deltas[2],
           (unsigned long long)g_feed_deltas[3],
           (unsigned long long)g_base_feed_delta);

    long cpu_count = 16;
    if (cpu_count > 0 && nthr > cpu_count * 4)
        fprintf(stderr, "Note: using %d threads on %ld online CPUs.\n", nthr, cpu_count);

    printf("MITM split: b=%d LOW bits in LUT, H=%d HIGH bits scanned\n", bits, 48 - bits);
    printf("L side: 2^%d = %" PRIu64 " values\n", bits, g_lower_size);
    printf("H side: 2^%d = %" PRIu64 " values\n", 48 - bits, g_upper_size);
    printf("LUT is built fully in RAM, then sorted in RAM; no on-disk table is used.\n");
    fflush(stdout);

    atomic_init(&g_phase, 0);
    atomic_init(&g_p1_next, 0);
    atomic_init(&g_p2_next, 0);
    atomic_init(&g_p1_done, 0);
    atomic_init(&g_p2_done, 0);
    atomic_init(&g_mitm_seeds, 0);
    atomic_init(&g_total_evals, 0);
    atomic_init(&g_test_queries, 0);
    atomic_init(&g_test_matches, 0);
    atomic_init(&g_test_lookup_ns, 0);
    atomic_init(&g_test_fortress_ns, 0);

    WorkerCtx *ctxs = (WorkerCtx *)xcalloc((size_t)nthr, sizeof(WorkerCtx));
    pthread_t *threads = (pthread_t *)xmalloc((size_t)nthr * sizeof(pthread_t));

    pthread_t mon;
    if (pthread_create(&mon, NULL, monitor_thread, NULL) != 0)
        die("pthread_create(monitor) failed");

    atomic_store_explicit(&g_phase, 1, memory_order_relaxed);
    printf("Phase 1: building LUT...\n");
    fflush(stdout);
    for (int i = 0; i < nthr; ++i) {
        if (pthread_create(&threads[i], NULL, phase1_worker, &ctxs[i]) != 0)
            die("pthread_create(phase1) failed");
    }
    for (int i = 0; i < nthr; ++i)
        pthread_join(threads[i], NULL);

    if (g_stop_requested) {
        atomic_store_explicit(&g_phase, 4, memory_order_relaxed);
        pthread_join(mon, NULL);
        goto output_results;
    }

    uint64_t raw_counts[LUT_BUCKETS] = {0, 0, 0};
    for (int r = 0; r < LUT_BUCKETS; ++r) {
        for (int i = 0; i < nthr; ++i)
            raw_counts[r] += (uint64_t)ctxs[i].v[r].count;
    }

    printf("Phase 1 complete: raw entries req0=%" PRIu64 " req1=%" PRIu64 " req2=%" PRIu64 " total=%" PRIu64 "\n",
           raw_counts[0], raw_counts[1], raw_counts[2],
           raw_counts[0] + raw_counts[1] + raw_counts[2]);
    fflush(stdout);

    g_lut_entries = raw_counts[0] + raw_counts[1] + raw_counts[2];
    if (g_lut_entries == 0) {
        printf("LUT is empty: the four placement constraints are unsatisfiable for this corner/delta class.\n");
        printf("No H seeds will be scanned. Choose another --corner-rx/--corner-rz.\n");
        atomic_store_explicit(&g_phase, 4, memory_order_relaxed);
        pthread_join(mon, NULL);
        for (int i = 0; i < nthr; ++i) {
            for (int r = 0; r < LUT_BUCKETS; ++r)
                free(ctxs[i].v[r].data);
        }
        free(threads);
        free(ctxs);
        return 0;
    }

    build_lut(ctxs, nthr);
    if (g_stop_requested) {
        atomic_store_explicit(&g_phase, 4, memory_order_relaxed);
        pthread_join(mon, NULL);
        goto output_results;
    }
    printf("LUT loaded in RAM: %" PRIu64 " entries, %.2f MiB incl. prefix index\n",
           g_lut_entries, (double)g_lut_bytes / (1024.0 * 1024.0));
    for (int r = 0; r < LUT_BUCKETS; ++r)
        printf("  bucket %d: %zu entries\n", r, g_lut[r].count);
    fflush(stdout);

    // Sanity check sorting; catches accidental unsorted-table bugs immediately.
    for (int r = 0; r < LUT_BUCKETS; ++r) {
        for (size_t i = 1; i < g_lut[r].count; ++i) {
            if (cmp_lut_entry(&g_lut[r].data[i-1], &g_lut[r].data[i]) > 0)
                die("internal error: LUT bucket is not sorted");
        }
    }
    if (g_stop_requested) {
        atomic_store_explicit(&g_phase, 4, memory_order_relaxed);
        pthread_join(mon, NULL);
        goto output_results;
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
    pthread_join(mon, NULL);

output_results:
    if (g_stop_requested)
        printf("\nSIGINT received; stopping search and reporting results found so far.\n");

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

    // Merge thread-local top 10 lists into one global top 10.
    SearchResult global_top[TOP_N];
    int global_top_n = 0;
    for (int i = 0; i < nthr; ++i) {
        for (int j = 0; j < ctxs[i].top_n; ++j)
            top_insert(global_top, &global_top_n, &ctxs[i].top[j]);
    }

    printf("\nTop %d results:\n", global_top_n);

    if (global_top_n == 0) {
        printf("No candidates survived the MITM + placement/quadrant filter.\n");
    } else {
        for (int i = 0; i < global_top_n; ++i)
            print_result(&global_top[i], i + 1);
    }

    for (int r = 0; r < LUT_BUCKETS; ++r) {
        free(g_lut[r].data);
        free(g_lut[r].prefix_max_rhi);
    }
    for (int i = 0; i < nthr; ++i) {
        for (int r = 0; r < LUT_BUCKETS; ++r)
            free(ctxs[i].v[r].data);
    }
    free(threads);
    free(ctxs);
    return 0;
}
