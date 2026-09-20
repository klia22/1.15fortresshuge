#ifndef FORTRESS_H_
#define FORTRESS_H_

#include <stdint.h>

#ifndef STRUCT
#define STRUCT(S) typedef struct S S; struct S
#endif

enum FortressMCVersion
{
    MC_1_7_10,
    MC_1_15,
};

static inline void setSeed(uint64_t *seed, uint64_t value)
{
    *seed = (value ^ 0x5deece66d) & ((1ULL << 48) - 1);
}

static inline int next(uint64_t *seed, int bits)
{
    *seed = (*seed * 0x5deece66d + 0xb) & ((1ULL << 48) - 1);
    return (int)((int64_t)*seed >> (48 - bits));
}

static inline int nextInt(uint64_t *seed, int n)
{
    int bits, value;
    const int mask = n - 1;
    if ((mask & n) == 0)
        return (int)(((uint64_t)n * next(seed, 31)) >> 31);
    do {
        bits = next(seed, 31);
        value = bits % n;
    } while ((int32_t)((uint32_t)bits - value + mask) < 0);
    return value;
}

static inline uint64_t nextLong(uint64_t *seed)
{
    return ((uint64_t)next(seed, 32) << 32) + next(seed, 32);
}

static inline void skipNextN(uint64_t *seed, uint64_t n)
{
    while (n--)
        next(seed, 32);
}

#define IABS(X) ((X) < 0 ? -(X) : (X))
#define UNREACHABLE() __builtin_unreachable()

#ifndef MASK48
#define MASK48 (((int64_t)1 << 48) - 1)
#endif

STRUCT(Pos) { int x, z; };
STRUCT(Pos3) { int x, y, z; };

STRUCT(Piece)
{
    const char *name;
    Pos3 pos, bb0, bb1;
    uint8_t rot;
    int8_t depth;
    int8_t type;
    int chestCount;
    Pos chestPoses[4];
    uint64_t lootSeeds[4];
    const char *lootTables[4];
    int additionalData;
    Piece *next;
};

#ifdef __cplusplus
extern "C" {
#endif

enum
{   // Fortress piece types
    FORTRESS_START,
    BRIDGE_STRAIGHT,
    BRIDGE_CROSSING,
    BRIDGE_FORTIFIED_CROSSING,
    BRIDGE_STAIRS,
    BRIDGE_SPAWNER,
    BRIDGE_CORRIDOR_ENTRANCE,
    CORRIDOR_STRAIGHT,
    CORRIDOR_CROSSING,
    CORRIDOR_TURN_RIGHT,
    CORRIDOR_TURN_LEFT,
    CORRIDOR_STAIRS,
    CORRIDOR_T_CROSSING,
    CORRIDOR_NETHER_WART,
    FORTRESS_END,
    FORTRESS_PIECE_COUNT,
};

STRUCT(FortressPieceEnv)
{
    Piece *list;
    Piece *tail;
    int *n;
    uint64_t *rng;
    int typlast;
    int nmax;
    int ntyp[FORTRESS_PIECE_COUNT];
};

STRUCT(FortressPieceInfo)
{
    Pos3 offset, size;
    int repeatable, weight, max;
    const char *name;
};

extern const FortressPieceInfo fortress_info[FORTRESS_PIECE_COUNT];

Piece *addFortressPiece(FortressPieceEnv *env, int typ, int x, int y, int z,
        int depth, int facing, int pending);
void extendFortress(FortressPieceEnv *env, Piece *p, int offh, int offv,
        int turn, int corridor);
void extendFortressPiece(FortressPieceEnv *env, Piece *p);

/* Generate the structure pieces of a Nether Fortress. The maximum number of
 * pieces that are generated is limited to 'n'. A buffer length of around 400
 * should be sufficient in practice, but a fortress can in theory contain many
 * more than that. The number of generated pieces is given by the return value.
 */
int getFortressPieces(Piece *list, int n, int mc, uint64_t seed, int chunkX, int chunkZ);

#ifdef __cplusplus
}
#endif

#endif //FORTRESS_H_
