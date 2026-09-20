MULT = 0x5DEECE66D
ADD = 0xB
INV = 0xDFE05BCB1365
MASK48 = (1 << 48) - 1






s = [0xe875d99ea1a5, 0xe70f2d7c7cfa, 0x1eb5066a0175, 0x1d4e5a47dcca] #<=============================EDIT THESE
CORNER = (3, 31)                                                     #<=============================EDIT THESE






REGIONS = None
regions = REGIONS or [(CORNER[0], CORNER[1]), (CORNER[0] + 1, CORNER[1]),
                      (CORNER[0], CORNER[1] + 1), (CORNER[0] + 1, CORNER[1] + 1)]
def lcg(seed):
    return (seed * MULT + ADD) & MASK48
def invlcg(seed):
    return ((seed - ADD) * INV) & MASK48
def region_delta(rx, rz):
    return (rx ^ (rz << 4)) & MASK48
class LCG:
    def __init__(self, seed):
        self.seed = seed & MASK48
    def next(self, bits):
        self.seed = lcg(self.seed)
        return self.seed >> (48 - bits)
    def nextInt(self, bound):
        if bound & (bound - 1) == 0:
            return (bound * self.next(31)) >> 31
        while True:
            bits = self.next(31)
            val = bits % bound
            if bits - val + (bound - 1) < (1 << 31):
                return val
world_seeds = []
for rng, (rx, rz) in zip(s, regions):
    state = rng
    for _ in range(4):                    
        state = invlcg(state)
    feed = state ^ MULT
    world_seeds.append(feed ^ region_delta(rx, rz))
if len(set(world_seeds)) != 1:
    print("ERROR: the rng seeds do not belong to one shared world seed "
          "(wrong CORNER/REGIONS?):")
    print("  " + ", ".join(f"0x{w:012x}" for w in world_seeds))
    raise SystemExit(1)
world = world_seeds[0]
coords = []
for rng, (rx, rz) in zip(s, regions):
    r = LCG((world ^ region_delta(rx, rz)) ^ MULT)
    r.next(31)                 
    r.nextInt(3)
    x = r.nextInt(8)
    z = r.nextInt(8)
    coords.append((rx * 16 + x + 4, rz * 16 + z + 4))
print(f"world seed: {world}  (0x{world:012x})")
for (cx, cz) in coords:
    print(f"chunk ({cx}, {cz})   block ({cx * 16}, {cz * 16})")
