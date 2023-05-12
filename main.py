import torch as t
from torch import sparse
from torchvision.io import read_image, write_png
from collections import namedtuple
import numpy

LOG_LEVEL = 0

def cheap_ones(size):
    return t.tensor(1.0).as_strided(size, (0,))


i = read_image("Angular.png")

# Palettize

def to_i32(i):
    i = i.to(t.int32)
    return (i[0] << 16) + (i[1] << 8) + (i[2] << 0)

def from_i32(i32):
    return t.stack([(i32 >> 16) & 0xFF, (i32 >> 8) & 0xFF, (i32 >> 0) & 0xFF], dim=0).to(t.uint8)

def make_palette(i32):
    return t.sort(t.unique(i32.flatten()))[0]

def palettize(palette, i32):
    return t.searchsorted(palette, i32)

def unpalettize(palette, pi):
    return palette[pi]



i32 = to_i32(i)

palette = make_palette(i32)
pattern_count = len(palette)

pi = palettize(palette, i32)

# Prepare model

# shape = p
frequencies = t.zeros(pattern_count)
frequencies.index_add_(0, pi.flatten(), cheap_ones((pi.numel(),)))

print(f"{pi=}")
print(f"{frequencies=}")

# shape = x y p


right_adjacencies = t.stack([pi[:-1,:].flatten(), pi[1:,:].flatten()])

# shape p p'
right_propagator = t.zeros((pattern_count, pattern_count))
right_propagator[list(right_adjacencies)] = 1
right_propagator = right_propagator.to_sparse_csr()
# TODO

# Prepare output topology
w = 10
h = 4
Dir = namedtuple('Dir', 'name filter move')
dirs = [
    Dir('right', lambda i: (i%w) != w -1, lambda i: i + 1),
    Dir('left', lambda i: (i%w) != 0, lambda i: i - 1),
    Dir('up', lambda i: i < w * (h-1), lambda i: i + w),
    Dir('down', lambda i: i >= w, lambda i: i - w),
]

# We use a single index for cells
# i = x + w * y
possibilities = t.ones((w * h, pattern_count,))

def print_possibilities():
    p2 = possibilities.reshape((h, w, pattern_count))
    for y in range(h):
        for x in range(w):
            for p in range(pattern_count):
                if p2[y, x, p]:
                    print(p, end="")
                else:
                    print(" ",end="")
            print(" ", end="")
        print()

def get_decided(possibilities):
    return possibilities.argmax(dim=-1)

def print_decided(decided):
    p2 = possibilities.reshape((h, w, pattern_count))
    p2 = p2.argmax(dim=2)
    for y in range(h):
        for x in range(w):
            print(p2[y,x].item(), end="")
        print()


# main loop
while True:
    # decide batch size, n
    # indices = choose n indices, possibly unnear each other
    # for indices: pick a random tile from possibiliites
    # for indices: update possibilites.
    # set changedTiles = list of changed tiles

    # Pick a random index to update
    undecided = (possibilities.sum(1) > 1).nonzero().flatten()
    if len(undecided) == 0:
        break

    i = numpy.random.choice(undecided, (1,)).item()
    # Pick a random possibility
    # TODO: Use frequencies
    possibles = (possibilities[i] > 0).nonzero().flatten()
    p = numpy.random.choice(possibles).item()
    if LOG_LEVEL >= 5: print(f"{i=}")
    if LOG_LEVEL >= 5: print(f"{p=}")

    # Select that specific possibility
    possibilities[i, :] = 0
    possibilities[i, p] = 1
    changed_cells = t.tensor([i], dtype=int)

    while len(changed_cells) > 0:
        print(f"{changed_cells=}")
        new_changed_cells = []
        for dir in dirs:
            # cells that have a cell to the right
            rightable_cells = changed_cells[dir.filter(changed_cells)] # b
            if LOG_LEVEL >= 10: print(f"{dir.name} {rightable_cells=}")

            # Corresponding cell to the right
            right_cells = dir.move(rightable_cells)
            if LOG_LEVEL >= 10: print(f"{dir.name} {right_cells=}")

            # For each index, retrieve possible patterns
            rightable_possibles = possibilities[rightable_cells] # b p
            if LOG_LEVEL >= 10: print(f"{dir.name} {rightable_possibles=}")

            # Check which patterns are supported
            right_support = t.sign(rightable_possibles @ right_propagator) # b p
            if LOG_LEVEL >= 10: print(f"{dir.name} {right_support=}")

            # Find changed cells
            right_changed = right_cells[t.any((right_support == 0) & (possibilities[right_cells] == 1), dim=1)]
            if LOG_LEVEL >= 10: print(f"{dir.name} {right_changed=}")
            new_changed_cells.append(right_changed)

            # Restrict possibilites to support
            possibilities[right_cells, :] *= right_support

        changed_cells = t.concat(new_changed_cells)
        continue


        # # cells that have a cell to the right
        # rightable_cells = changed_cells[(changed_cells % w) != w - 1] # b
        # print(f"{rightable_cells=}")

        # # Corresponding cell to the right
        # right_cells = rightable_cells + 1 # b
        # print(f"{rightable_cells=}")

        # # For each index, retrieve possible patterns
        # rightable_possibles = possibilities[rightable_cells] # b p
        # print(f"{rightable_possibles=}")

        # # Check which patterns are supported
        # right_support = t.sign(rightable_possibles @ right_propagator) # b p
        # print(f"{right_support=}")

        # # Find changed cells
        # right_changed = right_cells[t.any((right_support == 0) & (possibilities[right_cells] == 1), dim=1)]
        # print(f"{right_changed=}")

        # # Restrict possibilites to support
        # possibilities[right_cells, :] *= right_support

        break
        right_of_changed = changed_cells + 1
        right_of_changed = right_of_changed[(right_of_changed % w) != 0]
        patterns_right_of_changed = t.cartesian_prod([right_of_changed, t.range(0, pattern_count)])
        # for recheckList, for pattern: find patterns with no support
        # update undecided
        # ban patterns, set changedTiles
        break
    
    if LOG_LEVEL >= 5: print_possibilities()

    #break
d = get_decided(possibilities)
print_decided(d)

d = d.reshape(h, w)
d = unpalettize(palette, d)
d = from_i32(d)
write_png(d, "output.png")