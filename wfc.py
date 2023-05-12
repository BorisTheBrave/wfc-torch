import torch as t
from torch import sparse
from torchvision.io import read_image, write_png
from collections import namedtuple
import numpy
from dataclasses import dataclass
import torch as t

LOG_LEVEL = 0

Dir = namedtuple('Dir', 'name filter move')
dirs = [
    Dir('right', lambda i,h,w: (i%w) != w -1, lambda i,h,w: i + 1),
    Dir('left', lambda i,h,w: (i%w) != 0, lambda i,h,w: i - 1),
    Dir('up', lambda i,h,w: i < w * (h-1), lambda i,h,w: i + w),
    Dir('down', lambda i,h,w: i >= w, lambda i,h,w: i - w),
]

def _cheap_ones(size):
    return t.tensor(1.0).as_strided(size, (0,))


@dataclass
class Model:
    pattern_count: int
    frequencies: t.tensor # shape=(p)
    propagators: list[t.Tensor] # shape=(d,from,to). Should be sparse, and of tpe float, with 0 and 1 entries

def make_adacent_model(pattern_count, pi: t.Tensor):
    frequencies = t.zeros(pattern_count)
    frequencies.index_add_(0, pi.flatten(), _cheap_ones((pi.numel(),)))

    h, w = pi.shape
    pi = pi.flatten()
    propagators = []
    for dir in dirs:
        indices = t.arange(0, len(pi), step=1, dtype=t.long,)
        a = indices[dir.filter(indices, h, w)]
        b = dir.move(a, h, w)
        a = pi[a]
        b = pi[b]
        adj = t.stack([a, b])
        prop = t.zeros((pattern_count, pattern_count))
        prop[list(adj)] = 1
        prop = prop.to_sparse_csr()
        propagators.append(prop)

    return Model(pattern_count, frequencies, propagators)


@dataclass
class WFCConfig:
    h: int
    w: int
    model: Model

def run(config: WFCConfig):
    h = config.h
    w = config.w
    model = config.model
    pattern_count = model.pattern_count
    frequenies = model.frequencies
    propagators = model.propagators
    possibilities = t.ones((w * h, pattern_count,))


    def print_possibilities():
        p2 = possibilities.reshape((h, w, pattern_count))
        for y in range(h):
            for x in range(w):
                if not t.any(p2[y, x]).item():
                    for p in range(pattern_count):
                        print("x" * len(str(p)), end="")
                else:
                    for p in range(pattern_count):
                        if p2[y, x, p]:
                            print(p, end="")
                        else:
                            print(" " * len(str(p)),end="")
                print("|", end="")
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
            if LOG_LEVEL >= 6: print(f"{changed_cells=}")
            new_changed_cells = []
            for dir, prop in zip(dirs, propagators):
                # cells that have a cell to the right
                rightable_cells = changed_cells[dir.filter(changed_cells, h, w)] # b
                if LOG_LEVEL >= 10: print(f"{dir.name} {rightable_cells=}")

                # Corresponding cell to the right
                right_cells = dir.move(rightable_cells, h, w)
                if LOG_LEVEL >= 10: print(f"{dir.name} {right_cells=}")

                # For each index, retrieve possible patterns
                rightable_possibles = possibilities[rightable_cells] # b p
                if LOG_LEVEL >= 10: print(f"{dir.name} {rightable_possibles=}")

                # Check which patterns are supported
                right_support = t.sign(rightable_possibles @ prop) # b p
                if LOG_LEVEL >= 10: print(f"{dir.name} {right_support=}")

                # Find changed cells
                right_changed = right_cells[t.any((right_support == 0) & (possibilities[right_cells] == 1), dim=1)]
                if LOG_LEVEL >= 10: print(f"{dir.name} {right_changed=}")
                new_changed_cells.append(right_changed)

                # Restrict possibilites to support
                possibilities[right_cells, :] *= right_support

            changed_cells = t.unique(t.concat(new_changed_cells))
            
            if LOG_LEVEL >= 5: print_possibilities()
        
        if LOG_LEVEL >= 5: print_possibilities()

    decided = get_decided(possibilities).reshape(h, w)

    return decided


    