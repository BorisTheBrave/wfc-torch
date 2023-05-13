import torch as t
from torch import sparse
from torchvision.io import read_image, write_png
from collections import namedtuple
import numpy
from dataclasses import dataclass
import torch as t
import datetime
import tqdm
from typing import Optional

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

def make_adacent_model(pattern_count, pis: list[t.Tensor]):
    frequencies = t.zeros(pattern_count)
    for pi in pis:
        frequencies.index_add_(0, pi.flatten(), _cheap_ones((pi.numel(),)))

    props = [t.zeros((pattern_count, pattern_count)) for d in dirs]

    for pi in pis: 
        h, w = pi.shape
        pi = pi.flatten()
        for i, dir in enumerate(dirs):
            indices = t.arange(0, len(pi), step=1, dtype=t.long,)
            a = indices[dir.filter(indices, h, w)]
            b = dir.move(a, h, w)
            a = pi[a]
            b = pi[b]
            adj = t.stack([a, b])
            props[i][list(adj)] = 1

    for i in range(len(props)):
        props[i] = props[i].T.to_sparse_csr()

    return Model(pattern_count, frequencies, props)


@dataclass
class WFCConfig:
    h: int
    w: int
    model: Model
    parallelism: int = 4
    device: str = "cpu"
    seed: Optional[int] = None

def _propagate_once(changed_cells, possibilities, propagators, h, w):
    new_changed_cells = []
    #for dir, prop in zip(dirs, propagators):
    for i in range(len(dirs)):
        dir = dirs[i]
        prop = propagators[i]
        # cells that have a cell to the right
        rightable_cells = changed_cells[dir.filter(changed_cells, h, w)] # b

        # Corresponding cell to the right
        right_cells = dir.move(rightable_cells, h, w)

        # For each index, retrieve possible patterns
        rightable_possibles = possibilities[rightable_cells] # b p

        # Check which patterns are supported
        right_support = t.sign(prop @ rightable_possibles.T).T # b p

        # Find changed cells
        right_changed = right_cells[t.any((right_support == 0) & (possibilities[right_cells] == 1), dim=1)]
        new_changed_cells.append(right_changed)

        # Restrict possibilites to support
        possibilities[right_cells, :] *= right_support
    
    return t.unique(t.concat(new_changed_cells))

@t.inference_mode()
def run(config: WFCConfig):

    if config.seed is not None:
        numpy.random.seed(config.seed)

    start = datetime.datetime.now()
    h = config.h
    w = config.w
    parallelism = config.parallelism
    model = config.model
    device = config.device
    pattern_count = model.pattern_count
    frequencies = model.frequencies
    propagators = [p.to(device) for p in model.propagators]
    possibilities = t.ones((w * h, pattern_count,), device=device)

    progress = tqdm.tqdm(total=possibilities.shape[0])



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

    def propagate(changed_cells):
        while len(changed_cells) > 0:
            if LOG_LEVEL >= 6: print(f"{changed_cells=}")
            #changed_cells = propagate_once_traced(changed_cells)
            changed_cells = _propagate_once(changed_cells, possibilities, propagators, h, w)


    # Remove any patterns that are already impossible
    for dir,prop in zip(dirs, propagators):
        indices = t.arange(0, possibilities.shape[0])
        mask = dir.filter(indices, h, w)
        # Does the pattern have any possible support
        possible_patterns = prop.to_dense().amax(dim=0)
        possibilities[mask] *= possible_patterns
        propagate(indices[mask].to(device))


    # Main loop
    while True:
        # Pick a random index to update
        contradiction = (possibilities.sum(1) == 0).nonzero().flatten()
        if len(contradiction) > 0:
            print("contradiction at ", contradiction.nonzero()[:1])
            break

        undecided = (possibilities.sum(1) > 1).nonzero().flatten().cpu()
        if len(undecided) == 0:
            break

        progress.update(possibilities.shape[0] - len(undecided) - progress.n)

        i = numpy.random.choice(undecided.cpu().numpy(), (parallelism,))
        # Pick a random possibility
        # TODO: Use frequencies
        possibles = (possibilities[i] > 0)
        weights = possibles * frequencies
        cweights = weights.cumsum(dim=1)
        r = t.rand((weights.shape[0])) * cweights[:, -1]
        p = t.searchsorted(cweights, r.unsqueeze(1)).squeeze(1)
        if LOG_LEVEL >= 5: print(f"{i=}")
        if LOG_LEVEL >= 5: print(f"{p=}")

        # Select that specific possibility
        possibilities[i, :] = 0
        possibilities[i, p] = 1
        changed_cells = t.unique(t.tensor(i))

        propagate(changed_cells)
        
        if LOG_LEVEL >= 5: print_possibilities()

    decided = get_decided(possibilities).reshape(h, w)

    end = datetime.datetime.now()
    progress.close()
    print(f"Took {end-start}")

    return decided.cpu()


    
