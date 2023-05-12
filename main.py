import torch as t
from torch import sparse
from torchvision.io import read_image, write_png
from collections import namedtuple
import numpy
from wfc import run, make_adacent_model, WFCConfig, Model

LOG_LEVEL = 0



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

config = WFCConfig(
    h = 10,
    w = 10,
    model = make_adacent_model(palette, pi)
)

result = run(config)
d = unpalettize(palette, result)
d = from_i32(d)
write_png(d, "output.png")