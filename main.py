import torch as t
from torch import sparse
from torchvision.io import read_image, write_png
from collections import namedtuple
import numpy
from wfc import run, make_adacent_model, WFCConfig, Model
from preprocess import adj_preprocess, overlap_preprocess, make_rotations

LOG_LEVEL = 0



i = read_image("Angular.png")

r = make_rotations(i)

#pis, palette, pattern_count, reverse_fn = adj_preprocess(r)
pis, palette, pattern_count, reverse_fn = overlap_preprocess(r, 3, 3)

config = WFCConfig(
    h = 300,
    w = 300,
    model = make_adacent_model(pattern_count, pis),
    device="cpu"
)

result = run(config)
output = reverse_fn(result)
write_png(output, "output.png")