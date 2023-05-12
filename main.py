import torch as t
from torch import sparse
from torchvision.io import read_image, write_png
from collections import namedtuple
import numpy
from wfc import run, make_adacent_model, WFCConfig, Model
from preprocess import adj_preprocess, overlap_preprocess

LOG_LEVEL = 0



i = read_image("Angular.png")

#pi, palette, pattern_count, reverse_fn = adj_preprocess(i)
pi, palette, pattern_count, reverse_fn = overlap_preprocess(i, 2, 2)

config = WFCConfig(
    h = 10,
    w = 10,
    model = make_adacent_model(pattern_count, pi)
)

result = run(config)
output = reverse_fn(result)
write_png(output, "output.png")