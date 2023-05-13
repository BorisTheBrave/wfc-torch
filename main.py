import torch as t
from torch import sparse
from torchvision.io import read_image, write_png
from collections import namedtuple
import numpy
from wfc import run, make_adacent_model, WFCConfig, Model
from preprocess import adj_preprocess, overlap_preprocess, make_rotations
from torch.profiler import profile, record_function, ProfilerActivity
import argparse

LOG_LEVEL = 0

parser = argparse.ArgumentParser('wfc', 'Runs the WaveFunctionCollapse algorithm on images', add_help=False)

parser.add_argument("--help", action="help")
parser.add_argument("-i", "--input", default="Angular.png")
parser.add_argument("-o", "--output", default="output.png")
parser.add_argument("-w", "--width", type=int, default=50)
parser.add_argument("-h", "--height", type=int, default=50)
parser.add_argument("-n", "--overlap-size", type=int, default=None)
parser.add_argument("-r", "--rotations", type=int, default=4)
parser.add_argument("-d", "--device", default="cpu")
parser.add_argument("--profile", type=bool, default=False)
parser.add_argument("--seed", type=int, default=None)

args = parser.parse_args()

i = read_image(args.input)

r = make_rotations(i, args.rotations)

w = args.width
h = args.height
n = args.overlap_size

if n:
    pis, palette, pattern_count, reverse_fn = overlap_preprocess(r, n, n)
    w -= n-1
    h -= n-1
else:
    pis, palette, pattern_count, reverse_fn = adj_preprocess(r)

config = WFCConfig(
    h = w,
    w = h,
    model = make_adacent_model(pattern_count, pis),
    device=args.device,
    seed=args.seed,
)

if args.profile:
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_stack=True) as prof:
        with record_function("run"):
            result = run(config)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(prof.key_averages(group_by_stack_n=2).table(sort_by="cpu_time_total", row_limit=10))
    prof.export_chrome_trace(f"trace_{config.device}.json")
    prof.export_stacks(f"profiler_stacks_{config.device}.txt", "self_cpu_time_total")
else:
    result = run(config)
output = reverse_fn(result)
write_png(output, args.output)
