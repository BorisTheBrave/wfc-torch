import torch as t
from torch import sparse
from torchvision.io import read_image, write_png
from collections import namedtuple
import numpy
from wfc import run, make_adacent_model, WFCConfig, Model
from preprocess import adj_preprocess, overlap_preprocess, make_rotations
from torch.profiler import profile, record_function, ProfilerActivity

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

if False:
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
write_png(output, "output.png")
