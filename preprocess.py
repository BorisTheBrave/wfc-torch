import torch as t

def to_i32(i):
    i = i.to(t.int32)
    return (i[0] << 16) + (i[1] << 8) + (i[2] << 0)

def from_i32(i32):
    return t.stack([(i32 >> 16) & 0xFF, (i32 >> 8) & 0xFF, (i32 >> 0) & 0xFF], dim=0).to(t.uint8)

def make_palette(i32s):
    return t.sort(t.unique(t.concat([i32.flatten() for i32 in i32s])))[0]

def palettize(palette, i32):
    return t.searchsorted(palette, i32)

def unpalettize(palette, pi):
    return palette[pi]


def adj_preprocess(imgs): # Returns: pis, palette, pattern_count, reverse_fn
    i32s = [to_i32(img) for img in imgs]

    palette = make_palette(i32s)
    pattern_count = len(palette)

    pis = [palettize(palette, i32) for i32 in i32s]

    def reverse_fn(pi):
        return from_i32(unpalettize(palette, pi))
    
    return pis, palette, pattern_count, reverse_fn

def overlap_preprocess(imgs, nx, ny):
    i32s = [to_i32(img) for img in imgs]
    tile_palette = make_palette(i32s)
    tis = [palettize(tile_palette, i32) for i32 in i32s]
    # ti contains an int indicating the index. Now stride to blocks
    all_blocks = []
    for ti in tis:
        h, w = ti.shape
        hs, ws = ti.stride()
        blocks = ti.as_strided((h-ny+1, w-nx+1,ny,nx), (hs, ws, hs, ws)).flatten(0, 1)
        all_blocks.append(blocks)

    all_blocks = t.concat(all_blocks)
    palette, inverse = t.unique(blocks, dim=0, return_inverse=True)

    pis = []
    i=0
    for ti in tis:
        h, w = ti.shape
        i2 = i +((h-ny+1)*(w-nx+1))
        pi = inverse[i:i2].unflatten(0, (h-ny+1, w-nx+1))
        i2 = i
        pis.append(pi)

    def reverse_fn(pi):
        ti = palette[:, 0, 0][pi]
        return from_i32(unpalettize(tile_palette, ti))

    return pi, palette, palette.shape[0], reverse_fn