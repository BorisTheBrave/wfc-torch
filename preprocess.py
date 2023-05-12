import torch as t

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


def adj_preprocess(img): # Returns: pi, palette, pattern_count, reverse_fn
    i32 = to_i32(img)

    palette = make_palette(i32)
    pattern_count = len(palette)

    pi = palettize(palette, i32)

    def reverse_fn(pi):
        return from_i32(unpalettize(palette, pi))
    
    return pi, palette, pattern_count, reverse_fn

def overlap_preprocess(img, nx, ny):
    i32 = to_i32(img)
    tile_palette = make_palette(i32)
    ti = palettize(tile_palette, i32)
    # ti contains an int indicating the index. Now stride to blocks
    h, w = ti.shape
    hs, ws = ti.stride()
    blocks = ti.as_strided((h-ny+1, w-nx+1,ny,nx), (hs, ws, hs, ws)).flatten(0, 1)
    palette, inverse = t.unique(blocks, dim=0, return_inverse=True)

    pi = inverse.unflatten(0, (h-ny+1, w-nx+1))

    def reverse_fn(pi):
        ti = palette[:, 0, 0][pi]
        return from_i32(unpalettize(tile_palette, ti))

    return pi, palette, palette.shape[0], reverse_fn