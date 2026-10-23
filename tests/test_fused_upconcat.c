#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fused_upconcat.h"

/* Detect wrong nearest-neighbor coordinates, channel order, alignment
 * assumptions, input writes, output overruns and altered FP32 bit patterns. */
typedef struct {
    unsigned char *raw;
    float *data;
    size_t bytes, prefix, total;
} buffer;

static buffer make_buffer(size_t elements, size_t offset)
{
    buffer b;
    b.bytes = elements * sizeof(float);
    b.prefix = 64 + offset * sizeof(float);
    b.total = b.prefix + b.bytes + 64;
    b.raw = malloc(b.total);
    if (!b.raw) { fputs("allocation failed\n", stderr); exit(2); }
    memset(b.raw, 0xa5, b.total);
    b.data = (float *)(void *)(b.raw + b.prefix);
    return b;
}

static uint32_t bits(const float *p)
{
    uint32_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static void fill_bits(buffer b, uint32_t seed)
{
    const uint32_t special[] = {
        0, UINT32_C(0x80000000), UINT32_C(0x7f800000), UINT32_C(0xff800000),
        UINT32_C(0x7fc00001), UINT32_C(0x7f800001), UINT32_C(0xffc12345), 1
    };
    for (size_t i = 0; i < b.bytes / sizeof(float); ++i) {
        seed = seed * UINT32_C(1664525) + UINT32_C(1013904223);
        uint32_t v = i < sizeof(special) / sizeof(special[0]) ? special[i] : seed;
        memcpy(b.data + i, &v, sizeof(v));
    }
}

static int guards_unchanged(buffer b)
{
    for (size_t i = 0; i < b.prefix; ++i)
        if (b.raw[i] != 0xa5) return 0;
    for (size_t i = b.prefix + b.bytes; i < b.total; ++i)
        if (b.raw[i] != 0xa5) return 0;
    return 1;
}

static int check_case(size_t h, size_t w, int stage, size_t offset)
{
    size_t cu = stage == 0 ? 256 : 128, cs = cu / 2, co = cu + cs;
    buffer up = make_buffer(h * w * cu, offset);
    buffer skip = make_buffer(4 * h * w * cs, 3 * offset);
    buffer out = make_buffer(4 * h * w * co, 5 * offset);
    fill_bits(up, 123); fill_bits(skip, 456);
    unsigned char *before_up = malloc(up.total), *before_skip = malloc(skip.total);
    if (!before_up || !before_skip) exit(2);
    memcpy(before_up, up.raw, up.total);
    memcpy(before_skip, skip.raw, skip.total);
    if (stage == 0) fused_upconcat_a(up.data, skip.data, out.data, h, w);
    else fused_upconcat_b(up.data, skip.data, out.data, h, w);

    for (size_t y = 0; y < 2 * h; ++y) {
        for (size_t x = 0; x < 2 * w; ++x) {
            for (size_t c = 0; c < co; ++c) {
                const float *source = c < cu ?
                    up.data + ((y / 2) * w + x / 2) * cu + c :
                    skip.data + (y * 2 * w + x) * cs + (c - cu);
                const float *actual = out.data + (y * 2 * w + x) * co + c;
                if (bits(actual) != bits(source)) {
                    fprintf(stderr, "FAIL stage=%c h=%zu w=%zu offset=%zu at %zu,%zu,%zu\n",
                            'A' + stage, h, w, offset, y, x, c);
                    return 1;
                }
            }
        }
    }
    if (memcmp(before_up, up.raw, up.total) ||
        memcmp(before_skip, skip.raw, skip.total) || !guards_unchanged(out)) {
        fprintf(stderr, "FAIL: input or buffer boundary modified\n"); return 1;
    }
    free(before_up); free(before_skip);
    free(up.raw); free(skip.raw); free(out.raw);
    return 0;
}

int main(void)
{
    const size_t shapes[][2] = {
        {0,0},{0,3},{3,0},{1,1},{1,5},{3,1},{3,5},
        {20,20},{32,32},{40,40},{64,64},{128,128}
    };
    size_t cases = 0;
    for (int stage = 0; stage < 2; ++stage)
        for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); ++i)
            for (size_t offset = 0; offset < 2; ++offset) {
                if (check_case(shapes[i][0], shapes[i][1], stage, offset)) return 1;
                ++cases;
            }
    printf("PASS: %zu fused NHWC VL_MIXED bit-pattern, rectangular/empty-shape, alignment and boundary cases\n", cases);
    return 0;
}
