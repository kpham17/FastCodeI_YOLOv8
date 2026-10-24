#include "residual_add.h"
#include <cassert>
#include <cstdio>

int main() {
    float conv[] = {1, 2, 3, 4};
    const float dense[] = {10, 20, 30, 40};
    residual_add_contiguous(conv, conv, dense, 2, 2);
    assert(conv[0] == 11 && conv[1] == 22 && conv[2] == 33 && conv[3] == 44);

    const float parent[] = {-1, -1, 5, 6, -1, -1, 7, 8};
    float out[4];
    residual_add_strided(out, conv, parent + 2, 2, 2, 4);
    assert(out[0] == 16 && out[1] == 28 && out[2] == 40 && out[3] == 52);
    std::puts("PASS: residual add C++ linkage and both APIs");
}
