#include "universal_concat_nhwc.h"
#include <cassert>
#include <cstring>
int main() {
    float src[] = {1, 2, 3}, out[3];
    uc_input_view input = {src, 3, 3, 3};
    assert(universal_concat_nhwc_f32(&input, 1, out, 3, 1, 1, 1) == UC_OK);
    assert(std::memcmp(src, out, sizeof(src)) == 0);
}
