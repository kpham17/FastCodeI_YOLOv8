#include "universal_concat.h"
#include "fused_upconcat.h"
#include "residual_add.h"
#include <cassert>
int main() {
    float a[16]={1},b[16]={2},out[32]={0};
    universal_concat_view v[]={{a,16,16},{b,16,16}};
    assert(!universal_concat_f32(v,2,out,32,1,1,1));
    assert(!residual_add(out,32,a,16,b,16,1,1,16));
    assert(out[0]==3);
    float skip[64]={0},upout[128]={0};
    assert(!fused_upconcat_f32(a,16,skip,64,upout,128,1,1,16,16));
}
