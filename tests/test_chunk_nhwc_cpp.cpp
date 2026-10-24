#include "chunk_nhwc.h"
int main(){float s[4]={1,2,3,4},a[2],b[2];chunk2_nhwc_f32(a,b,s,1,2);return a[0]!=1||a[1]!=2||b[0]!=3||b[1]!=4;}
