#ifndef UC_WORKLOADS_H
#define UC_WORKLOADS_H
#include <stddef.h>
typedef struct {
    int site;
    size_t scale, count, channels[4];
} uc_workload;
static const uc_workload uc_workloads[] = {
    {2, 4, 2, {32, 16, 0, 0}},     {4, 8, 3, {64, 32, 32, 0}},       {6, 16, 3, {128, 64, 64, 0}},
    {8, 32, 2, {256, 128, 0, 0}},  {9, 32, 4, {128, 128, 128, 128}}, {12, 16, 2, {128, 64, 0, 0}},
    {15, 8, 2, {64, 32, 0, 0}},    {17, 16, 2, {64, 128, 0, 0}},     {18, 16, 2, {128, 64, 0, 0}},
    {20, 32, 2, {128, 256, 0, 0}}, {21, 32, 2, {256, 128, 0, 0}}};
#define UC_SITE_COUNT (sizeof(uc_workloads) / sizeof(uc_workloads[0]))
#endif
