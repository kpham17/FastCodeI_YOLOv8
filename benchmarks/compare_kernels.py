#!/usr/bin/env python3
"""Compare custom NHWC C kernels against PyTorch CPU baselines."""

import argparse
import ctypes
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional


def output_size(size, kernel, stride, padding):
    return (size + 2 * padding - kernel) // stride + 1


def timed(function, warmups, repeats):
    for _ in range(warmups):
        function()
    start = time.perf_counter()
    for _ in range(repeats):
        function()
    return (time.perf_counter() - start) * 1000.0 / repeats


def load_library(path):
    library = ctypes.CDLL(str(path))
    library.fast_batchnorm.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
    ]
    library.fast_batchnorm.restype = ctypes.c_int
    library.fast_maxpool2d.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    library.fast_maxpool2d.restype = ctypes.c_int
    return library


def pointer(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def print_result(name, flops, pytorch_ms, custom_ms):
    def throughput(milliseconds):
        return flops / (milliseconds * 1e6)

    custom = "not implemented"
    if custom_ms is not None:
        custom = "{:.3f} ms ({:.3f} GFLOP/s)".format(
            custom_ms, throughput(custom_ms))
    print(
        "{:12s} | PyTorch {:10.3f} ms ({:8.3f} GFLOP/s) | custom {}".format(
            name, pytorch_ms, throughput(pytorch_ms), custom))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, default=None,
                        help="custom shared library; omit while kernels are skeletons")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--shape", nargs=4, type=int, metavar=("N", "H", "W", "C"),
                        default=(1, 80, 80, 64))
    args = parser.parse_args()

    if args.warmups < 0 or args.repeats <= 0:
        parser.error("--warmups must be non-negative and --repeats must be positive")
    if min(args.shape) <= 0:
        parser.error("all dimensions in --shape must be positive")
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)

    N, H, W, C = args.shape
    library = load_library(args.library) if args.library else None
    rng = np.random.RandomState(18645)
    input_nhwc = np.ascontiguousarray(
        rng.standard_normal((N, H, W, C)).astype(np.float32))
    running_mean = np.ascontiguousarray(rng.standard_normal(C).astype(np.float32))
    running_var = np.ascontiguousarray((rng.rand(C) + 0.5).astype(np.float32))
    gamma = np.ascontiguousarray(rng.standard_normal(C).astype(np.float32))
    beta = np.ascontiguousarray(rng.standard_normal(C).astype(np.float32))

    input_torch = torch.from_numpy(input_nhwc).permute(0, 3, 1, 2)
    mean_torch = torch.from_numpy(running_mean)
    var_torch = torch.from_numpy(running_var)
    gamma_torch = torch.from_numpy(gamma)
    beta_torch = torch.from_numpy(beta)
    bn_call = lambda: functional.batch_norm(
        input_torch, mean_torch, var_torch, gamma_torch, beta_torch,
        training=False, momentum=0.1, eps=1e-5)
    bn_ms = timed(bn_call, args.warmups, args.repeats)

    kernel_h = kernel_w = 2
    stride_h = stride_w = 2
    padding_h = padding_w = 0
    pool_h = output_size(H, kernel_h, stride_h, padding_h)
    pool_w = output_size(W, kernel_w, stride_w, padding_w)
    pool_call = lambda: functional.max_pool2d(
        input_torch, (kernel_h, kernel_w), (stride_h, stride_w),
        (padding_h, padding_w))
    pool_ms = timed(pool_call, args.warmups, args.repeats)

    bn_flops = 4 * N * H * W * C
    pool_flops = N * pool_h * pool_w * C * (kernel_h * kernel_w - 1)
    if library is None:
        print_result("batchnorm", bn_flops, bn_ms, None)
        print_result("maxpool2d", pool_flops, pool_ms, None)
        print("Custom results skipped: pass --library after implementing the skeletons.")
        return

    bn_output = np.empty_like(input_nhwc)
    bn_custom = lambda: library.fast_batchnorm(
        pointer(input_nhwc), pointer(bn_output), N, H, W, C,
        pointer(running_mean), pointer(running_var), pointer(gamma),
        pointer(beta), ctypes.c_float(1e-5))
    status = bn_custom()
    if status == 0:
        print_result("batchnorm", bn_flops, bn_ms,
                     timed(bn_custom, args.warmups, args.repeats))
    else:
        print_result("batchnorm", bn_flops, bn_ms, None)
        print("Custom batchnorm skipped: fast_batchnorm returned an error.")

    pool_output = np.empty((N, pool_h, pool_w, C), dtype=np.float32)
    pool_custom = lambda: library.fast_maxpool2d(
        pointer(input_nhwc), pointer(pool_output), N, H, W, C,
        kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w)
    status = pool_custom()
    if status == 0:
        print_result("maxpool2d", pool_flops, pool_ms,
                     timed(pool_custom, args.warmups, args.repeats))
    else:
        print_result("maxpool2d", pool_flops, pool_ms, None)
        print("Custom maxpool2d skipped: fast_maxpool2d returned an error.")


if __name__ == "__main__":
    main()
