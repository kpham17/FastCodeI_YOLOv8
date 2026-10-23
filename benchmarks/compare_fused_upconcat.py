"""Check and time only NHWC PyTorch upsample+cat versus fused VL_MIXED."""
import argparse
import csv
import ctypes
import os
from pathlib import Path
import random
import socket
import statistics
import time

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results" / "nhwc_vl"
SHAPES = [(640, "A", 20, 256, 128), (640, "B", 40, 128, 64),
          (1024, "A", 32, 256, 128), (1024, "B", 64, 128, 64),
          (2048, "A", 64, 256, 128), (2048, "B", 128, 128, 64)]


def start():
    # Pin first, so all tensor memory is created on the selected CPU.
    os.sched_setaffinity(0, {0})
    flags = next(line.split(":", 1)[1].split()
                 for line in Path("/proc/cpuinfo").read_text().splitlines()
                 if line.startswith("flags"))
    if not {"avx512f", "avx512vl"}.issubset(flags):
        raise RuntimeError("VL_MIXED requires AVX-512F and AVX-512VL")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    library = ctypes.CDLL(str(ROOT / "kernels" / "libfused_upconcat.so"))
    for stage in "ab":
        fn = getattr(library, "fused_upconcat_" + stage)
        fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_size_t] * 2
        fn.restype = None
    return torch, library


def same_bits(torch, a, b):
    return torch.equal(a.contiguous().view(torch.int32),
                       b.contiguous().view(torch.int32))


def calls(torch, library, up, skip, out, stage):
    h, w = up.shape[2:]
    tmp = torch.empty((1, up.shape[1], 2*h, 2*w),
                      dtype=torch.float32, memory_format=torch.channels_last)
    fn = getattr(library, "fused_upconcat_" + stage.lower())
    pointers = [ctypes.c_void_p(t.data_ptr()) for t in (up, skip, out)]

    def baseline():
        torch.ops.aten.upsample_nearest2d.out(up, [2*h, 2*w], 2.0, 2.0, out=tmp)
        return torch.cat((tmp, skip), dim=1, out=out)

    def vl_mixed():
        fn(*pointers, h, w)
        return out

    # Both versions use the exact same inputs and final output buffer.
    return {"PyTorch": baseline, "VL_MIXED": vl_mixed}


def guarded(torch, h, c, offset):
    count = h*h*c
    backing = torch.empty(count + offset + 16, dtype=torch.float32)
    backing.view(torch.int32).fill_(0x5A5A5A5A)
    data = backing[offset:offset+count].view(1, h, h, c).permute(0, 3, 1, 2)
    return data, backing


def check(torch, library):
    count = 0
    generator = torch.Generator().manual_seed(123)
    # Raw bits include NaNs, infinities, signed zero and small values.
    special = torch.tensor([0, -(2**31), 0x7F800000, -0x00800000,
                            0x7FC00001, 0x7FC12345, -0x00354322,
                            0x7F800001, 1], dtype=torch.int32)
    for _, stage, h, cu, cs in SHAPES:
        for offset in (0, 1):
            up, ub = guarded(torch, h, cu, offset)
            skip, sb = guarded(torch, 2*h, cs, 3*offset)
            out, ob = guarded(torch, 2*h, cu+cs, 5*offset)
            for data in (up, skip):
                flat = data.permute(0, 2, 3, 1).reshape(-1).view(torch.int32)
                flat.copy_(torch.randint(-(2**31), 2**31, flat.shape,
                                         dtype=torch.int32, generator=generator))
                flat[:len(special)] = special
            before_u, before_s = ub.clone(), sb.clone()
            functions = calls(torch, library, up, skip, out, stage)
            functions["PyTorch"]()
            reference = out.clone()
            out.permute(0, 2, 3, 1).reshape(-1).view(torch.int32).fill_(0x5A5A5A5A)
            functions["VL_MIXED"]()
            assert same_bits(torch, reference, out), (stage, h, offset)
            assert same_bits(torch, before_u, ub), "input changed"
            assert same_bits(torch, before_s, sb), "skip changed"
            assert bool((ob[:5*offset].view(torch.int32) == 0x5A5A5A5A).all())
            assert bool((ob[5*offset+out.numel():].view(torch.int32) == 0x5A5A5A5A).all())
            count += 1
    print("PASS: all 12 output, input and buffer-boundary checks")
    return count


def benchmark(run):
    torch, library = start()
    rng = random.Random(9000+run)
    torch.manual_seed(123)
    cpu_model = next(line.split(":", 1)[1].strip()
                     for line in Path("/proc/cpuinfo").read_text().splitlines()
                     if line.startswith("model name"))
    RESULTS.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        check(torch, library)
        groups = []
        for image, stage, h, cu, cs in SHAPES:
            up = torch.rand(1, cu, h, h).contiguous(memory_format=torch.channels_last)
            skip = torch.rand(1, cs, 2*h, 2*h).contiguous(memory_format=torch.channels_last)
            out = torch.empty((1, cu+cs, 2*h, 2*h), dtype=torch.float32,
                              memory_format=torch.channels_last)
            functions = calls(torch, library, up, skip, out, stage)
            reference = functions["PyTorch"]().clone()
            assert same_bits(torch, reference, functions["VL_MIXED"]())
            groups.append((image, stage, functions, reference))
        with (RESULTS / ("run-%d.csv" % run)).open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pid", "host", "cpu_model", "torch", "round", "image", "stage",
                             "version", "order", "calls", "ns_per_call"])
            for round_index in range(21):
                rng.shuffle(groups)
                for image, stage, functions, reference in groups:
                    order = list(functions)
                    rng.shuffle(order)
                    for position, name in enumerate(order):
                        fn = functions[name]
                        for _ in range(4):
                            fn()
                        started = time.perf_counter_ns()
                        for _ in range(5):
                            result = fn()
                        elapsed = time.perf_counter_ns() - started
                        assert same_bits(torch, reference, result)
                        writer.writerow([os.getpid(), socket.gethostname(), cpu_model, torch.__version__,
                                         round_index, image, stage, name, position, 5, elapsed/5])
    print("PASS: run %d" % run)


def report():
    medians, pids = {}, set()
    for run in (1, 2, 3):
        with (RESULTS / ("run-%d.csv" % run)).open(newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 21*6*2
        ids = {r["pid"] for r in rows}
        assert len(ids) == 1 and not ids.intersection(pids)
        pids.update(ids)
        expected = {(r, image, stage, version) for r in range(21)
                    for image, stage, _, _, _ in SHAPES for version in ("PyTorch", "VL_MIXED")}
        assert {(int(r["round"]), int(r["image"]), r["stage"], r["version"]) for r in rows} == expected
        assert all(int(r["calls"]) == 5 and float(r["ns_per_call"]) > 0 for r in rows)
        for image, stage, _, _, _ in SHAPES:
            for version in ("PyTorch", "VL_MIXED"):
                values = [float(r["ns_per_call"])/1e6 for r in rows
                          if int(r["image"]) == image and r["stage"] == stage and r["version"] == version]
                medians[run, image, stage, version] = statistics.median(values)
    def time_ms(image, stage, version):
        return statistics.median(medians[run, image, stage, version] for run in (1, 2, 3))
    lines = ["# NHWC VL_MIXED versus PyTorch", "",
             "Both versions double the image height and width, then join the skip input.",
             "Both use NHWC storage and 32-bit floating-point numbers. Input and output buffers are ready before timing.", "",
             "- Baseline: PyTorch nearest-neighbor upsampling, followed by concatenate.",
             "- Our version: the same work combined in one VL_MIXED C function using 256-bit moves and AVX-512VL.",
             "- A is YOLOv8n layers 10+11. B is layers 13+14.", "",
             "Machine: %s, %s, CPU 0, one thread. PyTorch %s." % (rows[0]["host"], rows[0]["cpu_model"], rows[0]["torch"]),
             "Times are in milliseconds. Smaller is better. Speedup is PyTorch time divided by VL_MIXED time.", "",
             "| Image | Stage | PyTorch ms | VL_MIXED ms | Speedup |",
             "|---|---|---:|---:|---:|"]
    for image, stage, _, _, _ in SHAPES:
        a, b = time_ms(image, stage, "PyTorch"), time_ms(image, stage, "VL_MIXED")
        lines.append("| %d | %s | %.4f | %.4f | %.2fx |" % (image, stage, a, b, a/b))
    lines += ["", "A+B adds the two stage times above.", "",
              "| Image | PyTorch A+B ms | VL_MIXED A+B ms | Speedup |",
              "|---|---:|---:|---:|"]
    for image in (640, 1024, 2048):
        a = sum(time_ms(image, s, "PyTorch") for s in "AB")
        b = sum(time_ms(image, s, "VL_MIXED") for s in "AB")
        lines.append("| %d | %.4f | %.4f | %.2fx |" % (image, a, b, a/b))
    lines += ["", "Every output byte matched. Inputs and the bytes around the buffers stayed unchanged. Checks included NaNs, infinities, signed zero, random bits and buffers starting four bytes past an aligned address.", "",
              "Each case was measured in three separate runs. Each run used 21 rounds, four warmup calls and five timed calls per round. The table uses the middle timing from each run, then the middle of those three values.", "",
              "The PyTorch baseline also has its temporary upsample buffer ready before timing. VL_MIXED avoids that temporary buffer. No layout conversion is timed because both versions already use NHWC.", "",
              "These numbers cover only upsampling and concatenation, not the full YOLOv8n model.", ""]
    (RESULTS / "result.md").write_text("\n".join(lines), encoding="utf-8")
    print("PASS: %s written" % (RESULTS / "result.md"))


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--run", type=int, choices=(1, 2, 3))
    action.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.report:
        report()
    elif args.run:
        benchmark(args.run)
    else:
        torch, library = start()
        with torch.inference_mode():
            check(torch, library)


if __name__ == "__main__":
    main()
