"""Check the fused kernel's 256-bit VL build using GNU objdump."""
from pathlib import Path
import re
import subprocess


def main():
    root = Path(__file__).resolve().parent.parent
    assembly = subprocess.check_output(
        ["objdump", "-d", "-Mintel", "--no-show-raw-insn",
         str(root / "kernels" / "libfused_upconcat.so")], universal_newlines=True)
    functions = dict(re.findall(
        r"^[0-9a-f]+ <([^>]+)>:\n(.*?)(?=^[0-9a-f]+ <|\Z)",
        assembly, re.M | re.S))
    for name in ("fused_upconcat_a", "fused_upconcat_b"):
        body = functions.get(name)
        if body is None:
            raise SystemExit("FAIL: missing function " + name)
        if re.search(r"\bzmm\d+\b", body):
            raise SystemExit("FAIL: " + name + " contains 512-bit ZMM instructions")
        if not any(int(n) >= 16 for n in re.findall(r"\bymm(\d+)\b", body)):
            raise SystemExit("FAIL: " + name + " does not use the extra YMM registers")
        frame_pointer = bool(re.search(r"\bmov\s+rbp,rsp\b", body))
        stack = r"\[(?:rsp|rbp)\b" if frame_pointer else r"\[rsp\b"
        for line in body.splitlines():
            if re.search(r"\b[xyz]mm\d+\b", line) and re.search(stack, line):
                raise SystemExit("FAIL: vector stack access in " + name + ": " + line.strip())
        print("PASS: " + name + ": high YMM registers, no ZMM instructions or vector spills")


if __name__ == "__main__":
    main()
