"""Extract median self-CUDA time, self-CPU time, and peak CUDA memory per method
from the raw torch.profiler text dumps in reports/figures/profiling/.

Usage: python src/profiling/parse_profiling_logs.py
"""
import re
import statistics
from pathlib import Path

DIR = Path(__file__).resolve().parents[2] / "reports" / "figures" / "profiling"

FILES = {
    "Naive Attention": "naive_attention_layer_5_head_1_tokens_200.txt",
    "Flash Attention": "flash_attention_layer_5_head_1_tokens_200.txt",
    "SVD Nystrom": "nystrom_attention_layer_5_head_1_tokens_200_k_45.txt",
    "K-means": "kmeans_layer_5_head_1_tokens_200_clusters_45.txt",
    "Hokus Pokus": "Hokus_Pokus_layer_5_head_1_tokens_200_k_45.txt",
    "Method 1 (K)": "method_1_layer_5_head_1_tokens_200_k_45.txt",
    "Method 2 (V)": "method_2_layer_5_head_1_tokens_200_k_45.txt",
    "Method 3 (KV sep.)": "method_3_layer_5_head_1_tokens_200_k_45.txt",
    "Method 4 (KV joint)": "method_4_layer_5_head_1_tokens_200_k_45.txt",
}

UNIT_MULT = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
TIME_MULT = {"us": 1e-3, "ms": 1.0, "s": 1e3}  # normalized to ms


def parse_qty(tok, mults):
    m = re.match(r"^(-?[0-9.]+)\s*([A-Za-z]+)$", tok)
    if not m:
        return None
    val, unit = m.groups()
    if unit not in mults:
        return None
    return float(val) * mults[unit]


def parse_file(path):
    """Return (self_cpu_ms, self_cuda_ms, peak_cuda_mem_bytes) lists, one entry per iteration."""
    text = path.read_text(errors="replace")
    iters = re.split(r"--- Iteration \d+ ---", text)[1:]
    self_cpu_ms, self_cuda_ms, peak_cuda_mem_bytes = [], [], []
    for it in iters:
        m = re.search(r"Self CPU time total:\s*([0-9.]+)(ms|us|s)", it)
        if m:
            self_cpu_ms.append(float(m.group(1)) * TIME_MULT[m.group(2)])
        m = re.search(r"Self CUDA time total:\s*([0-9.]+)(ms|us|s)", it)
        if m:
            self_cuda_ms.append(float(m.group(1)) * TIME_MULT[m.group(2)])
        # Peak CUDA memory: max value seen in the "CUDA Mem" column (3rd from the end)
        # across all op rows in this iteration block.
        max_mem = 0.0
        for line in it.splitlines():
            if not re.match(r"^\s*\S", line) or line.strip().startswith("-"):
                continue
            cols = re.split(r"\s{2,}", line.strip())
            if len(cols) < 15:
                continue
            val = parse_qty(cols[-3], UNIT_MULT)
            if val is not None and val > max_mem:
                max_mem = val
        if max_mem > 0:
            peak_cuda_mem_bytes.append(max_mem)
    return self_cpu_ms, self_cuda_ms, peak_cuda_mem_bytes


def fmt_mem(b):
    return f"{b / 1024**2:.2f} MB" if b >= 1024**2 else f"{b / 1024:.1f} KB"


if __name__ == "__main__":
    header = f"{'Method':<20}{'n':>3}{'CPU med(ms)':>13}{'CUDA med(us)':>14}{'Peak CUDA mem':>15}"
    print(header)
    for name, fname in FILES.items():
        cpu_ms, cuda_ms, mem_b = parse_file(DIR / fname)
        cuda_us = [x * 1000 for x in cuda_ms]
        cpu_med = statistics.median(cpu_ms)
        cuda_med = statistics.median(cuda_us)
        mem_med = statistics.median(mem_b) if mem_b else 0
        print(f"{name:<20}{len(cpu_ms):>3}{cpu_med:>13.3f}{cuda_med:>14.2f}{fmt_mem(mem_med):>15}")
