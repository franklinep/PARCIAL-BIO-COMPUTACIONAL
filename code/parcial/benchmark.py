"""
Benchmark de tiempo y memoria: NW clasico O(mn) vs Hirschberg O(min(m,n)).

Se prueban tamanios crecientes (200, 400, 800, 1600, 3200, 6400 bases) y
se mide:
  - Tiempo de ejecucion (time.perf_counter, 1 corrida por punto + warmup).
  - Memoria pico durante la ejecucion (memory_profiler.memory_usage).

Las secuencias se construyen concatenando copias del CDS de citocromo C
humano y raton (mutados para introducir algo de variabilidad), de modo
que sigan siendo datos *reales* en su base pero con longitudes mayores.
"""
from __future__ import annotations
import time
import random
import tracemalloc
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import List

from memory_profiler import memory_usage

from alignment import SimpleMatrix, NeedlemanWunsch
from parcial.hirschberg import HirschbergAligner


@dataclass
class BenchmarkPoint:
    length: int          # longitud aproximada de las secuencias
    algorithm: str       # "NW" o "Hirschberg"
    time_s: float
    peak_mem_mib: float      # RSS pico durante la ejecucion (incluye baseline)
    delta_mem_mib: float     # RSS adicional respecto al baseline
    trace_peak_mib: float = float("nan")  # pico Python medido con tracemalloc


def build_pair(base_human: str, base_mouse: str, target_len: int, seed: int = 0) -> tuple[str, str]:
    """Construye un par de secuencias de longitud ~target_len.
    Concatena copias del CDS y aplica mutaciones puntuales para evitar
    que las secuencias sean trivialmente identicas."""
    rng = random.Random(seed)
    s1 = (base_human * ((target_len // len(base_human)) + 1))[:target_len]
    s2 = (base_mouse * ((target_len // len(base_mouse)) + 1))[:target_len]
    # ~2% mutaciones aleatorias en s2
    alphabet = list({c for c in s1 + s2 if c in "ACGT"})
    s2 = list(s2)
    n_mut = max(1, target_len // 50)
    for _ in range(n_mut):
        idx = rng.randrange(len(s2))
        s2[idx] = rng.choice(alphabet)
    return s1, "".join(s2)


def time_and_memory(fn, trace_python_memory: bool = False) -> tuple[float, float, float, float]:
    """Mide tiempo, RSS del proceso y memoria Python de una funcion."""
    baseline = memory_usage(-1, interval=0.05, timeout=0.1, max_usage=True)
    t0 = time.perf_counter()
    mem_trace = memory_usage(
        (fn, (), {}),
        interval=0.02,
        max_usage=True,
        retval=False,
    )
    t1 = time.perf_counter()
    peak = mem_trace if isinstance(mem_trace, float) else max(mem_trace)

    if trace_python_memory:
        # Se mide en una segunda corrida para no capturar asignaciones internas
        # de memory_profiler dentro del pico de tracemalloc.
        gc.collect()
        tracemalloc.start()
        fn()
        _, trace_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        trace_peak_mib = trace_peak / (1024 * 1024)
    else:
        trace_peak_mib = float("nan")

    return (t1 - t0), peak, peak - baseline, trace_peak_mib


def _trace_label(trace_peak_mib: float) -> str:
    if trace_peak_mib != trace_peak_mib:
        return "py_peak=     n/a"
    return f"py_peak={trace_peak_mib:8.4f} MiB"


def run_benchmark(
    base_human: str,
    base_mouse: str,
    lengths: List[int],
    gap: int = -1,
    trace_python_memory: bool = False,
) -> List[BenchmarkPoint]:
    matrix = SimpleMatrix()
    points: List[BenchmarkPoint] = []
    nw = NeedlemanWunsch(matrix, gap_penalty=gap)
    hb = HirschbergAligner(matrix, gap_penalty=gap)

    for L in lengths:
        s1, s2 = build_pair(base_human, base_mouse, L)
        print(f"\n  Longitud {L}  (|s1|={len(s1)}, |s2|={len(s2)})")

        # NW clasico (puede explotar para L grande)
        try:
            t, peak, delta, trace_peak = time_and_memory(
                lambda: nw.align(s1, s2),
                trace_python_memory=trace_python_memory,
            )
            print(
                f"    NW          t={t*1000:8.1f} ms   "
                f"rss_d={delta:8.3f} MiB   {_trace_label(trace_peak)}"
            )
            points.append(BenchmarkPoint(L, "NW", t, peak, delta, trace_peak))
        except MemoryError:
            print(f"    NW          MemoryError (matriz demasiado grande)")
            points.append(
                BenchmarkPoint(
                    L, "NW", float("inf"), float("inf"), float("inf"), float("inf")
                )
            )

        # Hirschberg
        t, peak, delta, trace_peak = time_and_memory(
            lambda: hb.align(s1, s2),
            trace_python_memory=trace_python_memory,
        )
        print(
            f"    Hirschberg  t={t*1000:8.1f} ms   "
            f"rss_d={delta:8.3f} MiB   {_trace_label(trace_peak)}"
        )
        points.append(BenchmarkPoint(L, "Hirschberg", t, peak, delta, trace_peak))

    return points


def points_to_csv(points: List[BenchmarkPoint], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(
            "length,algorithm,time_s,peak_mem_mib,delta_mem_mib,"
            "trace_peak_mib\n"
        )
        for p in points:
            f.write(
                f"{p.length},{p.algorithm},{p.time_s:.6f},"
                f"{p.peak_mem_mib:.3f},{p.delta_mem_mib:.3f},"
                f"{p.trace_peak_mib:.6f}\n"
            )
