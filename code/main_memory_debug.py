"""
Debug de memoria para comparar NW clasico, NW de dos filas y Hirschberg.

Este script usa tracemalloc, que mide asignaciones Python con mas detalle
que memory_profiler. Es util para entender por que el benchmark principal
puede mostrar 0.00 MiB en Hirschberg: el aumento existe, pero puede ser
demasiado pequeno para el muestreo por RSS del proceso.

Ejecutar desde la raiz del proyecto:

    python code/main_memory_debug.py

Tambien se pueden pasar longitudes:

    python code/main_memory_debug.py 100 200 400
"""
from __future__ import annotations

import gc
import sys
import time
import tracemalloc
from pathlib import Path
from sys import getsizeof
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from Bio import SeqIO

from alignment import NeedlemanWunsch, SimpleMatrix
from alignment.result import AlignmentResult
from parcial.benchmark import build_pair
from parcial.hirschberg import HirschbergAligner
from parcial.nw_two_rows import nw_score_only


DATASET = ROOT / "dataset"


def deep_size(obj: Any, seen: set[int] | None = None) -> int:
    """Estimacion recursiva del tamano retenido por un objeto Python."""
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    size = getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(deep_size(k, seen) + deep_size(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(deep_size(item, seen) for item in obj)
    elif hasattr(obj, "__dict__"):
        size += deep_size(vars(obj), seen)
    return size


def mib(value_bytes: int | float) -> float:
    return value_bytes / (1024 * 1024)


def kib(value_bytes: int | float) -> float:
    return value_bytes / 1024


def measure(label: str, fn: Callable[[], Any]) -> tuple[Any, float, int, int]:
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed_s = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    retained = deep_size(result)
    print(
        f"  {label:<14} t={elapsed_s * 1000:9.1f} ms  "
        f"peak={mib(peak):8.4f} MiB ({kib(peak):8.1f} KiB)  "
        f"retenido={mib(retained):8.4f} MiB ({kib(retained):8.1f} KiB)"
    )
    return result, elapsed_s, current, peak


def matrix_cells(result: AlignmentResult) -> int:
    return sum(len(row) for row in result.score_matrix)


def debug_length(length: int, base_human: str, base_mouse: str) -> None:
    matrix = SimpleMatrix(match=1, mismatch=-1)
    gap = -1
    s1, s2 = build_pair(base_human, base_mouse, length)

    nw = NeedlemanWunsch(matrix, gap_penalty=gap)
    hb = HirschbergAligner(matrix, gap_penalty=gap)

    print("\n" + "=" * 86)
    print(f"Longitud objetivo L={length}  |s1|={len(s1)}  |s2|={len(s2)}")
    print("=" * 86)

    expected_nw_cells = (len(s1) + 1) * (len(s2) + 1)
    expected_two_row_cells = 2 * (min(len(s1), len(s2)) + 1)
    print("Celdas DP esperadas:")
    print(f"  NW clasico   : {(len(s1) + 1)} x {(len(s2) + 1)} = {expected_nw_cells:,} celdas")
    print(f"  Dos filas/HB : aprox. {expected_two_row_cells:,} celdas activas por paso")
    print(f"  Ratio teorico: NW guarda ~{expected_nw_cells / expected_two_row_cells:,.1f} veces mas celdas")
    print()

    nw_result, _, _, _ = measure("NW clasico", lambda: nw.align(s1, s2))
    two_row_score, _, _, _ = measure("NW dos filas", lambda: nw_score_only(s1, s2, matrix, gap))
    hb_result, _, _, _ = measure("Hirschberg", lambda: hb.align(s1, s2))

    print("\nResultados:")
    print(f"  score NW clasico  : {nw_result.score}")
    print(f"  score dos filas   : {two_row_score}")
    print(f"  score Hirschberg  : {hb_result.score}")
    print(f"  scores coinciden? : {nw_result.score == two_row_score == hb_result.score}")
    print(f"  matriz retenida NW: {matrix_cells(nw_result):,} celdas")
    print(f"  matriz retenida HB: {matrix_cells(hb_result):,} celdas")


def parse_lengths() -> list[int]:
    if len(sys.argv) > 1:
        return [int(arg) for arg in sys.argv[1:]]
    return [100, 200, 400]


def main() -> None:
    base_human = str(SeqIO.read(DATASET / "cytc_human_cds.fasta", "fasta").seq)
    base_mouse = str(SeqIO.read(DATASET / "cytc_mouse_cds.fasta", "fasta").seq)

    print("Debug de memoria con tracemalloc")
    print("Matriz SIMPLE match=+1 mismatch=-1, gap=-1")
    print(
        "\nLectura de columnas:\n"
        "  peak     = maximo de memoria Python asignada durante la funcion.\n"
        "  retenido = tamano aproximado del objeto devuelto al terminar.\n"
    )

    for length in parse_lengths():
        debug_length(length, base_human, base_mouse)

    print("\nNota:")
    print(
        "  Si memory_profiler muestra 0.00 MiB para Hirschberg pero aqui aparece un\n"
        "  peak pequeno, ambas cosas son compatibles. Hirschberg si asigna memoria;\n"
        "  simplemente asigna mucha menos que NW y puede quedar oculto por el baseline\n"
        "  o por la resolucion de muestreo del benchmark principal."
    )


if __name__ == "__main__":
    main()
