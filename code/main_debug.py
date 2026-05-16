"""
Debug pequeno para entender Needleman-Wunsch e Hirschberg.

Muestra:
  - La matriz de sustitucion que se esta usando.
  - La matriz DP completa de NW para secuencias pequenas.
  - El alineamiento y score de NW clasico.
  - El alineamiento y score de Hirschberg.
  - Una comparacion con la version de dos filas.

Ejecutar desde la raiz del proyecto:

    python code/main_debug.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from alignment import BiopythonMatrix, NeedlemanWunsch, SimpleMatrix
from alignment.matrices import SubstitutionMatrix
from parcial.hirschberg import HirschbergAligner
from parcial.nw_two_rows import nw_score_last_row, nw_score_only


def header(text: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {text}")
    print("=" * 78)


def format_substitution_matrix(matrix: SubstitutionMatrix, alphabet: str) -> str:
    labels = list(alphabet)
    width = max(5, max(len(str(matrix.score(a, b))) for a in labels for b in labels) + 2)
    lines = ["".rjust(width) + "".join(symbol.rjust(width) for symbol in labels)]
    for a in labels:
        row = [str(matrix.score(a, b)).rjust(width) for b in labels]
        lines.append(a.rjust(width) + "".join(row))
    return "\n".join(lines)


def format_alignment(a: str, b: str) -> str:
    mid = "".join("|" if x == y and x != "-" else (" " if "-" in (x, y) else ".")
                  for x, y in zip(a, b))
    return "\n".join([a, mid, b])


def explain_memory_columns(length_a: int, length_b: int) -> None:
    nw_cells = (length_a + 1) * (length_b + 1)
    two_row_cells = 2 * (min(length_a, length_b) + 1)
    print("\nMemoria esperada por estructura DP:")
    print(f"  NW clasico     : guarda {(length_a + 1)} x {(length_b + 1)} = {nw_cells} celdas")
    print(f"  Dos filas/HB   : guarda aprox. 2 x ({min(length_a, length_b)} + 1) = {two_row_cells} celdas por paso")
    print(f"  Relacion celdas: NW usa ~{nw_cells / two_row_cells:.1f} veces mas celdas en este ejemplo")


def debug_pair(
    title: str,
    seq1: str,
    seq2: str,
    matrix: SubstitutionMatrix,
    gap: int,
    alphabet: str,
) -> None:
    header(title)
    print(f"Seq1 = {seq1}")
    print(f"Seq2 = {seq2}")
    print(f"Matriz = {matrix.name}, gap = {gap}")

    print("\nMatriz de sustitucion:")
    print(format_substitution_matrix(matrix, alphabet))

    nw = NeedlemanWunsch(matrix, gap_penalty=gap)
    hb = HirschbergAligner(matrix, gap_penalty=gap)

    nw_result = nw.align(seq1, seq2, "Seq1", "Seq2")
    hb_result = hb.align(seq1, seq2, "Seq1", "Seq2")
    last_row = nw_score_last_row(seq1, seq2, matrix, gap)
    two_row_score = nw_score_only(seq1, seq2, matrix, gap)

    print("\nMatriz DP completa de Needleman-Wunsch:")
    print(nw.format_score_matrix(seq1, seq2, nw_result.score_matrix))

    print("\nNW clasico:")
    print(f"  score final = {nw_result.score}")
    print(f"  score en esquina inferior derecha = {nw_result.score_matrix[-1][-1]}")
    print(format_alignment(nw_result.seq1_aligned, nw_result.seq2_aligned))

    print("\nNW con dos filas:")
    print(f"  ultima fila = {last_row}")
    print(f"  score final = {two_row_score}")

    print("\nHirschberg:")
    print(f"  score final = {hb_result.score}")
    print(f"  score_matrix guardada = {hb_result.score_matrix}")
    print(format_alignment(hb_result.seq1_aligned, hb_result.seq2_aligned))

    print("\nVerificacion:")
    print(f"  score NW == score dos filas == score Hirschberg ? {nw_result.score == two_row_score == hb_result.score}")
    print(f"  alineamiento identico? {nw_result.seq1_aligned == hb_result.seq1_aligned and nw_result.seq2_aligned == hb_result.seq2_aligned}")
    print("  Nota: si el score coincide pero el alineamiento no, puede seguir estando bien;")
    print("        a veces hay varios alineamientos optimos con el mismo score.")

    explain_memory_columns(len(seq1), len(seq2))


def explain_benchmark_zeroes() -> None:
    header("Por que Hirschberg puede salir con delta de memoria 0.00 MiB")
    print(
        "En benchmark.csv, delta_mem_mib no es la memoria teorica del algoritmo.\n"
        "Es: memoria pico del proceso Python durante la corrida menos el baseline\n"
        "del proceso antes de correr la funcion.\n"
    )
    print(
        "Para NW clasico, la matriz completa crece como O(m*n), entonces el pico\n"
        "sube mucho y se ve claramente. Para Hirschberg, la memoria extra crece\n"
        "como O(min(m,n)); normalmente son listas pequenas comparadas con lo que\n"
        "Python ya tenia reservado. Por eso memory_profiler puede medir el mismo\n"
        "pico que el baseline, o una diferencia menor al redondeo, y la tabla\n"
        "termina mostrando 0.00 MiB.\n"
    )
    print(
        "Interpretacion correcta: la columna representa memoria adicional observada\n"
        "del proceso, no 'numero de bytes exactos del algoritmo'. El 0.00 no dice\n"
        "que Hirschberg use cero memoria; dice que el aumento no fue visible para\n"
        "esa medicion."
    )


def main() -> None:
    debug_pair(
        title="Ejemplo ADN pequeno con SIMPLE",
        seq1="GATTACA",
        seq2="GCATGCA",
        matrix=SimpleMatrix(match=1, mismatch=-1),
        gap=-1,
        alphabet="ACGT",
    )

    debug_pair(
        title="Ejemplo proteico pequeno con BLOSUM62",
        seq1="MTEYK",
        seq2="MTEFK",
        matrix=BiopythonMatrix("BLOSUM62"),
        gap=-4,
        alphabet="EFKMTY",
    )

    explain_benchmark_zeroes()


if __name__ == "__main__":
    main()
