"""
Needleman-Wunsch en espacio O(min(n, m)) -- "tecnica de las dos filas".

Solo devuelve el SCORE final y la ULTIMA FILA de la matriz; no permite
reconstruir el alineamiento por si solo, pero es la primitiva basica
sobre la que se construye Hirschberg.

Idea: la celda H[i,j] solo depende de H[i-1, *] y H[i, j-1], asi que
basta con mantener dos filas (la anterior y la actual). Memoria O(n+1).

Si deseamos el menor uso de memoria posible elegimos como "filas" la
dimension mas corta (longitud min(m, n)).
"""
from __future__ import annotations
from typing import List

from alignment.matrices import SubstitutionMatrix


def nw_score_last_row(
    seq1: str,
    seq2: str,
    matrix: SubstitutionMatrix,
    gap: int,
) -> List[int]:
    """
    Devuelve solo la ULTIMA fila H[m, *] de la matriz Needleman-Wunsch.
    Memoria: 2 * (n+1) enteros = O(n).

    Esta funcion es la primitiva que necesita Hirschberg para localizar
    el punto de cruce del alineamiento optimo.
    """
    m, n = len(seq1), len(seq2)
    prev = [j * gap for j in range(n + 1)]
    cur = [0] * (n + 1)
    for i in range(1, m + 1):
        cur[0] = i * gap
        si = seq1[i - 1]
        for j in range(1, n + 1):
            diag = prev[j - 1] + matrix.score(si, seq2[j - 1])
            up   = prev[j]     + gap
            left = cur[j - 1]  + gap
            cur[j] = max(diag, up, left)
        prev, cur = cur, prev
    return prev  # tras el swap, prev contiene la fila final


def nw_score_only(
    seq1: str,
    seq2: str,
    matrix: SubstitutionMatrix,
    gap: int,
) -> int:
    """Score global NW en espacio O(min(m, n))."""
    # Elegir la secuencia mas corta como "columna" para minimizar memoria
    if len(seq2) > len(seq1):
        seq1, seq2 = seq2, seq1
    return nw_score_last_row(seq1, seq2, matrix, gap)[-1]
