"""
Algoritmo de Hirschberg: alineamiento global Needleman-Wunsch
en O(m * n) tiempo y O(min(m, n)) espacio, reconstruyendo el alineamiento.

Idea central (Hirschberg, 1975):
  - Si una de las secuencias tiene longitud <= 1, resolver el caso base
    con NW directo (O(n) memoria, trivial).
  - En caso general, partir la primera secuencia en la mitad i = m/2.
    Calcular:
        F = NW_score(seq1[:i], seq2)         -- fila final hacia adelante
        B = NW_score(reverse(seq1[i:]),
                     reverse(seq2))           -- fila final hacia atras
    Encontrar j* = argmax_j (F[j] + B[n-j]).
    Resolver recursivamente sobre (seq1[:i], seq2[:j*]) y
    (seq1[i:], seq2[j*:]) y concatenar.
  - La recurrencia divide el trabajo a la mitad: T(m, n) = T(m/2, j*) +
    T(m/2, n-j*) + O(m*n). Esto resuelve a O(m*n) tiempo total, igual que
    NW clasico.
  - Memoria: cada llamada usa O(n) y la recursion tiene profundidad
    O(log m), pero solo necesitamos O(n) en la celda mas profunda.
"""
from __future__ import annotations
from typing import Tuple

from alignment.matrices import SubstitutionMatrix
from alignment.result import AlignmentResult
from .nw_two_rows import nw_score_last_row, nw_score_only


def _nw_base(
    seq1: str, seq2: str,
    matrix: SubstitutionMatrix, gap: int,
) -> Tuple[str, str]:
    """Caso base de Hirschberg: una de las dos secuencias tiene longitud 0 o 1.
    Aqui podemos resolver NW exacto en espacio lineal trivialmente."""
    m, n = len(seq1), len(seq2)
    if m == 0:
        return "-" * n, seq2
    if n == 0:
        return seq1, "-" * m
    if m == 1:
        # Encontrar la mejor posicion para incrustar seq1[0] en seq2
        best_j, best_score = -1, None
        # Opcion 1: emparejar seq1[0] con alguna seq2[j], gaps en el resto
        for j in range(n):
            score = (n - 1) * gap + matrix.score(seq1[0], seq2[j])
            if best_score is None or score > best_score:
                best_score, best_j = score, j
        # Opcion 2: dejar seq1[0] como gap (no emparejar)
        skip_score = n * gap + gap  # gap para seq1[0] + n gaps en seq1
        if skip_score > best_score:
            a = "-" * n + seq1[0]
            b = seq2 + "-"
            return a, b
        a = list("-" * n)
        a[best_j] = seq1[0]
        return "".join(a), seq2
    if n == 1:
        a, b = _nw_base(seq2, seq1, matrix, gap)
        return b, a
    raise ValueError("Caso base solo cuando min(m, n) <= 1")


def _hirschberg(
    seq1: str, seq2: str,
    matrix: SubstitutionMatrix, gap: int,
) -> Tuple[str, str]:
    m, n = len(seq1), len(seq2)
    if m <= 1 or n <= 1:
        return _nw_base(seq1, seq2, matrix, gap)

    mid = m // 2
    # Fila final hacia adelante: NW(seq1[:mid], seq2)
    F = nw_score_last_row(seq1[:mid], seq2, matrix, gap)
    # Fila final hacia atras: NW(reverse(seq1[mid:]), reverse(seq2))
    B = nw_score_last_row(seq1[mid:][::-1], seq2[::-1], matrix, gap)

    # Encontrar la posicion de cruce optima
    best_j, best_total = 0, F[0] + B[n]
    for j in range(1, n + 1):
        total = F[j] + B[n - j]
        if total > best_total:
            best_total, best_j = total, j

    left_a,  left_b  = _hirschberg(seq1[:mid],  seq2[:best_j], matrix, gap)
    right_a, right_b = _hirschberg(seq1[mid:],  seq2[best_j:], matrix, gap)
    return left_a + right_a, left_b + right_b


class HirschbergAligner:
    """
    Alineador Needleman-Wunsch en espacio O(min(m, n)).

    Misma interfaz publica que NeedlemanWunsch (`align()`), pero internamente
    no construye la matriz score completa, solo dos filas y la recursion
    divide-and-conquer.
    """

    def __init__(self, matrix: SubstitutionMatrix, gap_penalty: int = -2):
        self.matrix = matrix
        self.gap_penalty = gap_penalty

    @property
    def algorithm_name(self) -> str:
        return "NW-Hirschberg"

    def align(
        self, seq1: str, seq2: str,
        seq1_name: str = "Seq1", seq2_name: str = "Seq2",
    ) -> AlignmentResult:
        # Para asegurar O(min(m, n)) espacio, conviene que la dimension
        # "fila" sea la mas corta. Hirschberg ya es correcto independiente
        # del orden, pero el consumo de RAM es minimo si len(seq2) <= len(seq1).
        swapped = False
        if len(seq2) > len(seq1):
            seq1, seq2 = seq2, seq1
            seq1_name, seq2_name = seq2_name, seq1_name
            swapped = True

        a, b = _hirschberg(seq1, seq2, self.matrix, self.gap_penalty)
        score = nw_score_only(seq1, seq2, self.matrix, self.gap_penalty)

        if swapped:
            a, b = b, a
            seq1_name, seq2_name = seq2_name, seq1_name

        return AlignmentResult(
            algorithm="NW-Hirschberg",
            matrix_name=self.matrix.name,
            gap_penalty=self.gap_penalty,
            seq1_name=seq1_name,
            seq2_name=seq2_name,
            seq1_aligned=a,
            seq2_aligned=b,
            score=score,
            score_matrix=[],  # Hirschberg NO almacena la matriz completa
        )
