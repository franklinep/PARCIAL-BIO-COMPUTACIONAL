from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple

from .matrices import SubstitutionMatrix
from .result import AlignmentResult


Matrix = List[List[int]]
Cell = Tuple[int, int]


class Aligner(ABC):
    """Esqueleto comun a NW y SW (Template Method)."""

    def __init__(self, matrix: SubstitutionMatrix, gap_penalty: int = -2):
        self.matrix = matrix
        self.gap_penalty = gap_penalty

    def align(
        self,
        seq1: str,
        seq2: str,
        seq1_name: str = "Seq1",
        seq2_name: str = "Seq2",
    ) -> AlignmentResult:
        m, n = len(seq1), len(seq2)
        score_matrix = self._initialize(m, n)
        start = self._fill(score_matrix, seq1, seq2)
        a1, a2 = self._traceback(score_matrix, seq1, seq2, start)
        score = score_matrix[start[0]][start[1]]
        return AlignmentResult(
            algorithm=self.algorithm_name,
            matrix_name=self.matrix.name,
            gap_penalty=self.gap_penalty,
            seq1_name=seq1_name,
            seq2_name=seq2_name,
            seq1_aligned=a1,
            seq2_aligned=a2,
            score=score,
            score_matrix=score_matrix,
        )

    @property
    @abstractmethod
    def algorithm_name(self) -> str: ...

    @abstractmethod
    def _initialize(self, m: int, n: int) -> Matrix: ...

    @abstractmethod
    def _fill(self, matrix: Matrix, seq1: str, seq2: str) -> Cell: ...

    @abstractmethod
    def _traceback(
        self, matrix: Matrix, seq1: str, seq2: str, start: Cell
    ) -> Tuple[str, str]: ...


class NeedlemanWunsch(Aligner):
    """Alineamiento GLOBAL: alinea las dos secuencias completas."""

    @property
    def algorithm_name(self) -> str:
        return "NW"

    def _initialize(self, m: int, n: int) -> Matrix:
        matrix = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            matrix[i][0] = i * self.gap_penalty
        for j in range(n + 1):
            matrix[0][j] = j * self.gap_penalty
        return matrix

    def _fill(self, matrix: Matrix, seq1: str, seq2: str) -> Cell:
        m, n = len(seq1), len(seq2)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                diag = matrix[i - 1][j - 1] + self.matrix.score(seq1[i - 1], seq2[j - 1])
                up   = matrix[i - 1][j]     + self.gap_penalty
                left = matrix[i][j - 1]     + self.gap_penalty
                matrix[i][j] = max(diag, up, left)
        return (m, n)  # NW siempre arranca traceback desde la esquina inferior derecha

    def _traceback(
        self, matrix: Matrix, seq1: str, seq2: str, start: Cell
    ) -> Tuple[str, str]:
        i, j = start
        a1, a2 = [], []
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                diag = matrix[i - 1][j - 1] + self.matrix.score(seq1[i - 1], seq2[j - 1])
                if matrix[i][j] == diag:
                    a1.append(seq1[i - 1]); a2.append(seq2[j - 1])
                    i -= 1; j -= 1
                    continue
            if i > 0 and matrix[i][j] == matrix[i - 1][j] + self.gap_penalty:
                a1.append(seq1[i - 1]); a2.append("-")
                i -= 1
            else:
                a1.append("-"); a2.append(seq2[j - 1])
                j -= 1
        return "".join(reversed(a1)), "".join(reversed(a2))


class SmithWaterman(Aligner):
    """Alineamiento LOCAL: encuentra la subregion mas similar."""

    @property
    def algorithm_name(self) -> str:
        return "SW"

    def _initialize(self, m: int, n: int) -> Matrix:
        # Diferencia clave 1: SW inicializa todo en cero.
        return [[0] * (n + 1) for _ in range(m + 1)]

    def _fill(self, matrix: Matrix, seq1: str, seq2: str) -> Cell:
        m, n = len(seq1), len(seq2)
        max_score, max_pos = 0, (0, 0)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                diag = matrix[i - 1][j - 1] + self.matrix.score(seq1[i - 1], seq2[j - 1])
                up   = matrix[i - 1][j]     + self.gap_penalty
                left = matrix[i][j - 1]     + self.gap_penalty
                # Diferencia clave 2: el max incluye 0 (reinicio del score).
                matrix[i][j] = max(0, diag, up, left)
                if matrix[i][j] > max_score:
                    max_score, max_pos = matrix[i][j], (i, j)
        # Diferencia clave 3: SW arranca traceback desde la celda con score maximo.
        return max_pos

    def _traceback(
        self, matrix: Matrix, seq1: str, seq2: str, start: Cell
    ) -> Tuple[str, str]:
        i, j = start
        a1, a2 = [], []
        # Diferencia clave 4: SW termina traceback al encontrar un 0.
        while i > 0 and j > 0 and matrix[i][j] != 0:
            diag = matrix[i - 1][j - 1] + self.matrix.score(seq1[i - 1], seq2[j - 1])
            if matrix[i][j] == diag:
                a1.append(seq1[i - 1]); a2.append(seq2[j - 1])
                i -= 1; j -= 1
            elif matrix[i][j] == matrix[i - 1][j] + self.gap_penalty:
                a1.append(seq1[i - 1]); a2.append("-")
                i -= 1
            else:
                a1.append("-"); a2.append(seq2[j - 1])
                j -= 1
        return "".join(reversed(a1)), "".join(reversed(a2))
