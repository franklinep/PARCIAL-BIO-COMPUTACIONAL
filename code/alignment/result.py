from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class AlignmentResult:
    """Encapsula la salida de un alineamiento (NW o SW)."""

    algorithm: str           # "NW" o "SW"
    matrix_name: str         # "SIMPLE", "PAM250", etc.
    gap_penalty: int
    seq1_name: str
    seq2_name: str
    seq1_aligned: str
    seq2_aligned: str
    score: int
    score_matrix: List[List[int]] = field(repr=False)

    @property
    def length(self) -> int:
        return len(self.seq1_aligned)

    @property
    def matches(self) -> int:
        return sum(
            1 for a, b in zip(self.seq1_aligned, self.seq2_aligned)
            if a == b and a != "-"
        )

    @property
    def mismatches(self) -> int:
        return sum(
            1 for a, b in zip(self.seq1_aligned, self.seq2_aligned)
            if a != b and a != "-" and b != "-"
        )

    @property
    def gaps(self) -> int:
        return self.seq1_aligned.count("-") + self.seq2_aligned.count("-")

    @property
    def identity(self) -> float:
        return (self.matches / self.length * 100) if self.length else 0.0
