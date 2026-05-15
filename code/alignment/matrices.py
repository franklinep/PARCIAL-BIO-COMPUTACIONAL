from __future__ import annotations
from abc import ABC, abstractmethod
from Bio.Align import substitution_matrices


class SubstitutionMatrix(ABC):
    """Interfaz comun para cualquier sistema de puntuacion."""

    @abstractmethod
    def score(self, a: str, b: str) -> int:
        """Devuelve la puntuacion para emparejar los simbolos a y b."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre legible (ej. SIMPLE, PAM250, BLOSUM62)."""


class SimpleMatrix(SubstitutionMatrix):
    """Tabla simple: match=+1, mismatch=-1. Util para ADN."""

    def __init__(self, match: int = 1, mismatch: int = -1):
        self.match = match
        self.mismatch = mismatch

    def score(self, a: str, b: str) -> int:
        return self.match if a == b else self.mismatch

    @property
    def name(self) -> str:
        return "SIMPLE"


class BiopythonMatrix(SubstitutionMatrix):
    """
    Carga matrices estandar PAM/BLOSUM via Bio.Align.substitution_matrices.

    Matrices disponibles: PAM30, PAM70, PAM120, PAM250,
                          BLOSUM45, BLOSUM62, BLOSUM80, BLOSUM90...
    """

    def __init__(self, matrix_name: str):
        self._name = matrix_name.upper()
        self._matrix = substitution_matrices.load(self._name)

    def score(self, a: str, b: str) -> int:
        try:
            return int(self._matrix[a, b])
        except KeyError:
            # Aminoacidos no estandar (X, B, Z): penalizacion neutra
            return -1

    @property
    def name(self) -> str:
        return self._name
