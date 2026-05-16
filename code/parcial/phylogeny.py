"""
Proyecto 2: Arbol filogenetico a partir de alineamientos NW.

Pipeline:
    1. Calcular el alineamiento global por pares (NW) con BLOSUM62.
    2. Convertir cada alineamiento en una distancia: d = 100 - %identidad.
    3. Construir un dendrograma jerarquico (vecino mas cercano / UPGMA / WPGMA).

Solo usa scipy.cluster.hierarchy para *dibujar*, segun la consigna.
La matriz de distancias y los scores los calcula NeedlemanWunsch propio.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from alignment import BiopythonMatrix, NeedlemanWunsch
from alignment.result import AlignmentResult


@dataclass
class Species:
    """Una especie con su etiqueta corta y su secuencia."""
    label: str           # "Humano", "Raton", ...
    accession: str       # P99999, P62897, ...
    sequence: str


class DistanceMatrixBuilder:
    """
    Construye la matriz simetrica de distancias entre N especies
    usando NW con la matriz de sustitucion indicada.

    Distancia entre dos secuencias = 100 - identidad(%) del alineamiento NW.
    Una matriz de distancias requiere d(i,i)=0 y d(i,j)=d(j,i); ambas
    propiedades se cumplen por construccion.
    """

    def __init__(self, matrix_name: str = "BLOSUM62", gap_penalty: int = -4):
        self.matrix = BiopythonMatrix(matrix_name)
        self.gap_penalty = gap_penalty
        self.aligner = NeedlemanWunsch(self.matrix, gap_penalty=gap_penalty)
        # Cache para no recalcular alineamientos
        self._pair_results: Dict[Tuple[str, str], AlignmentResult] = {}

    def align_pair(self, a: Species, b: Species) -> AlignmentResult:
        key = tuple(sorted((a.label, b.label)))
        if key not in self._pair_results:
            self._pair_results[key] = self.aligner.align(
                a.sequence, b.sequence, a.label, b.label
            )
        return self._pair_results[key]

    def build(
        self,
        species: List[Species],
        external_identities: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Devuelve (matriz_distancias, matriz_identidades) NxN.

        Si `external_identities` esta dado, para cada par (label_i, label_j)
        ordenado alfabeticamente que aparezca en el dict se usa ese valor
        (porcentaje 0-100) en vez de calcular NW. Los pares ausentes caen
        de vuelta al alineamiento NW propio. Esto permite construir una
        segunda matriz con identidades de BLAST sin duplicar logica."""
        n = len(species)
        D = np.zeros((n, n))
        I = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                key = tuple(sorted((species[i].label, species[j].label)))
                if external_identities is not None and key in external_identities:
                    identity = external_identities[key]
                else:
                    result = self.align_pair(species[i], species[j])
                    identity = result.identity
                D[i, j] = D[j, i] = 100.0 - identity
                I[i, j] = I[j, i] = identity
            I[i, i] = 100.0
        return D, I

    def pair_results(self) -> Dict[Tuple[str, str], AlignmentResult]:
        return self._pair_results


def matrix_to_csv(matrix: np.ndarray, labels: List[str], path: Path) -> None:
    """Persiste una matriz simetrica con cabecera de etiquetas."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("," + ",".join(labels) + "\n")
        for i, lbl in enumerate(labels):
            row = ",".join(f"{matrix[i, j]:.2f}" for j in range(len(labels)))
            f.write(f"{lbl},{row}\n")


def matrix_to_latex(matrix: np.ndarray, labels: List[str], caption: str) -> str:
    """Genera una tabla LaTeX (tabular) con la matriz simetrica."""
    n = len(labels)
    col_spec = "l" + "r" * n
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & " + " & ".join(labels) + r" \\",
        r"\midrule",
    ]
    for i, lbl in enumerate(labels):
        row = " & ".join(f"{matrix[i, j]:.1f}" for j in range(n))
        lines.append(f"{lbl} & {row} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)
