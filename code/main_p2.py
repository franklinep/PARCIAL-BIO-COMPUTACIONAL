"""
Driver del Proyecto 2: filogenia de citocromo C a partir de NW.

Pasos:
  1) Carga 6 secuencias proteicas de citocromo C (Humano, Chimpance, Gorila,
     Raton, Gallina, Atun).
  2) Construye matriz de distancias 6x6 (d = 100 - %identidad) con NW+BLOSUM62.
  3) Dibuja dendrograma (UPGMA y vecino mas cercano) con scipy.
  4) Compara con Bio.Align.PairwiseAligner como referencia "BLAST-like".
  5) Guarda matriz CSV, tabla LaTeX y figura PNG en output/.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Bio import SeqIO
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from parcial.phylogeny import (
    Species, DistanceMatrixBuilder,
    matrix_to_csv, matrix_to_latex,
)
from parcial.blast_compare import compare_with_biopython


DATASET = ROOT / "dataset"
OUTPUT = ROOT / "output"
FIGURES = ROOT / "figures"
OUTPUT.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


SPECIES_FILES = [
    ("Humano",    "P99999",       "cytc_human.fasta"),
    ("Chimpance", "P99998",       "cytc_chimp.fasta"),
    ("Gorila",    "G4XXM2",       "cytc_gorilla.fasta"),
    ("Raton",     "P62897",       "cytc_mouse.fasta"),
    ("Gallina",   "P67881",       "cytc_chicken.fasta"),
    ("Atun",      "P00025",       "cytc_tuna.fasta"),
]


def load_species() -> list[Species]:
    species = []
    for label, acc, fname in SPECIES_FILES:
        record = SeqIO.read(DATASET / fname, "fasta")
        species.append(Species(label=label, accession=acc, sequence=str(record.seq)))
    return species


def print_header(text: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {text}")
    print("=" * 72)


def main() -> None:
    print_header("PROYECTO 2 - Filogenia del citocromo C (6 especies)")

    species = load_species()
    labels = [s.label for s in species]
    print("\nSecuencias cargadas:")
    for s in species:
        print(f"  {s.label:<10}  {s.accession:<10}  {len(s.sequence):>3} aa")

    # ---- (1) Matriz de distancias 6x6 con NW + BLOSUM62 ----
    print_header("Construyendo matriz de distancias (NW + BLOSUM62, gap=-4)")
    builder = DistanceMatrixBuilder(matrix_name="BLOSUM62", gap_penalty=-4)
    D, I = builder.build(species)

    print("\nMatriz de IDENTIDAD (%):")
    print(_pretty(I, labels))
    print("\nMatriz de DISTANCIA (100 - identidad):")
    print(_pretty(D, labels))

    # Persistencia
    matrix_to_csv(I, labels, OUTPUT / "identity_matrix.csv")
    matrix_to_csv(D, labels, OUTPUT / "distance_matrix.csv")
    (OUTPUT / "distance_matrix.tex").write_text(
        matrix_to_latex(
            D, labels,
            "Matriz de distancias (\\%) entre las 6 especies. "
            "$d_{ij} = 100 - \\text{identidad}_{ij}$ del alineamiento global "
            "Needleman--Wunsch con BLOSUM62."
        )
    )
    (OUTPUT / "identity_matrix.tex").write_text(
        matrix_to_latex(I, labels, "Matriz de identidad (\\%) entre las 6 especies.")
    )
    print(f"\n[guardado] {OUTPUT / 'distance_matrix.csv'}")
    print(f"[guardado] {OUTPUT / 'distance_matrix.tex'}")

    # ---- (2) Dendrogramas: UPGMA (average) y vecino mas cercano (single) ----
    print_header("Construyendo dendrogramas")
    cond = squareform(D, checks=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, method, title in [
        (axes[0], "average", "UPGMA (average linkage)"),
        (axes[1], "single",  "Vecino mas cercano (single linkage)"),
    ]:
        Z = linkage(cond, method=method)
        dendrogram(Z, labels=labels, ax=ax, leaf_font_size=11)
        ax.set_title(title)
        ax.set_ylabel("Distancia (100 - %id)")
    fig.suptitle("Arbol filogenetico - Citocromo C (NW + BLOSUM62)", fontsize=12)
    fig.tight_layout()
    out_png = FIGURES / "dendrogram.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[guardado] {out_png}")

    # ---- (3) Comparacion con Bio.Align.PairwiseAligner ----
    print_header("Validacion contra Bio.Align.PairwiseAligner (BLAST-like)")
    print("\nPar                       NW score    NW %id    Bio score   Bio %id")
    print("-" * 72)
    rows = []
    for i in range(len(species)):
        for j in range(i + 1, len(species)):
            res = builder.pair_results()[
                tuple(sorted((species[i].label, species[j].label)))
            ]
            cmp = compare_with_biopython(
                pair_label=f"{species[i].label} vs {species[j].label}",
                seq_a=species[i].sequence,
                seq_b=species[j].sequence,
                matrix_name="BLOSUM62",
                gap_open=-4,
                gap_extend=-4,
                nw_score=res.score,
                nw_identity=res.identity,
            )
            rows.append(cmp)
            print(
                f"{cmp.pair:<25} {cmp.nw_score:>9.0f} {cmp.nw_identity:>8.1f}%"
                f"  {cmp.bio_score:>9.1f} {cmp.bio_identity:>8.1f}%"
            )

    # Guardar tabla de comparacion como LaTeX
    tex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{Validacion de nuestro NW contra Bio.Align.PairwiseAligner (BLOSUM62, gap lineal $-4$).}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Par & Score NW & \%id NW & Score Bio & \%id Bio \\",
        r"\midrule",
    ]
    for r in rows:
        tex_lines.append(
            f"{r.pair} & {r.nw_score:.0f} & {r.nw_identity:.1f} & "
            f"{r.bio_score:.1f} & {r.bio_identity:.1f} \\\\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (OUTPUT / "blast_comparison.tex").write_text("\n".join(tex_lines))
    print(f"\n[guardado] {OUTPUT / 'blast_comparison.tex'}")

    # ---- (4) Mostrar un alineamiento ejemplo: humano vs raton ----
    print_header("Ejemplo de alineamiento: Humano vs Raton")
    res = builder.pair_results()[("Humano", "Raton")]
    _print_alignment(res)


def _pretty(M, labels) -> str:
    n = len(labels)
    head = "        " + "".join(f"{l:>10}" for l in labels)
    rows = [head]
    for i in range(n):
        row = f"{labels[i]:<8}" + "".join(f"{M[i, j]:>10.2f}" for j in range(n))
        rows.append(row)
    return "\n".join(rows)


def _print_alignment(r) -> None:
    a, b = r.seq1_aligned, r.seq2_aligned
    match = "".join("|" if x == y and x != "-" else (" " if "-" in (x, y) else ".")
                    for x, y in zip(a, b))
    print(f"  Score      : {r.score}")
    print(f"  Identidad  : {r.identity:.1f}%")
    print(f"  Longitud   : {r.length}")
    for start in range(0, r.length, 60):
        end = start + 60
        print(f"  {r.seq1_name:<10} {start+1:>4}  {a[start:end]}")
        print(f"  {'':<10} {'':>4}  {match[start:end]}")
        print(f"  {r.seq2_name:<10} {start+1:>4}  {b[start:end]}")
        print()


if __name__ == "__main__":
    main()
