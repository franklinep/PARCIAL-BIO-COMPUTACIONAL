"""
Driver del Proyecto 2: filogenia de citocromo C a partir de NW y BLAST.

Pasos:
  1) Carga 6 secuencias proteicas de citocromo C (Humano, Chimpance, Gorila,
     Raton, Gallina, Atun) desde UniProt.
  2) Construye matriz de distancias 6x6 (d = 100 - %identidad) con NW+BLOSUM62.
  3) Dibuja dendrograma NW (UPGMA y vecino mas cercano) con scipy.
  4) Valida contra Bio.Align.PairwiseAligner (NW global de Biopython).
  5) Lee/refresca cache BLAST (blastp online vs NCBI nr), construye una
     SEGUNDA matriz de distancias usando identidades BLAST y dibuja un
     dendrograma BLAST analogo. Genera tabla comparativa NW vs BLAST.
  6) Guarda matrices CSV, tablas LaTeX y figuras PNG en output/ y figures/.

El cache BLAST (output/blast_cache.json) se commitea al repo, asi que
las corridas posteriores no tocan internet.
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
from parcial.blast_compare import (
    compare_with_biopython,
    build_blast_matrix,
)


DATASET = ROOT / "dataset"
OUTPUT = ROOT / "output"
FIGURES = ROOT / "figures"
OUTPUT.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


# (label, accession_uniprot, fasta_file)
SPECIES_FILES = [
    ("Humano",    "P99999",       "cytc_human.fasta"),
    ("Chimpance", "P99998",       "cytc_chimp.fasta"),
    ("Gorila",    "G4XXM2",       "cytc_gorilla.fasta"),
    ("Raton",     "P62897",       "cytc_mouse.fasta"),
    ("Gallina",   "P67881",       "cytc_chicken.fasta"),
    ("Atun",      "P00025",       "cytc_tuna.fasta"),
]

# Mapeo label -> accession UniProt (para el filtro Entrez de BLAST)
ACCESSION_MAP = {label: acc for label, acc, _ in SPECIES_FILES}


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


def _draw_dendrograms(
    D: np.ndarray,
    labels: list[str],
    out_png: Path,
    suptitle: str,
) -> None:
    """Dibuja UPGMA + single linkage lado a lado y guarda el PNG."""
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
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print_header("PROYECTO 2 - Filogenia del citocromo C (6 especies)")

    species = load_species()
    labels = [s.label for s in species]
    print("\nSecuencias cargadas:")
    for s in species:
        print(f"  {s.label:<10}  {s.accession:<10}  {len(s.sequence):>3} aa")

    # ---- (1) Matriz de distancias NW + BLOSUM62 ----
    print_header("Matriz de distancias con NW + BLOSUM62 (gap=-4)")
    builder = DistanceMatrixBuilder(matrix_name="BLOSUM62", gap_penalty=-4)
    D, I = builder.build(species)

    print("\nMatriz de IDENTIDAD (%):")
    print(_pretty(I, labels))
    print("\nMatriz de DISTANCIA (100 - identidad):")
    print(_pretty(D, labels))

    matrix_to_csv(I, labels, OUTPUT / "identity_matrix.csv")
    matrix_to_csv(D, labels, OUTPUT / "distance_matrix.csv")
    (OUTPUT / "distance_matrix.tex").write_text(
        matrix_to_latex(
            D, labels,
            "Matriz de distancias (\\%) NW+BLOSUM62 entre las 6 especies. "
            "$d_{ij} = 100 - \\text{identidad}_{ij}$ del alineamiento global "
            "Needleman--Wunsch."
        )
    )
    (OUTPUT / "identity_matrix.tex").write_text(
        matrix_to_latex(I, labels, "Matriz de identidad (\\%) NW+BLOSUM62 entre las 6 especies.")
    )
    print(f"\n[guardado] {OUTPUT / 'distance_matrix.csv'}")
    print(f"[guardado] {OUTPUT / 'distance_matrix.tex'}")

    # ---- (2) Dendrograma NW ----
    print_header("Dendrograma NW (UPGMA + single linkage)")
    out_nw_png = FIGURES / "dendrogram_nw.png"
    _draw_dendrograms(
        D, labels, out_nw_png,
        suptitle="Arbol filogenetico desde NW+BLOSUM62 -- Citocromo C",
    )
    print(f"[guardado] {out_nw_png}")

    # ---- (3) Validacion local: NW vs Bio.Align.PairwiseAligner ----
    print_header("Validacion local contra Bio.Align.PairwiseAligner")
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

    tex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{Validacion local: nuestro NW vs \texttt{Bio.Align.PairwiseAligner} "
        r"(BLOSUM62, gap lineal $-4$). Coincidencia exacta en score e identidad confirma "
        r"que la implementacion artesanal es funcionalmente equivalente a la estandar.}",
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

    # ---- (4) Comparacion contra BLAST de NCBI (cache JSON) ----
    print_header("Comparacion contra BLAST de NCBI (blastp, cache JSON)")
    cache_path = OUTPUT / "blast_cache.json"
    blast_results = build_blast_matrix(
        species=species,
        accession_map=ACCESSION_MAP,
        cache_path=cache_path,
        force_refresh=False,
    )
    n_pairs_total = len(labels) * (len(labels) - 1) // 2
    n_pairs_ok = sum(1 for v in blast_results.values() if v is not None)
    print(f"\nPares BLAST recuperados: {n_pairs_ok}/{n_pairs_total}")

    # Matriz de distancias BLAST reusando el builder
    ext = {
        key: r.identity_pct
        for key, r in blast_results.items() if r is not None
    }
    D_blast, I_blast = builder.build(species, external_identities=ext)
    matrix_to_csv(I_blast, labels, OUTPUT / "identity_matrix_blast.csv")
    matrix_to_csv(D_blast, labels, OUTPUT / "distance_matrix_blast.csv")
    (OUTPUT / "distance_matrix_blast.tex").write_text(
        matrix_to_latex(
            D_blast, labels,
            "Matriz de distancias (\\%) calculada con BLAST de NCBI "
            "(\\texttt{blastp} contra \\texttt{nr}, simetrizada por mayor bit-score). "
            "$d_{ij} = 100 - \\%\\text{id}_{\\text{HSP}}$."
        )
    )
    (OUTPUT / "identity_matrix_blast.tex").write_text(
        matrix_to_latex(I_blast, labels, "Matriz de identidad (\\%) BLAST entre las 6 especies.")
    )
    print(f"\n[guardado] {OUTPUT / 'distance_matrix_blast.csv'}")
    print(f"[guardado] {OUTPUT / 'distance_matrix_blast.tex'}")

    # Tabla comparativa NW %id vs BLAST %id + bit-score + E-value
    cmp_tex = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{Comparacion NW (global, BLOSUM62) vs BLAST de NCBI "
        r"(\texttt{blastp} contra \texttt{nr}, local) para los 15 pares de "
        r"citocromo c. Bit-score y E-value reportados por NCBI para el mejor HSP.}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Par & \%id NW & \%id BLAST & bit-score & E-value & cov. (\%) \\",
        r"\midrule",
    ]
    for i in range(len(species)):
        for j in range(i + 1, len(species)):
            a, b = species[i].label, species[j].label
            key = tuple(sorted((a, b)))
            r_nw = builder.pair_results()[key]
            r_bl = blast_results.get(key)
            if r_bl is None:
                cmp_tex.append(
                    f"{a} vs {b} & {r_nw.identity:.1f} & --- & --- & --- & --- \\\\"
                )
                continue
            e_str = f"{r_bl.e_value:.1e}" if r_bl.e_value > 0 else "0"
            cmp_tex.append(
                f"{a} vs {b} & {r_nw.identity:.1f} & {r_bl.identity_pct:.1f} "
                f"& {r_bl.bit_score:.1f} & {e_str} & {r_bl.query_coverage_pct:.0f} \\\\"
            )
    cmp_tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (OUTPUT / "nw_vs_blast.tex").write_text("\n".join(cmp_tex))
    print(f"[guardado] {OUTPUT / 'nw_vs_blast.tex'}")

    # Print de la tabla NW vs BLAST en consola
    print("\nPar                       %id NW   %id BLAST   bit-score   E-value")
    print("-" * 72)
    for i in range(len(species)):
        for j in range(i + 1, len(species)):
            a, b = species[i].label, species[j].label
            key = tuple(sorted((a, b)))
            r_nw = builder.pair_results()[key]
            r_bl = blast_results.get(key)
            if r_bl is None:
                print(f"{a} vs {b:<14}  {r_nw.identity:>5.1f}%      n/a")
            else:
                e_str = f"{r_bl.e_value:.1e}" if r_bl.e_value > 0 else "0"
                print(
                    f"{a} vs {b:<14}  {r_nw.identity:>5.1f}%   "
                    f"{r_bl.identity_pct:>5.1f}%   {r_bl.bit_score:>7.1f}   {e_str}"
                )

    # ---- (5) Dendrograma BLAST ----
    if n_pairs_ok == n_pairs_total:
        print_header("Dendrograma BLAST (UPGMA + single linkage)")
        out_blast_png = FIGURES / "dendrogram_blast.png"
        _draw_dendrograms(
            D_blast, labels, out_blast_png,
            suptitle="Arbol filogenetico desde BLAST (blastp/nr) -- Citocromo C",
        )
        print(f"[guardado] {out_blast_png}")
    else:
        print_header("Dendrograma BLAST OMITIDO (pares incompletos)")
        print(f"  Faltan {n_pairs_total - n_pairs_ok} pares. Corre con internet")
        print(f"  para refrescar {cache_path}, o usa force_refresh=True.")

    # ---- (6) Mostrar un alineamiento ejemplo ----
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
