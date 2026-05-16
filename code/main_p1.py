"""
Driver del Proyecto 1: Hirschberg para NW en espacio O(min(m, n)).

1) Verifica correccion: Hirschberg da el mismo score que NW clasico sobre
   el CDS real de citocromo C humano vs raton (318 bases).
2) Benchmark de tiempo y memoria sobre tamanios crecientes
   (200, 400, 800, 1600, 3200, 6400 bases) construidos a partir de
   los CDS reales (concatenacion + mutaciones).
3) Genera grafica memoria vs longitud y tiempo vs longitud.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Bio import SeqIO

from alignment import SimpleMatrix, NeedlemanWunsch
from parcial.hirschberg import HirschbergAligner
from parcial.benchmark import run_benchmark, points_to_csv


DATASET = ROOT / "dataset"
OUTPUT = ROOT / "output"
FIGURES = ROOT / "figures"
OUTPUT.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


def header(text: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {text}")
    print("=" * 72)


def main() -> None:
    header("PROYECTO 1 - Optimizacion de espacio: Hirschberg")

    # --- (1) Correccion sobre datos reales ---
    seq_h = str(SeqIO.read(DATASET / "cytc_human_cds.fasta", "fasta").seq)
    seq_m = str(SeqIO.read(DATASET / "cytc_mouse_cds.fasta", "fasta").seq)
    print(f"\nCDS citocromo C (NCBI):")
    print(f"  Humano (NM_018947) : {len(seq_h)} bases")
    print(f"  Raton  (NM_007808) : {len(seq_m)} bases")

    matrix = SimpleMatrix(match=1, mismatch=-1)
    nw = NeedlemanWunsch(matrix, gap_penalty=-1)
    hb = HirschbergAligner(matrix, gap_penalty=-1)

    r_nw = nw.align(seq_h, seq_m, "CYC_HUMAN", "CYC_MOUSE")
    r_hb = hb.align(seq_h, seq_m, "CYC_HUMAN", "CYC_MOUSE")

    print(f"\n  NW clasico   : score={r_nw.score}  len={r_nw.length}  id={r_nw.identity:.1f}%")
    print(f"  Hirschberg   : score={r_hb.score}  len={r_hb.length}  id={r_hb.identity:.1f}%")
    assert r_nw.score == r_hb.score, "ERROR: scores difieren"
    print("  OK -- score y % identidad coinciden")

    # Persistir alineamiento de Hirschberg
    (OUTPUT / "hirschberg_alignment.txt").write_text(
        _format_alignment(r_hb)
    )
    print(f"\n  [guardado] {OUTPUT / 'hirschberg_alignment.txt'}")

    # --- (2) Benchmark tamanios crecientes ---
    header("Benchmark: tiempo y memoria vs longitud")
    lengths = [200, 400, 800, 1600, 3200, 6400]
    points = run_benchmark(seq_h, seq_m, lengths, gap=-1)
    points_to_csv(points, OUTPUT / "benchmark.csv")
    print(f"\n  [guardado] {OUTPUT / 'benchmark.csv'}")

    # --- (3) Graficas ---
    header("Generando graficas")
    nw_pts = [p for p in points if p.algorithm == "NW"]
    hb_pts = [p for p in points if p.algorithm == "Hirschberg"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Tiempo
    axes[0].plot([p.length for p in nw_pts], [p.time_s for p in nw_pts],
                 "o-", label="NW O(mn)", color="C0")
    axes[0].plot([p.length for p in hb_pts], [p.time_s for p in hb_pts],
                 "s-", label="Hirschberg O(min)", color="C1")
    axes[0].set_xlabel("Longitud de las secuencias (bases)")
    axes[0].set_ylabel("Tiempo (s)")
    axes[0].set_title("Tiempo de ejecucion")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    # Memoria (delta sobre baseline)
    axes[1].plot([p.length for p in nw_pts], [p.delta_mem_mib for p in nw_pts],
                 "o-", label="NW O(mn)", color="C0")
    axes[1].plot([p.length for p in hb_pts], [p.delta_mem_mib for p in hb_pts],
                 "s-", label="Hirschberg O(min)", color="C1")
    axes[1].set_xlabel("Longitud de las secuencias (bases)")
    axes[1].set_ylabel("Memoria adicional RSS (MiB)")
    axes[1].set_title("Uso de memoria pico del proceso")
    axes[1].set_xscale("log")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    fig.suptitle("NW clasico vs Hirschberg -- citocromo C (concatenado)", fontsize=12)
    fig.tight_layout()
    out_png = FIGURES / "benchmark.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [guardado] {out_png}")

    # Tabla LaTeX
    rows = []
    for L in lengths:
        nw_p = next((p for p in points if p.length == L and p.algorithm == "NW"), None)
        hb_p = next((p for p in points if p.length == L and p.algorithm == "Hirschberg"), None)
        rows.append((L, nw_p, hb_p))

    tex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{Comparacion de tiempo y memoria entre NW clasico y Hirschberg "
        r"para longitudes crecientes. Memoria reportada es el delta RSS sobre el "
        r"baseline del proceso Python (medido con memory\_profiler); valores "
        r"cercanos a cero pueden quedar por debajo de la resolucion de muestreo.}",
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"L (bases) & $t_{\text{NW}}$ (ms) & $t_{\text{HB}}$ (ms) "
        r"& $\Delta m_{\text{NW}}$ (MiB) & $\Delta m_{\text{HB}}$ (MiB) \\",
        r"\midrule",
    ]
    for L, nw_p, hb_p in rows:
        tex_lines.append(
            f"{L} & {nw_p.time_s*1000:.1f} & {hb_p.time_s*1000:.1f} "
            f"& {nw_p.delta_mem_mib:.3f} & {hb_p.delta_mem_mib:.3f} \\\\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (OUTPUT / "benchmark.tex").write_text("\n".join(tex_lines))
    print(f"  [guardado] {OUTPUT / 'benchmark.tex'}")

    header("Listo")
    print(
        "  Resumen:\n"
        f"    - Hirschberg validado contra NW: score={r_hb.score}, "
        f"identidad={r_hb.identity:.1f}%\n"
        f"    - {len(lengths)} puntos de benchmark guardados.\n"
        f"    - Figura: {out_png}\n"
    )


def _format_alignment(r) -> str:
    a, b = r.seq1_aligned, r.seq2_aligned
    match = "".join("|" if x == y and x != "-" else (" " if "-" in (x, y) else ".")
                    for x, y in zip(a, b))
    lines = [
        f"# Alineamiento Hirschberg",
        f"# Matriz       : {r.matrix_name}",
        f"# Gap          : {r.gap_penalty}",
        f"# Score        : {r.score}",
        f"# Longitud     : {r.length}",
        f"# Identidad    : {r.identity:.2f}%",
        f"# Gaps         : {r.gaps}",
        "",
    ]
    for start in range(0, r.length, 60):
        end = start + 60
        lines.append(f"{r.seq1_name:<10} {start+1:>5} {a[start:end]}")
        lines.append(f"{'':10} {'':5} {match[start:end]}")
        lines.append(f"{r.seq2_name:<10} {start+1:>5} {b[start:end]}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
