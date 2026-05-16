"""
Parser de los archivos BLAST en texto plano descargados desde la web
de NCBI (salidas-BLAST/{a,b,C,d,e}/*.txt).

Extrae para cada archivo:
  - query y subject accession (UniProt) deducidos de los headers `Query:`
    y `Subject:`
  - bit_score y e_value de la linea `Score:NNN bits(...), Expect:E,`
  - identidades y align_length de `Identities:M/L(P%)`
  - query coverage de la fila de la tabla "Clusters producing significant
    alignments" (columna Query cover)

Genera `code/output/blast_cache.json` con una entrada *direccional* por
archivo. El driver del Proyecto 2 (build_blast_matrix) simetriza despues
si hay dos sentidos del mismo par; aqui solo hay un sentido por par y
eso es suficiente.

Uso:
    python parse_blast_txt.py
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from parcial.blast_compare import BlastPairResult


SALIDAS = ROOT.parent / "salidas-BLAST"
OUTPUT = ROOT / "output" / "blast_cache.json"

# UniProt accession -> etiqueta usada en main_p2.py
ACC_TO_LABEL = {
    "P99999": "Humano",
    "P99998": "Chimpance",
    "G4XXM2": "Gorila",
    "P62897": "Raton",
    "P67881": "Gallina",
    "P00025": "Atun",
}


# Regex compiladas una vez
_RE_QUERY = re.compile(r"^Query:\s*(\S+)", re.MULTILINE)
_RE_SUBJECT = re.compile(r"^Subject:\s*(\S+)", re.MULTILINE)
_RE_ACCESSION = re.compile(r"(?:sp|tr)\|([A-Z0-9]+)\|")
_RE_SCORE = re.compile(
    r"Score:\s*([0-9.]+)\s*bits\([0-9]+\)\s*,\s*Expect:\s*([0-9.eE+\-]+)"
)
_RE_IDENT = re.compile(r"Identities:\s*(\d+)\s*/\s*(\d+)\s*\(\s*([0-9.]+)%\)")
# Para query coverage: la linea del cluster tiene formato variable; la
# extraemos buscando el primer porcentaje de cobertura que aparece despues
# del "Total Score" en la linea del cluster. Mas robusto: buscar el patron
# "<num>%<whitespace><e-value>" cerca del comienzo, donde el porcentaje es
# el query cover y el e-value es la columna E.
_RE_QCOVER = re.compile(r"\s(\d+)%\s+[0-9.eE+\-]+\s+\d+\.\d+\s")


def _accession_from_header(header: str) -> str:
    m = _RE_ACCESSION.search(header)
    if not m:
        raise ValueError(f"No se pudo extraer accession de: {header!r}")
    return m.group(1)


def parse_blast_txt(path: Path) -> BlastPairResult:
    text = path.read_text(encoding="utf-8")

    q_header = _RE_QUERY.search(text).group(1)
    s_header = _RE_SUBJECT.search(text).group(1)
    q_acc = _accession_from_header(q_header)
    s_acc = _accession_from_header(s_header)
    if q_acc not in ACC_TO_LABEL or s_acc not in ACC_TO_LABEL:
        raise ValueError(
            f"Accession desconocido en {path.name}: q={q_acc}, s={s_acc}"
        )
    q_label = ACC_TO_LABEL[q_acc]
    s_label = ACC_TO_LABEL[s_acc]

    sm = _RE_SCORE.search(text)
    if not sm:
        raise ValueError(f"No se encontro 'Score:...Expect:' en {path.name}")
    bit_score = float(sm.group(1))
    e_value = float(sm.group(2))

    im = _RE_IDENT.search(text)
    if not im:
        raise ValueError(f"No se encontro 'Identities:M/L(P%)' en {path.name}")
    identities = int(im.group(1))
    align_length = int(im.group(2))
    # `(91%)` viene redondeado a entero; calculamos exacto.
    identity_pct = 100.0 * identities / max(1, align_length)

    qm = _RE_QCOVER.search(text)
    q_cover = float(qm.group(1)) if qm else 100.0

    return BlastPairResult(
        species_a=q_label,
        species_b=s_label,
        accession_b=s_acc,
        bit_score=bit_score,
        e_value=e_value,
        identity_pct=identity_pct,
        align_length=align_length,
        query_coverage_pct=q_cover,
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def main() -> None:
    files = sorted(SALIDAS.rglob("*-Alignment.txt"))
    print(f"Archivos encontrados: {len(files)}")
    if len(files) != 15:
        print(f"  AVISO: esperaba 15 archivos, hay {len(files)}")

    rows = []
    for f in files:
        try:
            r = parse_blast_txt(f)
        except Exception as e:
            print(f"  [ERROR] {f.relative_to(SALIDAS)}: {e}")
            continue
        rows.append(r)
        print(
            f"  {f.parent.name}/{f.name}: "
            f"{r.species_a:>10} -> {r.species_b:<10} "
            f"%id={r.identity_pct:6.2f}  bit={r.bit_score:6.1f}  "
            f"E={r.e_value:.1e}  cov={r.query_coverage_pct:.0f}%"
        )

    rows.sort(key=lambda r: (r.species_a, r.species_b))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")
    print(f"\n[guardado] {OUTPUT}  ({len(rows)} entradas)")


if __name__ == "__main__":
    main()
