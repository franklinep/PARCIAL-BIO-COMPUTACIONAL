"""
Comparacion del NW propio con dos referencias:

  1. Bio.Align.PairwiseAligner (alineamiento global exacto en C, mismo
     algoritmo Needleman-Wunsch que el nuestro pero implementacion
     estandar de Biopython). Sirve para *validar* que nuestro NW da
     exactamente el mismo score. Funcion: `compare_with_biopython`.

  2. BLAST oficial de NCBI (blastp contra base nr, online), llamado via
     `Bio.Blast.NCBIWWW.qblast`. BLAST es heuristico y local; se usa
     para *comparar* score, identidad, bit-score y E-value con NW.
     Funciones: `blast_query_one_vs_many`, `build_blast_matrix`.

Como NCBI rate-limita y cada consulta tarda 30-120 s, los resultados se
persisten en `blast_cache.json`; corridas posteriores leen el cache y
no llaman a internet.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from Bio.Align import PairwiseAligner, substitution_matrices

if TYPE_CHECKING:
    from parcial.phylogeny import Species


# ---------------------------------------------------------------------------
# (1) Validacion local: NW propio vs Bio.Align.PairwiseAligner
# ---------------------------------------------------------------------------

@dataclass
class BlastlikeResult:
    pair: str
    nw_score: float
    nw_identity: float
    bio_score: float
    bio_identity: float


def _identity_pct(a: str, b: str) -> float:
    matches = sum(1 for x, y in zip(a, b) if x == y and x != "-")
    aligned = sum(1 for x, y in zip(a, b) if x != "-" or y != "-")
    return 100.0 * matches / max(1, aligned)


def compare_with_biopython(
    pair_label: str,
    seq_a: str,
    seq_b: str,
    matrix_name: str = "BLOSUM62",
    gap_open: float = -4,
    gap_extend: float = -4,
    nw_score: float = 0.0,
    nw_identity: float = 0.0,
) -> BlastlikeResult:
    """Ejecuta Bio.Align.PairwiseAligner global sobre el par y reporta
    score / identidad para contrastar con nuestro NW."""
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load(matrix_name)
    aligner.open_gap_score = gap_open
    aligner.extend_gap_score = gap_extend

    aln = aligner.align(seq_a, seq_b)[0]
    a, b = aln[0], aln[1]
    return BlastlikeResult(
        pair=pair_label,
        nw_score=nw_score,
        nw_identity=nw_identity,
        bio_score=float(aln.score),
        bio_identity=_identity_pct(a, b),
    )


# ---------------------------------------------------------------------------
# (2) BLAST real contra NCBI: blastp online con cache JSON
# ---------------------------------------------------------------------------

@dataclass
class BlastPairResult:
    """Un hit BLAST entre una secuencia 'query' y un 'subject' identificado
    por su accession UniProt/NCBI. Los porcentajes son numeros entre 0 y 100."""
    species_a: str                  # etiqueta de la secuencia consulta
    species_b: str                  # etiqueta del subject (deducido por accession)
    accession_b: str                # accession UniProt/NCBI del subject
    bit_score: float                # bit-score del mejor HSP
    e_value: float                  # E-value del mejor HSP
    identity_pct: float             # 100 * identities / align_length
    align_length: int               # longitud del HSP
    query_coverage_pct: float       # 100 * (q_end - q_start + 1) / len(seq_a)
    fetched_at: str = ""            # ISO timestamp UTC


def _strip_version(accession: str) -> str:
    """`P99999.2` -> `P99999`."""
    return accession.split(".")[0]


def blast_query_one_vs_many(
    query_label: str,
    query_seq: str,
    subject_accessions: Dict[str, str],   # other_label -> accession (sin version)
    database: str = "swissprot",
    sleep_after_request: float = 2.0,
) -> Dict[str, BlastPairResult]:
    """Ejecuta UNA llamada blastp online (`Bio.Blast.NCBIWWW.qblast`) usando
    `query_seq`. Pedimos suficientes hits y filtramos los que correspondan
    a los accessions de los subjects (las 6 secuencias de citocromo c son
    ortologos directos entre si, asi que aparecen entre los top hits).

    Usamos `database='swissprot'` (UniProt reviewed): es donde estan
    indexados los accessions UniProt que ya tenemos en los FASTA locales,
    y la busqueda es mas rapida y deterministica que contra `nr`.

    Devuelve dict {other_label: BlastPairResult}. Labels no encontrados se
    omiten (no aparecen en el dict)."""
    from Bio.Blast import NCBIWWW, NCBIXML

    # `hitlist_size` con margen: ortologos de citocromo c hay decenas, y
    # queremos garantizar que las 5 especies objetivo esten entre los hits.
    hitlist_size = max(100, len(subject_accessions) * 10)

    print(f"  [BLAST] {query_label}: enviando blastp ({database}) a NCBI...")
    handle = NCBIWWW.qblast(
        program="blastp",
        database=database,
        sequence=query_seq,
        hitlist_size=hitlist_size,
        expect=10.0,
    )
    record = NCBIXML.read(handle)
    handle.close()

    # Reverse map: accession (sin version) -> label
    acc_to_label = {acc: lbl for lbl, acc in subject_accessions.items()}

    results: Dict[str, BlastPairResult] = {}
    q_len = len(query_seq)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for aln in record.alignments:
        # Para swissprot, aln.accession devuelve directamente "P99998"
        # (sin sufijo de version). Pero hit_def a veces concatena varios
        # accessions (`>sp|P99999.2| ...`) por lo que tambien escaneamos
        # ahi como respaldo.
        candidate_accs = {_strip_version(aln.accession)}
        for token in aln.hit_def.replace("|", " ").split():
            candidate_accs.add(_strip_version(token.strip(",;.()")))

        matched_label = None
        matched_acc = None
        for acc in candidate_accs:
            if acc in acc_to_label:
                matched_label = acc_to_label[acc]
                matched_acc = acc
                break
        if matched_label is None:
            continue
        if matched_label in results:
            continue  # ya tenemos el mejor HSP de esa especie

        hsp = aln.hsps[0]  # NCBIXML ya entrega HSPs ordenados por score
        q_cov = 100.0 * (hsp.query_end - hsp.query_start + 1) / max(1, q_len)
        results[matched_label] = BlastPairResult(
            species_a=query_label,
            species_b=matched_label,
            accession_b=matched_acc,
            bit_score=float(hsp.bits),
            e_value=float(hsp.expect),
            identity_pct=100.0 * hsp.identities / max(1, hsp.align_length),
            align_length=int(hsp.align_length),
            query_coverage_pct=q_cov,
            fetched_at=now,
        )

    print(f"  [BLAST] {query_label}: {len(results)}/{len(subject_accessions)} subjects mapeados")
    if sleep_after_request > 0:
        time.sleep(sleep_after_request)
    return results


# ---------------------------------------------------------------------------
# Cache JSON
# ---------------------------------------------------------------------------

def _load_cache(path: Path) -> Dict[Tuple[str, str], BlastPairResult]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[Tuple[str, str], BlastPairResult] = {}
    for entry in raw:
        r = BlastPairResult(**entry)
        out[(r.species_a, r.species_b)] = r
    return out


def _save_cache(path: Path, cache: Dict[Tuple[str, str], BlastPairResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in cache.values()]
    rows.sort(key=lambda r: (r["species_a"], r["species_b"]))
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Driver: matriz BLAST pareada con simetrizacion
# ---------------------------------------------------------------------------

def build_blast_matrix(
    species: List["Species"],
    accession_map: Dict[str, str],       # label -> accession (sin version)
    cache_path: Path,
    force_refresh: bool = False,
) -> Dict[Tuple[str, str], Optional[BlastPairResult]]:
    """Ejecuta una blastp query por especie (cada secuencia como query contra
    las otras 5 via entrez_query) y simetriza los resultados.

    Devuelve un dict indexado por tupla *ordenada* (label_a, label_b) con
    el mejor de los dos sentidos i->j / j->i. Pares para los que no se
    pudo recuperar nada quedan en None.

    El cache se actualiza incrementalmente: solo llama a NCBI para queries
    que aun no estan en disco (a menos que `force_refresh=True`)."""
    directional: Dict[Tuple[str, str], BlastPairResult] = _load_cache(cache_path)
    if force_refresh:
        directional = {}

    label_to_seq = {s.label: s.sequence for s in species}
    labels = [s.label for s in species]

    for q_label in labels:
        subj_accs = {lbl: accession_map[lbl] for lbl in labels if lbl != q_label}
        # Un par (a,b) se considera ya cubierto si tenemos cualquiera de los
        # dos sentidos: el dendrograma BLAST se construye sobre la matriz
        # simetrizada al final, no necesitamos repetir la consulta en sentido
        # inverso si ya tenemos un HSP valido.
        missing = [
            lbl for lbl in subj_accs
            if (q_label, lbl) not in directional and (lbl, q_label) not in directional
        ]
        if not missing:
            continue
        try:
            hits = blast_query_one_vs_many(
                query_label=q_label,
                query_seq=label_to_seq[q_label],
                subject_accessions={lbl: subj_accs[lbl] for lbl in missing},
            )
        except Exception as e:
            print(f"  [BLAST] error en {q_label}: {type(e).__name__}: {e}")
            continue
        for lbl, hit in hits.items():
            directional[(q_label, lbl)] = hit
        # persistir parcialmente: si la corrida se interrumpe no perdemos nada
        _save_cache(cache_path, directional)

    # Simetrizar: para cada par no ordenado, escoger el HSP de mayor bit_score.
    # Las claves del dict resultante estan ordenadas alfabeticamente para que
    # coincidan con tuple(sorted(...)) que usa DistanceMatrixBuilder.
    pairwise: Dict[Tuple[str, str], Optional[BlastPairResult]] = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            key = tuple(sorted((a, b)))
            r_ab = directional.get((a, b))
            r_ba = directional.get((b, a))
            if r_ab and r_ba:
                pairwise[key] = r_ab if r_ab.bit_score >= r_ba.bit_score else r_ba
            elif r_ab:
                pairwise[key] = r_ab
            elif r_ba:
                pairwise[key] = r_ba
            else:
                pairwise[key] = None
    return pairwise
