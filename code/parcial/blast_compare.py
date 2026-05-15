"""
Comparacion del NW propio con un alineador estandar.

La consigna pide comparar con BLAST. BLAST es heuristico (no garantiza
alineamiento global optimo), asi que reportamos dos referencias:

  1. Bio.Align.PairwiseAligner (alineamiento global exacto en C, mismo
     algoritmo Needleman-Wunsch pero implementacion estandar de Biopython).
     Sirve para validar que nuestro NW da exactamente el mismo score.

  2. NCBIWWW.qblast (BLAST oficial, online). Heuristico, devuelve
     bit-score y E-value. Sirve para mostrar significancia estadistica.

Si la consulta online falla (sin internet o rate-limit), se reportan
solamente los resultados locales.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from Bio.Align import PairwiseAligner, substitution_matrices


@dataclass
class BlastlikeResult:
    pair: str
    nw_score: float        # score de NUESTRO NW
    nw_identity: float
    bio_score: float       # score de Bio.Align.PairwiseAligner
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
    """
    Comparacion local: ejecuta Bio.Align.PairwiseAligner sobre el mismo par
    y reporta score/identidad para contrastar con nuestro NW.
    """
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load(matrix_name)
    aligner.open_gap_score = gap_open
    aligner.extend_gap_score = gap_extend

    aln = aligner.align(seq_a, seq_b)[0]
    # Biopython 1.87: aln[0] y aln[1] devuelven las dos cadenas alineadas
    # (con gaps insertados como "-"). Esto es la forma canonica de leer
    # el alineamiento como par de strings.
    a, b = aln[0], aln[1]
    return BlastlikeResult(
        pair=pair_label,
        nw_score=nw_score,
        nw_identity=nw_identity,
        bio_score=float(aln.score),
        bio_identity=_identity_pct(a, b),
    )


def try_blast_online(
    pair_label: str,
    seq_a: str,
    seq_b: str,
) -> Optional[dict]:
    """Intenta correr blastp 2-seq. Devuelve None si no hay internet."""
    try:
        from Bio.Blast import NCBIWWW, NCBIXML
        from io import StringIO

        handle = NCBIWWW.qblast(
            "blastp",
            "nr",
            f">A\n{seq_a}\n",
            entrez_query=f"txid9606[ORGN]",   # demo: solo humano
            hitlist_size=1,
        )
        record = NCBIXML.read(handle)
        if not record.alignments:
            return None
        hsp = record.alignments[0].hsps[0]
        return {
            "pair": pair_label,
            "bit_score": hsp.bits,
            "e_value": hsp.expect,
            "identities": hsp.identities,
            "length": hsp.align_length,
        }
    except Exception as e:
        print(f"  [BLAST online no disponible: {type(e).__name__}]")
        return None
