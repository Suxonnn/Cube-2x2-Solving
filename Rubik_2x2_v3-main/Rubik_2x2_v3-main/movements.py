"""
movements.py
=================================================================
Fungsi rotasi (gerakan) untuk kubus 2x2x2.

Hanya 3 muka yang dipakai: U, R, F.
Alasannya: kubus 2x2x2 tidak punya center, jadi memutar muka D sama saja
dengan memutar muka U lalu memiringkan seluruh kubus. Dengan hanya memakai
U, R, F, corner DBL (index 6) TIDAK PERNAH bergerak. Ini otomatis
"memfiksasi" satu corner sebagai acuan, sehingga jumlah state yang bisa
dicapai persis 7! x 3^6 = 3.674.160 (bukan 8! x 3^7 = 88 juta).

Tiap muka punya 3 gerakan, sehingga total ada 9 gerakan:
    U  U'  U2    R  R'  R2    F  F'  F2
    (X = 90 derajat searah jarum jam, X' = berlawanan, X2 = 180 derajat)

Cara kerja satu gerakan pada state (perm, orient):
    perm_baru[i]   = perm[T.perm[i]]
    orient_baru[i] = (orient[T.perm[i]] + T.orient[i]) mod 3
dengan T = "tabel gerakan", yaitu state kubus setelah gerakan itu
dilakukan pada kubus yang sudah solved.
"""

from typing import Dict, List, Sequence, Union

from cube_model import CubeState

# Alias index slot corner supaya tabel gerakan mudah dibaca.
# (urutannya sama dengan CORNER_NAMES di cube_model.py)
URF, UFL, ULB, UBR, DFR, DLF, DBL, DRB = range(8)


# ---------------------------------------------------------------
# Tabel 3 putaran dasar (90 derajat searah jarum jam)
# ---------------------------------------------------------------
# perm[i]   = slot asal dari corner yang sekarang berada di slot i
# orient[i] = tambahan twist pada corner yang masuk ke slot i

_BASE_TURNS: Dict[str, CubeState] = {
    "U": CubeState(
        perm=(UBR, URF, UFL, ULB, DFR, DLF, DBL, DRB),
        orient=(0, 0, 0, 0, 0, 0, 0, 0),
    ),
    "R": CubeState(
        perm=(DFR, UFL, ULB, URF, DRB, DLF, DBL, UBR),
        orient=(2, 0, 0, 1, 1, 0, 0, 2),
    ),
    "F": CubeState(
        perm=(UFL, DLF, ULB, UBR, URF, DFR, DBL, DRB),
        orient=(1, 2, 0, 0, 2, 1, 0, 0),
    ),
}


def _apply_table(state: CubeState, table: CubeState) -> CubeState:
    """Terapkan satu tabel gerakan `table` pada `state`."""
    perm = tuple(state.perm[table.perm[i]] for i in range(8))
    orient = tuple(
        (state.orient[table.perm[i]] + table.orient[i]) % 3 for i in range(8)
    )
    return CubeState(perm, orient)


def _build_moves() -> Dict[str, CubeState]:
    """Bangun tabel untuk 9 gerakan (X, X2, X') dari 3 putaran dasar."""
    moves: Dict[str, CubeState] = {}
    for face, quarter in _BASE_TURNS.items():
        half = _apply_table(quarter, quarter)      # X + X   = X2
        three = _apply_table(half, quarter)        # X2 + X  = X'
        moves[face] = quarter
        moves[face + "2"] = half
        moves[face + "'"] = three
    return moves


MOVES: Dict[str, CubeState] = _build_moves()

# Daftar gerakan (urutan tetap) -> dipakai BFS di solver.py
ALL_MOVES = ("U", "U'", "U2", "R", "R'", "R2", "F", "F'", "F2")


# ---------------------------------------------------------------
# API publik
# ---------------------------------------------------------------

def apply_move(state: CubeState, move: str) -> CubeState:
    """Terapkan SATU gerakan (mis. "R", "U'", "F2") dan kembalikan state baru."""
    if move not in MOVES:
        raise ValueError(f"Gerakan tidak dikenal: {move!r}. Pilihan: {ALL_MOVES}")
    return _apply_table(state, MOVES[move])


def parse_moves(sequence: str) -> List[str]:
    """
    Ubah teks seperti "R U2 F' R'" menjadi list ["R", "U2", "F'", "R'"].
    Melempar ValueError jika ada token yang tidak valid.
    """
    normalized = sequence.replace("\u2019", "'").replace("\u2018", "'")
    tokens = normalized.split()
    for token in tokens:
        if token not in MOVES:
            raise ValueError(
                f"Gerakan tidak dikenal: {token!r}. Pilihan: {ALL_MOVES}"
            )
    return tokens


def apply_moves(state: CubeState, sequence: Union[str, Sequence[str]]) -> CubeState:
    """Terapkan urutan gerakan (string "R U2 F'" atau list) secara berurutan."""
    moves = parse_moves(sequence) if isinstance(sequence, str) else list(sequence)
    for move in moves:
        state = apply_move(state, move)
    return state


def inverse_move(move: str) -> str:
    """Kebalikan satu gerakan: R -> R', R' -> R, R2 -> R2."""
    if move not in MOVES:
        raise ValueError(f"Gerakan tidak dikenal: {move!r}")
    if move.endswith("2"):
        return move
    if move.endswith("'"):
        return move[:-1]
    return move + "'"


def inverse_moves(sequence: Union[str, Sequence[str]]) -> str:
    """
    Kebalikan seluruh urutan: urutan dibalik dan tiap gerakan di-invers.
    Contoh: "R U2 F'" -> "F U2 R'"
    Dipakai solver: jalur dari solved -> state, dibalik = solusi state -> solved.
    """
    moves = parse_moves(sequence) if isinstance(sequence, str) else list(sequence)
    return " ".join(inverse_move(m) for m in reversed(moves))


if __name__ == "__main__":
    from cube_model import SOLVED_STATE

    s = apply_moves(SOLVED_STATE, "R U2 F' R'")
    print("Setelah R U2 F' R':", s)
    print("Balik lagi        :", apply_moves(s, inverse_moves("R U2 F' R'")).is_solved())
