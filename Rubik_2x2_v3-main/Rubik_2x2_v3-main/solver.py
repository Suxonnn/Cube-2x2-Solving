"""
solver.py
=================================================================
Solver optimal untuk kubus 2x2x2 via BFS (breadth-first search)
menyisir SELURUH 3.674.160 state yang bisa dicapai lewat gerakan
U, R, F (lihat movements.py untuk penjelasan kenapa hanya 3 muka ini
sudah cukup mewakili seluruh state space).

--------------------------------------------------------------
Encoding state -> integer (0 .. 3.674.159)
--------------------------------------------------------------
Corner index 6 (DBL) selalu diam di posisi 6, orientasi 0 (dijamin oleh
movements.py). Jadi cukup dikodekan:
    - permutasi 7 corner lain di 7 posisi lain      -> 7!  = 5.040
    - orientasi 6 dari 7 posisi itu (yang ke-7 ikut
      constraint total-mod-3 == 0)                   -> 3^6 = 729
Total = 5.040 x 729 = 3.674.160  (persis 8! x 3^7 / 24, jumlah baku
posisi kubus 2x2x2 yang sudah didokumentasikan luas).

Tujuannya: tabel BFS bisa disimpan sebagai array 1-byte per state
(~3.5 MB) alih-alih dictionary tuple (>1 GB).

--------------------------------------------------------------
Strategi tabel BFS
--------------------------------------------------------------
BFS dimulai dari SOLVED_STATE, menjelajah lewat 9 gerakan. Untuk tiap
state baru, kita simpan HANYA satu gerakan: gerakan yang harus
diterapkan pada state itu supaya satu langkah lebih dekat ke solved
(yaitu invers dari gerakan yang dipakai BFS untuk pertama kali
menemukannya). Untuk menyelesaikan kubus mana pun: baca tabel,
terapkan gerakan yang tersimpan, ulangi sampai solved. Karena BFS
menjamin jarak minimum, hasilnya otomatis OPTIMAL.
"""

import os
from array import array
from typing import List, Optional

from cube_model import CubeState, SOLVED_STATE, is_valid_state
from movements import ALL_MOVES, apply_move

# ----------------------------------------------------------------
# Encoding
# ----------------------------------------------------------------

FREE_POS = (0, 1, 2, 3, 4, 5, 7)            # semua posisi kecuali slot 6
N_FREE = len(FREE_POS)                       # 7
FREE_CORNER_IDS = tuple(sorted(FREE_POS))    # (0,1,2,3,4,5,7)
_CORNER_RANK = {cid: i for i, cid in enumerate(FREE_CORNER_IDS)}

_FACT = [1] * (N_FREE + 1)
for _i in range(1, N_FREE + 1):
    _FACT[_i] = _FACT[_i - 1] * _i

PERM_TABLE_SIZE = _FACT[N_FREE]              # 5040
ORIENT_TABLE_SIZE = 3 ** (N_FREE - 1)        # 729
TABLE_SIZE = PERM_TABLE_SIZE * ORIENT_TABLE_SIZE  # 3_674_160


def _lehmer_encode(ranks: List[int]) -> int:
    """Encode permutasi N_FREE elemen berbeda (0..N_FREE-1) -> 0..N_FREE!-1."""
    ranks = list(ranks)
    n = len(ranks)
    code = 0
    for i in range(n):
        smaller = sum(1 for j in range(i + 1, n) if ranks[j] < ranks[i])
        code += smaller * _FACT[n - 1 - i]
    return code


def _lehmer_decode(code: int, n: int) -> List[int]:
    """Kebalikan dari _lehmer_encode."""
    elements = list(range(n))
    result = []
    for i in range(n, 0, -1):
        f = _FACT[i - 1]
        idx, code = divmod(code, f)
        result.append(elements.pop(idx))
    return result


def encode_state(state: CubeState) -> int:
    """CubeState -> integer index tabel (0 .. TABLE_SIZE-1)."""
    ranks = [_CORNER_RANK[state.perm[p]] for p in FREE_POS]
    perm_code = _lehmer_encode(ranks)

    orient_code = 0
    for p in FREE_POS[:-1]:          # 6 digit bebas; digit ke-7 turunan
        orient_code = orient_code * 3 + state.orient[p]

    return perm_code * ORIENT_TABLE_SIZE + orient_code


def decode_state(index: int) -> CubeState:
    """Kebalikan dari encode_state. Dipakai untuk testing/debugging."""
    perm_code, orient_code = divmod(index, ORIENT_TABLE_SIZE)
    ranks = _lehmer_decode(perm_code, N_FREE)

    perm = [0] * 8
    perm[6] = 6
    for pos, rank in zip(FREE_POS, ranks):
        perm[pos] = FREE_CORNER_IDS[rank]

    digits = []
    c = orient_code
    for _ in range(N_FREE - 1):
        c, d = divmod(c, 3)
        digits.append(d)
    digits.reverse()

    orient = [0] * 8
    total = 0
    for pos, d in zip(FREE_POS[:-1], digits):
        orient[pos] = d
        total += d
    orient[FREE_POS[-1]] = (-total) % 3

    return CubeState(tuple(perm), tuple(orient))


def is_encodable(state: CubeState) -> bool:
    """
    True jika `state` valid secara fisik DAN sesuai konvensi modul ini:
    corner index 6 (DBL) berada tepat di slot 6 dengan orientasi 0.

    Catatan penting untuk modul input warna (Step 6): kubus asli di dunia
    nyata tidak "tahu" mana yang harus jadi slot 6 -- itu murni pilihan
    representasi kita. Modul Step 6 WAJIB menormalisasi (merotasi secara
    logis) hasil pembacaan warna pengguna sehingga corner yang jadi acuan
    selalu ditempatkan di slot 6, SEBELUM memanggil solve(). encode_state()
    sendiri tidak melakukan pengecekan ini (dipakai jutaan kali saat BFS,
    jadi sengaja dibuat tanpa validasi demi kecepatan) -- makanya solve()
    memanggil is_encodable() di awal sebagai pintu gerbang keamanan.
    """
    if not is_valid_state(state.perm, state.orient):
        return False
    if state.perm[6] != 6 or state.orient[6] != 0:
        return False
    return True


# ----------------------------------------------------------------
# Tabel BFS
# ----------------------------------------------------------------

_INVERSE_INDEX = {}
for _i, _m in enumerate(ALL_MOVES):
    if _m.endswith("2"):
        _inv = _m
    elif _m.endswith("'"):
        _inv = _m[:-1]
    else:
        _inv = _m + "'"
    _INVERSE_INDEX[_m] = ALL_MOVES.index(_inv)

UNVISITED = -1
SOLVED_MARK = -2

DEFAULT_TABLE_PATH = os.path.join(os.path.dirname(__file__), "bfs_table.bin")


def build_bfs_table(verbose: bool = True) -> array:
    """Jelajahi seluruh state via BFS dan bangun tabel 1-byte-per-state."""
    table = array("b", (UNVISITED for _ in range(TABLE_SIZE)))
    table[encode_state(SOLVED_STATE)] = SOLVED_MARK

    frontier = [SOLVED_STATE]
    total = 1
    depth = 0
    if verbose:
        print(f"  depth {depth}: total = {total}")

    while frontier:
        next_frontier = []
        for state in frontier:
            for move in ALL_MOVES:
                new_state = apply_move(state, move)
                enc = encode_state(new_state)
                if table[enc] == UNVISITED:
                    table[enc] = _INVERSE_INDEX[move]
                    next_frontier.append(new_state)
        if not next_frontier:
            break
        depth += 1
        total += len(next_frontier)
        if verbose:
            print(f"  depth {depth}: total = {total} (+{len(next_frontier)})")
        frontier = next_frontier

    if total != TABLE_SIZE:
        raise RuntimeError(
            f"BFS hanya menemukan {total} state, seharusnya {TABLE_SIZE}. "
            "Ada bug di movements.py atau encoding."
        )
    if verbose:
        print(f"Selesai. Total state = {total} (kedalaman maksimum = {depth})")
    return table


def save_table(table: array, path: str = DEFAULT_TABLE_PATH) -> None:
    with open(path, "wb") as f:
        table.tofile(f)


def load_table(path: str = DEFAULT_TABLE_PATH) -> array:
    table = array("b")
    with open(path, "rb") as f:
        table.fromfile(f, TABLE_SIZE)
    return table


_TABLE_CACHE: Optional[array] = None


def get_table(path: str = DEFAULT_TABLE_PATH, verbose: bool = True) -> array:
    """
    Ambil tabel BFS: pakai cache di memori kalau sudah pernah dimuat,
    kalau tidak coba load dari file, kalau file belum ada -> build lalu
    simpan ke file supaya proses berikutnya tinggal load (cepat).
    """
    global _TABLE_CACHE
    if _TABLE_CACHE is not None:
        return _TABLE_CACHE
    if os.path.exists(path):
        _TABLE_CACHE = load_table(path)
    else:
        _TABLE_CACHE = build_bfs_table(verbose=verbose)
        save_table(_TABLE_CACHE, path)
    return _TABLE_CACHE


# ----------------------------------------------------------------
# API solve
# ----------------------------------------------------------------

MAX_SOLVE_LEN = 20  # jaga-jaga terhadap infinite loop kalau tabel korup


def solve(state: CubeState, table: Optional[array] = None) -> List[str]:
    """
    Kembalikan daftar gerakan (mis. ["R", "U2", "F'", "R'"]) yang
    menyelesaikan `state` ke posisi solved. List kosong jika `state`
    sudah solved. Solusi dijamin OPTIMAL (jumlah gerakan minimum).
    """
    if table is None:
        table = get_table(verbose=False)

    if not is_encodable(state):
        raise ValueError(
            "State tidak bisa diselesaikan: harus valid secara fisik "
            "(cube_model.is_valid_state) DAN corner ke-6 (DBL) berada di "
            "slot 6 dengan orientasi 0. Modul input warna (Step 6) harus "
            "menormalisasi orientasi kubus terlebih dahulu -- lihat "
            "solver.is_encodable()."
        )

    moves: List[str] = []
    current = state
    for _ in range(MAX_SOLVE_LEN):
        enc = encode_state(current)
        code = table[enc]
        if code == UNVISITED:
            # Seharusnya tidak pernah terjadi kalau tabel dibangun benar dan
            # is_encodable() sudah lolos -- disimpan sbg jaring pengaman.
            raise RuntimeError(
                "State encodable tapi tidak ada di tabel BFS -- tabel "
                "kemungkinan korup/tidak lengkap. Coba build ulang."
            )
        if code == SOLVED_MARK:
            return moves
        move = ALL_MOVES[code]
        moves.append(move)
        current = apply_move(current, move)

    raise RuntimeError("Solve melebihi batas gerakan maksimum, ada yang salah.")


def solve_to_string(state: CubeState, table: Optional[array] = None) -> str:
    moves = solve(state, table)
    return " ".join(moves) if moves else "(sudah solved)"


if __name__ == "__main__":
    from movements import apply_moves

    print("Membangun/memuat tabel BFS...")
    t = get_table()

    scramble = "R U2 F' R' U F2 R U'"
    scrambled = apply_moves(SOLVED_STATE, scramble)
    solution = solve_to_string(scrambled, t)
    print(f"Scramble : {scramble}")
    print(f"Solusi   : {solution}")
    print(f"Terverifikasi solved: {apply_moves(scrambled, solution or '').is_solved()}")
