"""
cube_model.py
=================================================================
Representasi data (model) untuk kubus Rubik 2x2x2 (Pocket Cube).

Kubus 2x2x2 hanya terdiri dari 8 CORNER CUBIE (tidak ada edge, tidak
ada center tetap). Karena itu, seluruh state kubus bisa dinyatakan
cukup dengan dua array:

    1. perm   -> permutasi 8 corner (corner mana menempati slot mana)
    2. orient -> orientasi tiap corner (0, 1, atau 2 = jumlah "twist"
                 searah jarum jam dari posisi solved-nya)

Konvensi penamaan & penomoran slot corner (dilihat dari luar kubus,
huruf muka standar: U=Up, D=Down, F=Front, B=Back, L=Left, R=Right):

    index : nama slot : muka-muka penyusun (urutan U/D, lalu 2 muka lain
            berlawanan arah jarum jam saat dilihat dari slot tsb)
    0     : URF
    1     : UFL
    2     : ULB
    3     : UBR
    4     : DFR
    5     : DLF
    6     : DBL
    7     : DRB

Urutan ini SAMA dengan konvensi standar corner pada 3x3x3 (mis. pada
notasi Kociemba), sehingga referensi/tabel dari literatur cubing bisa
dipakai ulang jika suatu saat perlu.
"""

from dataclasses import dataclass
from typing import Tuple

# ---------------------------------------------------------------
# Konstanta struktur kubus
# ---------------------------------------------------------------

CORNER_NAMES = ("URF", "UFL", "ULB", "UBR", "DFR", "DLF", "DBL", "DRB")

# Untuk tiap slot corner, muka fisik dari 3 stiker-nya (dipakai nanti
# oleh modul input warna di Step 6 untuk memetakan warna -> facelet).
CORNER_FACES = {
    0: ("U", "R", "F"),  # URF
    1: ("U", "F", "L"),  # UFL
    2: ("U", "L", "B"),  # ULB
    3: ("U", "B", "R"),  # UBR
    4: ("D", "F", "R"),  # DFR
    5: ("D", "L", "F"),  # DLF
    6: ("D", "B", "L"),  # DBL
    7: ("D", "R", "B"),  # DRB
}

# Skema warna default (standar Barat). Bisa nanti dijadikan opsional/
# custom di modul input warna (Step 6).
DEFAULT_FACE_COLORS = {
    "U": "White",
    "D": "Yellow",
    "F": "Green",
    "B": "Blue",
    "L": "Orange",
    "R": "Red",
}

SOLVED_PERM: Tuple[int, ...] = tuple(range(8))
SOLVED_ORIENT: Tuple[int, ...] = tuple([0] * 8)


# ---------------------------------------------------------------
# Model state kubus
# ---------------------------------------------------------------

@dataclass(frozen=True)
class CubeState:
    """
    State lengkap kubus 2x2x2.

    perm[i]   = corner asli (0-7) yang sekarang menempati slot ke-i
    orient[i] = orientasi corner di slot ke-i (0/1/2)

    Dibuat frozen (immutable) + berbasis tuple supaya:
      - aman dipakai sebagai key dictionary/set saat BFS
      - tidak sengaja termutasi saat dipakai di banyak tempat
    """

    perm: Tuple[int, ...] = SOLVED_PERM
    orient: Tuple[int, ...] = SOLVED_ORIENT

    def is_solved(self) -> bool:
        return self.perm == SOLVED_PERM and self.orient == SOLVED_ORIENT

    def as_key(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Representasi hashable untuk tabel BFS (dict state -> gerakan)."""
        return (self.perm, self.orient)

    def __str__(self) -> str:
        parts = [
            f"{CORNER_NAMES[i]}:{CORNER_NAMES[self.perm[i]]}(+{self.orient[i]})"
            for i in range(8)
        ]
        return " ".join(parts)


SOLVED_STATE = CubeState(SOLVED_PERM, SOLVED_ORIENT)


# ---------------------------------------------------------------
# Validasi fisik state (dipakai ulang nanti oleh modul input warna)
# ---------------------------------------------------------------

def is_valid_state(perm: Tuple[int, ...], orient: Tuple[int, ...]) -> bool:
    """
    Mengecek apakah kombinasi perm+orient MUNGKIN dicapai lewat gerakan
    legal kubus (bukan hasil bongkar-pasang manual/stiker tertukar).

    Aturan fisik kubus 2x2x2:
      1. perm harus permutasi valid dari 0..7
      2. tiap orient harus 0, 1, atau 2
      3. total orientasi harus habis dibagi 3 (mod 3 == 0)

    Catatan: TIDAK ada aturan paritas permutasi di 2x2x2. Di 3x3x3 paritas
    corner terikat dengan paritas edge, tapi 2x2x2 tidak punya edge, dan
    satu putaran 90 derajat (siklus-4) adalah permutasi ganjil. Jadi semua
    8! permutasi corner bisa dicapai.
    """
    if sorted(perm) != list(range(8)):
        return False
    if any(o not in (0, 1, 2) for o in orient):
        return False
    if sum(orient) % 3 != 0:
        return False
    return True


if __name__ == "__main__":
    # Sanity check singkat
    print("Solved state:", SOLVED_STATE)
    print("Is solved   :", SOLVED_STATE.is_solved())
    print("Valid?      :", is_valid_state(SOLVED_PERM, SOLVED_ORIENT))
