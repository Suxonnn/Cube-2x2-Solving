"""
color_mapping.py
=================================================================
Step 6: Konversi input warna pengguna -> CubeState (permutasi & orientasi)
yang dipahami solver.py, lengkap dengan validasi.

------------------------------------------------------------------
FORMAT INPUT
------------------------------------------------------------------
Pengguna memegang kubus fisiknya pada satu orientasi tetap (bebas
orientasi apa saja, asal konsisten selama input), lalu untuk SETIAP
dari 8 corner (slot 0..7, lihat CORNER_NAMES di cube_model.py),
melaporkan 3 warna yang ia lihat, dalam urutan: [warna muka U-atau-D
di corner itu, lalu 2 muka lain searah jarum jam bila dilihat dari
luar kubus] -- urutan ini PERSIS mengikuti CORNER_FACES.

    corners = {
        0: ("White", "Red", "Green"),    # slot 0 = URF
        1: ("White", "Green", "Orange"), # slot 1 = UFL
        ...
        7: ("Yellow", "Red", "Blue"),    # slot 7 = DRB
    }

------------------------------------------------------------------
KENAPA PERLU "KANONIKALISASI" (ROTASI SELURUH KUBUS)
------------------------------------------------------------------
solver.py cuma bisa menyelesaikan state di mana corner id 6 (piece
berwarna {Yellow, Blue, Orange} pada skema default) berada di slot 6
dengan orientasi 0 -- konsekuensi memakai gerakan U,R,F saja. Pada
kubus yang benar-benar teracak, piece itu HAMPIR PASTI tidak di sana.

Solusi: rotasi SELURUH kubus (memutar kubus utuh di tangan, BUKAN
gerakan lapisan -- tidak mengubah acakan aslinya) sampai piece itu di
slot 6 orientasi 0. Ada tepat 24 rotasi kubus yang mungkin, dan untuk
state manapun, TEPAT SATU di antaranya akan membawa piece manapun ke
(slot 6, orientasi 0) -- ini fakta grup rotasi kubus (aksinya
transitif pada 8 corner, dgn stabilizer berorde 3 di tiap corner: 8*3
= 24). Rotasi-rotasi ini dibangun dari koordinat geometris (bukan
tabel hasil hafalan) supaya sifatnya bisa diverifikasi lewat kode,
bukan cuma dipercaya begitu saja.

Karena rotasi ini murni alat bantu INTERNAL (bukan gerakan yang
ditampilkan ke user), solusi akhir ditampilkan sbg instruksi 2 bagian:
  1. "Putar SELURUH kubus supaya corner X ada di posisi DBL" (memakai
     nama slot yg SAMA dgn yg diketik user -- tidak perlu penamaan
     rotasi baru sama sekali).
  2. Baru terapkan solusi U/R/F yg sudah divalidasi ekstensif di Step 3-5.
Desain ini sengaja MENGHINDARI penerjemahan gerakan U/R/F -> D/L/B
(yang butuh konjugasi rotasi & rawan salah arah) -- lebih aman dan
lebih gampang dipahami.
"""

from typing import Dict, List, Optional, Tuple

from cube_model import (
    CORNER_FACES,
    CORNER_NAMES,
    CubeState,
    DEFAULT_FACE_COLORS,
    is_valid_state,
)
from movements import apply_move
from solver import is_encodable, solve as bfs_solve

# ----------------------------------------------------------------
# Skema warna & pasangan berlawanan
# ----------------------------------------------------------------

OPPOSITE_FACE = {"U": "D", "D": "U", "F": "B", "B": "F", "L": "R", "R": "L"}


class ColorMappingError(ValueError):
    """Error validasi input warna (pesan siap ditampilkan ke user)."""


def _face_of_color(color_scheme: Dict[str, str], color: str) -> str:
    for face, c in color_scheme.items():
        if c == color:
            return face
    raise ColorMappingError(f"Warna '{color}' tidak dikenal di skema warna.")


def _opposite_color(color_scheme: Dict[str, str], color: str) -> str:
    return color_scheme[OPPOSITE_FACE[_face_of_color(color_scheme, color)]]


def _rotate3(tup: Tuple, o: int) -> Tuple:
    """Rotasi cyclic: hasil[i] = tup[(i+o) % 3]."""
    return tuple(tup[(i + o) % 3] for i in range(3))


# ----------------------------------------------------------------
# Validasi input mentah
# ----------------------------------------------------------------

def validate_raw_input(
    corners: Dict[int, Tuple[str, str, str]],
    color_scheme: Dict[str, str] = DEFAULT_FACE_COLORS,
) -> None:
    """
    Melempar ColorMappingError dgn pesan jelas kalau ada masalah:
      1. Harus ada tepat 8 corner (slot 0..7).
      2. Tiap corner harus 3 warna yg dikenal skema warna.
      3. Tiap corner tidak boleh py warna yg SAMA dobel.
      4. Tiap corner TIDAK BOLEH punya 2 warna dari pasangan berlawanan
         (mis. Merah & Orange sekaligus) -- mustahil secara fisik.
      5. Kombinasi 3-warna tiap corner harus cocok salah satu dari 8
         kombinasi corner yg mungkin ada di kubus manapun.
      6. Tidak boleh 2 slot dgn kombinasi warna corner yg sama (piece
         fisik tidak mungkin dobel).
      7. Tiap warna harus muncul PERSIS 4 kali di seluruh 24 stiker
         (4 stiker/muka x 6 muka, merata).
    """
    if set(corners.keys()) != set(range(8)):
        raise ColorMappingError(
            f"Input harus punya tepat 8 corner (slot 0-7). Ditemukan slot: "
            f"{sorted(corners.keys())}"
        )

    valid_colors = set(color_scheme.values())
    color_counter = {c: 0 for c in valid_colors}
    canonical_sets = {
        i: frozenset(color_scheme[f] for f in CORNER_FACES[i]) for i in range(8)
    }
    seen_corner_sets: Dict[frozenset, int] = {}

    for slot, colors in corners.items():
        name = CORNER_NAMES[slot]
        if len(colors) != 3:
            raise ColorMappingError(f"Corner {name} (slot {slot}) harus punya 3 warna.")

        for c in colors:
            if c not in valid_colors:
                raise ColorMappingError(
                    f"Corner {name}: warna '{c}' tidak dikenal. Pilihan: "
                    f"{sorted(valid_colors)}"
                )
            color_counter[c] += 1

        if len(set(colors)) != 3:
            raise ColorMappingError(
                f"Corner {name}: warnanya tidak boleh ada yang sama "
                f"({colors}) -- satu piece kubus punya 3 warna berbeda."
            )

        for c in colors:
            opp = _opposite_color(color_scheme, c)
            if opp in colors:
                raise ColorMappingError(
                    f"Corner {name}: tidak mungkin punya warna '{c}' dan "
                    f"'{opp}' sekaligus -- keduanya muka berseberangan pada "
                    f"kubus, tidak pernah bertemu di satu corner."
                )

        color_set = frozenset(colors)
        if color_set not in canonical_sets.values():
            raise ColorMappingError(
                f"Corner {name}: kombinasi warna {colors} bukan salah satu "
                f"dari 8 kombinasi corner yang mungkin ada di kubus manapun."
            )
        if color_set in seen_corner_sets:
            other = seen_corner_sets[color_set]
            raise ColorMappingError(
                f"Corner {name} dan corner {CORNER_NAMES[other]} punya "
                f"kombinasi warna yang sama ({colors}) -- tidak mungkin dua "
                f"piece fisik identik."
            )
        seen_corner_sets[color_set] = slot

    for c, n in color_counter.items():
        if n != 4:
            raise ColorMappingError(
                f"Warna '{c}' muncul {n} kali di input, seharusnya tepat 4 "
                f"kali (4 stiker per muka)."
            )


# ----------------------------------------------------------------
# Identifikasi corner dari warna -> (corner_id, orientasi)
# ----------------------------------------------------------------

def _canonical_orders(color_scheme: Dict[str, str]) -> Dict[int, Tuple[str, str, str]]:
    return {i: tuple(color_scheme[f] for f in CORNER_FACES[i]) for i in range(8)}


def _identify_corner(
    colors: Tuple[str, str, str],
    canonical_orders: Dict[int, Tuple[str, str, str]],
) -> Tuple[int, int]:
    obs = tuple(colors)
    obs_set = frozenset(obs)
    for cid, canon in canonical_orders.items():
        if frozenset(canon) != obs_set:
            continue
        for o in range(3):
            if _rotate3(canon, o) == obs:
                return cid, (3 - o) % 3
    raise ColorMappingError(f"Tidak ada corner kanonik yang cocok dengan warna {colors}.")


def raw_state_from_colors(
    corners: Dict[int, Tuple[str, str, str]],
    color_scheme: Dict[str, str] = DEFAULT_FACE_COLORS,
) -> CubeState:
    """Validasi lalu ubah input warna -> CubeState MENTAH (belum dikanonikalisasi)."""
    validate_raw_input(corners, color_scheme)
    canon = _canonical_orders(color_scheme)

    perm: List[Optional[int]] = [None] * 8
    orient: List[Optional[int]] = [None] * 8
    for slot, colors in corners.items():
        cid, o = _identify_corner(colors, canon)
        perm[slot] = cid
        orient[slot] = o

    perm_t, orient_t = tuple(perm), tuple(orient)
    if not is_valid_state(perm_t, orient_t):
        raise ColorMappingError(
            "Kombinasi hasil scan tidak mungkin dicapai secara fisik (total "
            "twist tidak habis dibagi 3). Kemungkinan ada 1 corner yang "
            "urutan warnanya salah arah saat diinput -- coba scan ulang."
        )
    return CubeState(perm_t, orient_t)


# ----------------------------------------------------------------
# Rotasi seluruh kubus (geometri koordinat) -- alat bantu INTERNAL
# ----------------------------------------------------------------

# Sumbu: x: R=+1/L=-1, y: U=+1/D=-1, z: F=+1/B=-1
CORNER_COORD = {
    0: (1, 1, 1), 1: (-1, 1, 1), 2: (-1, 1, -1), 3: (1, 1, -1),
    4: (1, -1, 1), 5: (-1, -1, 1), 6: (-1, -1, -1), 7: (1, -1, -1),
}
COORD_TO_SLOT = {v: k for k, v in CORNER_COORD.items()}

FACE_TO_AXIS_DIR = {
    "U": (0, 1, 0), "D": (0, -1, 0),
    "F": (0, 0, 1), "B": (0, 0, -1),
    "R": (1, 0, 0), "L": (-1, 0, 0),
}

CANONICAL_DIRS = {i: tuple(FACE_TO_AXIS_DIR[f] for f in CORNER_FACES[i]) for i in range(8)}


def _rot_x(v):
    x, y, z = v
    return (x, -z, y)


def _rot_y(v):
    x, y, z = v
    return (z, y, -x)


def _generate_rotations():
    """
    BFS menyusun 24 rotasi dari 2 generator (_rot_x, _rot_y), SEKALIGUS
    mencatat jalur (urutan token 'x'/'y') terpendek yg menyusun tiap
    rotasi -- dipakai utk animasi reorientasi: tiap token = 1 langkah
    putaran 90 derajat pada SELURUH kubus yg bisa dianimasikan satu-satu.
    """
    def compose(f, g):
        return lambda v: f(g(v))

    basis = ((1, 0, 0), (0, 1, 0), (0, 0, 1))

    def sig(f):
        return tuple(f(b) for b in basis)

    identity = lambda v: v
    seen = {sig(identity): identity}
    rotations = [identity]
    paths = [[]]
    frontier = [(identity, [])]
    generators = [("x", _rot_x), ("y", _rot_y)]
    while frontier:
        nxt = []
        for f, path in frontier:
            for token, g in generators:
                h = compose(g, f)
                s = sig(h)
                if s not in seen:
                    seen[s] = h
                    rotations.append(h)
                    paths.append(path + [token])
                    nxt.append((h, path + [token]))
        frontier = nxt
    return rotations, paths


ROTATIONS, ROTATION_PATHS = _generate_rotations()  # harus persis 24 elemen (dicek di test)
ROTATION_GENERATOR = {"x": _rot_x, "y": _rot_y}


def apply_rotation(state: CubeState, rot_fn) -> CubeState:
    """Terapkan satu rotasi kubus utuh pada `state` -> CubeState baru."""
    new_perm: List[Optional[int]] = [None] * 8
    new_orient: List[Optional[int]] = [None] * 8

    for old_slot in range(8):
        new_slot = COORD_TO_SLOT[rot_fn(CORNER_COORD[old_slot])]

        # Arah tiap stiker MILIK PIECE (indeks = urutan stiker kanonik piece).
        # Dgn konvensi Kociemba, stiker piece ke-j ada di muka posisi
        # (j + orient) % 3 dari slot -> rotate3(dirs_slot, orient).
        old_dirs_observed = _rotate3(CANONICAL_DIRS[old_slot], state.orient[old_slot])
        new_dirs_observed = tuple(rot_fn(d) for d in old_dirs_observed)

        target_canon = CANONICAL_DIRS[new_slot]
        o = next(
            cand for cand in range(3)
            if _rotate3(target_canon, cand) == new_dirs_observed
        )

        new_perm[new_slot] = state.perm[old_slot]
        new_orient[new_slot] = o

    return CubeState(tuple(new_perm), tuple(new_orient))


def canonicalize(raw_state: CubeState):
    """
    Cari (di antara 24 rotasi) satu-satunya yang membawa corner id 6 ke
    slot 6 orientasi 0. Return (canonical_state, original_slot_of_ref,
    original_orient_of_ref, rotation_path) -- rotation_path adalah list
    token 'x'/'y' (tiap token = 1 langkah putar 90 derajat SELURUH
    kubus), dipakai utk animasi reorientasi.
    """
    original_slot = raw_state.perm.index(6)
    original_orient = raw_state.orient[original_slot]

    for rot, path in zip(ROTATIONS, ROTATION_PATHS):
        candidate = apply_rotation(raw_state, rot)
        if is_encodable(candidate):
            return candidate, original_slot, original_orient, path

    raise RuntimeError(
        "Tidak ditemukan rotasi yang valid -- seharusnya mustahil secara "
        "matematis, cek implementasi apply_rotation()."
    )


# ----------------------------------------------------------------
# Simulasi kebalikan (state -> warna) -- dipakai untuk TESTING
# ----------------------------------------------------------------

def state_to_colors(
    state: CubeState, color_scheme: Dict[str, str] = DEFAULT_FACE_COLORS
) -> Dict[int, Tuple[str, str, str]]:
    """Kebalikan dari raw_state_from_colors. Berguna utk simulasi & test."""
    canon = _canonical_orders(color_scheme)
    return {slot: _rotate3(canon[state.perm[slot]], (3 - state.orient[slot]) % 3) for slot in range(8)}


# ----------------------------------------------------------------
# API utama
# ----------------------------------------------------------------

def _states_to_json_colors(states: List[CubeState], color_scheme) -> List[Dict[str, List[str]]]:
    """Ubah daftar CubeState -> daftar dict warna yg aman di-JSON-kan (dipakai UI utk preview tiap langkah)."""
    out = []
    for st in states:
        colors = state_to_colors(st, color_scheme)
        out.append({str(slot): list(c) for slot, c in colors.items()})
    return out


def colors_to_solution(
    corners: Dict[int, Tuple[str, str, str]],
    color_scheme: Dict[str, str] = DEFAULT_FACE_COLORS,
) -> Dict[str, object]:
    """Pipeline lengkap: validasi -> identifikasi -> kanonikalisasi -> solve."""
    raw_state = raw_state_from_colors(corners, color_scheme)

    if raw_state.is_solved():
        return {
            "moves": [],
            "moves_str": "(sudah solved)",
            "needs_reorientation": False,
            "reorientation_steps": [],
            "already_solved": True,
            "states": _states_to_json_colors([raw_state], color_scheme),
        }

    canonical_state, orig_slot, orig_orient, rotation_path = canonicalize(raw_state)
    moves = bfs_solve(canonical_state)

    # --- Tahap 1: animasi REORIENTASI (putar SELURUH kubus, bukan 1 sisi) ---
    # rotation_path = daftar token 'x'/'y', tiap token = 1 langkah putar 90
    # derajat utk SEMUA 8 corner sekaligus. reorientation_steps = state
    # (dlm bentuk warna) SETELAH tiap langkah itu diterapkan -- dipakai UI
    # utk menganimasikan kubus "berputar sendiri" dari posisi apa adanya
    # (sesuai input user) ke posisi kanonik, tanpa perlu teks penjelasan.
    reorientation_steps = []
    cur_raw = raw_state
    for token in rotation_path:
        cur_raw = apply_rotation(cur_raw, ROTATION_GENERATOR[token])
        reorientation_steps.append({
            "axis": token,
            "state": {str(s): list(c) for s, c in state_to_colors(cur_raw, color_scheme).items()},
        })
    # langkah terakhir (kalau ada) harus persis == canonical_state
    needs_reorientation = len(rotation_path) > 0

    # --- Tahap 2: animasi TIAP GERAKAN solusi (U, R, F, dst) ---
    states = [canonical_state]
    cur = canonical_state
    for m in moves:
        cur = apply_move(cur, m)
        states.append(cur)

    return {
        "moves": moves,
        "moves_str": " ".join(moves),
        "needs_reorientation": needs_reorientation,
        "reorientation_steps": reorientation_steps,  # [] kalau tdk perlu
        "already_solved": False,
        "states": _states_to_json_colors(states, color_scheme),
    }


if __name__ == "__main__":
    # Demo konkret: acak kubus dgn urutan gerakan TERTENTU (kita "tahu"
    # jawabannya), lalu pura-pura jadi user yg baru saja scan warna kubus
    # itu -- tanpa memberi tahu solver urutan acakannya.
    import random
    from cube_model import SOLVED_STATE
    from movements import apply_moves

    scramble = "R U2 F' R' U F2 R U'"
    print(f"(Untuk demo) kubus diacak dengan: {scramble}")
    scrambled = apply_moves(SOLVED_STATE, scramble)

    # Simulasikan user memegang kubus dlm orientasi ACAK (bukan kebetulan
    # pas dgn kerangka internal) -- ini yg bikin demo realistis.
    random.seed(42)
    held = apply_rotation(scrambled, random.choice(ROTATIONS))

    colors = state_to_colors(held)
    print("\nWarna yang 'dibaca' dari tiap corner (ini yg akan diketik user):")
    for slot in range(8):
        print(f"  slot {slot} ({CORNER_NAMES[slot]:>3}): {colors[slot]}")

    result = colors_to_solution(colors)
    print("\n--- HASIL ---")
    print(f"Butuh reorientasi? {result['needs_reorientation']} "
          f"({len(result['reorientation_steps'])} langkah putar 90 derajat)")
    print("Solusi:", result["moves_str"])

    # Verifikasi: solusi dr atas beneran menyelesaikan kubus (dlm kerangka
    # kanonik -- persis seperti yg akan dilakukan setelah reorientasi fisik)
    canonical, _, _, _ = canonicalize(held)
    final = apply_moves(canonical, result["moves_str"]) if result["moves"] else canonical
    print("Terverifikasi solved:", final.is_solved())
