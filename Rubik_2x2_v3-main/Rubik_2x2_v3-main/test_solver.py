"""
test_solver.py
=================================================================
Unit test untuk cube_model.py, movements.py, dan solver.py.

Jalankan salah satu:
    python3 -m unittest test_solver.py -v
    python3 test_solver.py

Catatan: TestSolverBFSTable memakai tabel BFS lengkap (get_table()).
Kalau bfs_table.bin belum ada, test ini akan MEMBANGUNNYA dulu
(~3-4 menit sekali jalan), lalu memakai file cache untuk run berikutnya.
"""

import random
import unittest

from cube_model import (
    CubeState,
    SOLVED_STATE,
    SOLVED_PERM,
    SOLVED_ORIENT,
    is_valid_state,
    CORNER_NAMES,
)
from movements import (
    ALL_MOVES,
    apply_move,
    apply_moves,
    parse_moves,
    inverse_move,
    inverse_moves,
)
import solver


# =================================================================
# cube_model.py
# =================================================================

class TestCubeModel(unittest.TestCase):

    def test_solved_state_is_solved(self):
        self.assertTrue(SOLVED_STATE.is_solved())

    def test_solved_state_is_valid(self):
        self.assertTrue(is_valid_state(SOLVED_PERM, SOLVED_ORIENT))

    def test_as_key_hashable(self):
        s = {SOLVED_STATE.as_key()}
        self.assertIn(SOLVED_STATE.as_key(), s)

    def test_invalid_permutation_rejected(self):
        bad_perm = (0, 0, 2, 3, 4, 5, 6, 7)  # 0 dobel, 1 hilang
        self.assertFalse(is_valid_state(bad_perm, SOLVED_ORIENT))

    def test_invalid_orientation_value_rejected(self):
        bad_orient = (3, 0, 0, 0, 0, 0, 0, 0)  # 3 bukan 0/1/2
        self.assertFalse(is_valid_state(SOLVED_PERM, bad_orient))

    def test_orientation_sum_not_multiple_of_3_rejected(self):
        bad_orient = (1, 0, 0, 0, 0, 0, 0, 0)  # sum = 1
        self.assertFalse(is_valid_state(SOLVED_PERM, bad_orient))

    def test_orientation_sum_multiple_of_3_accepted(self):
        ok_orient = (1, 2, 0, 0, 0, 0, 0, 0)  # sum = 3
        self.assertTrue(is_valid_state(SOLVED_PERM, ok_orient))

    def test_corner_names_length(self):
        self.assertEqual(len(CORNER_NAMES), 8)


# =================================================================
# movements.py
# =================================================================

class TestMovements(unittest.TestCase):

    def test_nine_moves_defined(self):
        self.assertEqual(len(ALL_MOVES), 9)
        self.assertEqual(
            set(ALL_MOVES), {"U", "U'", "U2", "R", "R'", "R2", "F", "F'", "F2"}
        )

    def test_dbl_corner_never_moves(self):
        for m in ALL_MOVES:
            s = apply_move(SOLVED_STATE, m)
            self.assertEqual(s.perm[6], 6, msg=m)
            self.assertEqual(s.orient[6], 0, msg=m)

    def test_every_move_result_is_physically_valid(self):
        for m in ALL_MOVES:
            s = apply_move(SOLVED_STATE, m)
            self.assertTrue(is_valid_state(s.perm, s.orient), msg=m)

    def test_quarter_turn_four_times_is_identity(self):
        for face in "URF":
            s = SOLVED_STATE
            for _ in range(4):
                s = apply_move(s, face)
            self.assertTrue(s.is_solved(), msg=face)

    def test_double_turn_equals_two_quarter_turns(self):
        for face in "URF":
            twice = apply_moves(SOLVED_STATE, f"{face} {face}")
            double = apply_move(SOLVED_STATE, f"{face}2")
            self.assertEqual(twice, double, msg=face)

    def test_move_then_inverse_is_identity(self):
        for m in ALL_MOVES:
            s = apply_moves(SOLVED_STATE, f"{m} {inverse_move(m)}")
            self.assertTrue(s.is_solved(), msg=m)

    def test_inverse_move_table(self):
        self.assertEqual(inverse_move("R"), "R'")
        self.assertEqual(inverse_move("R'"), "R")
        self.assertEqual(inverse_move("R2"), "R2")

    def test_parse_moves_valid(self):
        self.assertEqual(parse_moves("R U2 F'"), ["R", "U2", "F'"])

    def test_parse_moves_invalid_token_raises(self):
        with self.assertRaises(ValueError):
            parse_moves("R X U2")

    def test_apply_move_unknown_raises(self):
        with self.assertRaises(ValueError):
            apply_move(SOLVED_STATE, "D")  # D bukan bagian dari 9 gerakan ini

    def test_inverse_moves_reverses_sequence(self):
        seq = "R U2 F' R' U F2 R"
        s = apply_moves(SOLVED_STATE, seq)
        back = apply_moves(s, inverse_moves(seq))
        self.assertTrue(back.is_solved())

    def test_sexy_move_order_is_six(self):
        s = SOLVED_STATE
        for i in range(1, 6):
            s = apply_moves(s, "R U R' U'")
            self.assertFalse(s.is_solved(), msg=f"iterasi {i} seharusnya belum solved")
        s = apply_moves(s, "R U R' U'")
        self.assertTrue(s.is_solved(), msg="iterasi ke-6 harus solved")

    def test_bfs_new_states_per_depth_matches_known_values(self):
        # Jumlah state BARU (bukan kumulatif) per kedalaman, hasil BFS 3 muka
        # U,R,F. Konsisten dengan total akhir 3.674.160 pada kedalaman 11
        # yang diverifikasi penuh di TestSolverBFSTable.
        expected_new_per_depth = [9, 54, 321, 1847]  # depth 1..4 (ringan)
        seen = {SOLVED_STATE.as_key()}
        frontier = [SOLVED_STATE]
        got = []
        for _ in range(len(expected_new_per_depth)):
            nxt = []
            for st in frontier:
                for m in ALL_MOVES:
                    ns = apply_move(st, m)
                    k = ns.as_key()
                    if k not in seen:
                        seen.add(k)
                        nxt.append(ns)
            got.append(len(nxt))
            frontier = nxt
        self.assertEqual(got, expected_new_per_depth)


# =================================================================
# solver.py -- encoding
# =================================================================

class TestSolverEncoding(unittest.TestCase):

    def test_table_size_matches_known_cube_count(self):
        # 8! x 3^7 / 24 = jumlah baku posisi kubus 2x2x2 (fakta matematis,
        # dihitung independen dari implementasi BFS).
        self.assertEqual(solver.TABLE_SIZE, 3_674_160)

    def test_encode_decode_round_trip_random_sample(self):
        random.seed(123)
        for _ in range(3000):
            idx = random.randrange(solver.TABLE_SIZE)
            st = solver.decode_state(idx)
            self.assertEqual(solver.encode_state(st), idx)
            self.assertTrue(is_valid_state(st.perm, st.orient))
            self.assertEqual(st.perm[6], 6)
            self.assertEqual(st.orient[6], 0)

    def test_encode_decode_solved_state(self):
        idx = solver.encode_state(SOLVED_STATE)
        self.assertEqual(solver.decode_state(idx), SOLVED_STATE)

    def test_is_encodable_accepts_solved(self):
        self.assertTrue(solver.is_encodable(SOLVED_STATE))

    def test_is_encodable_rejects_non_canonical(self):
        # Corner 6 berada di slot 0, bukan slot 6 -- valid secara fisik,
        # tapi di luar konvensi modul ini.
        bad = CubeState((6, 1, 2, 3, 4, 5, 0, 7), SOLVED_ORIENT)
        self.assertTrue(is_valid_state(bad.perm, bad.orient))
        self.assertFalse(solver.is_encodable(bad))

    def test_is_encodable_rejects_physically_invalid(self):
        bad = CubeState((0, 0, 2, 3, 4, 5, 6, 7), SOLVED_ORIENT)
        self.assertFalse(solver.is_encodable(bad))


# =================================================================
# solver.py -- tabel BFS & solve()
# =================================================================

class TestSolverBFSTable(unittest.TestCase):
    """Memakai tabel BFS lengkap (get_table() -- otomatis cache ke file)."""

    @classmethod
    def setUpClass(cls):
        cls.table = solver.get_table(verbose=False)

    def test_table_length(self):
        self.assertEqual(len(self.table), solver.TABLE_SIZE)

    def test_no_unvisited_states(self):
        # Setiap entri tabel sudah terisi -> bukti seluruh 3,67 juta state
        # benar-benar reachable dari solved lewat 9 gerakan ini.
        self.assertNotIn(solver.UNVISITED, self.table)

    def test_solved_state_solution_is_empty(self):
        self.assertEqual(solver.solve(SOLVED_STATE, self.table), [])

    def test_single_move_scrambles_solve_in_one_move(self):
        for m in ALL_MOVES:
            scrambled = apply_move(SOLVED_STATE, m)
            sol = solver.solve(scrambled, self.table)
            self.assertEqual(len(sol), 1, msg=m)
            self.assertTrue(apply_moves(scrambled, sol[0]).is_solved(), msg=m)

    def test_random_scrambles_are_correctly_solved(self):
        random.seed(1)
        for _ in range(50):
            n = random.randint(0, 20)
            seq = [random.choice(ALL_MOVES) for _ in range(n)]
            scrambled = apply_moves(SOLVED_STATE, " ".join(seq)) if seq else SOLVED_STATE
            sol = solver.solve(scrambled, self.table)
            result = apply_moves(scrambled, " ".join(sol)) if sol else scrambled
            self.assertTrue(result.is_solved(), msg=f"scramble={seq} sol={sol}")

    def test_solution_optimality_vs_brute_force(self):
        random.seed(7)
        for _ in range(20):
            n = random.randint(1, 5)
            seq = [random.choice(ALL_MOVES) for _ in range(n)]
            scrambled = apply_moves(SOLVED_STATE, " ".join(seq))
            sol = solver.solve(scrambled, self.table)
            brute = self._brute_shortest(scrambled, max_len=6)
            self.assertIsNotNone(brute, msg=f"brute-force gagal utk {seq}")
            self.assertEqual(len(sol), len(brute), msg=f"scramble={seq} sol={sol} brute={brute}")

    def test_max_solution_length_is_eleven(self):
        # God's number utk 2x2x2 pada metrik 9-gerakan (U,R,F HTM) = 11,
        # sudah dibuktikan penuh saat build tabel (kedalaman maksimum BFS).
        # Di sini kita cek sampel acak tidak pernah melebihi itu.
        random.seed(99)
        max_len_found = 0
        for _ in range(200):
            n = random.randint(0, 25)
            seq = [random.choice(ALL_MOVES) for _ in range(n)]
            scrambled = apply_moves(SOLVED_STATE, " ".join(seq)) if seq else SOLVED_STATE
            sol = solver.solve(scrambled, self.table)
            max_len_found = max(max_len_found, len(sol))
        self.assertLessEqual(max_len_found, 11)

    def test_invalid_state_raises_value_error(self):
        bad = CubeState((0, 0, 2, 3, 4, 5, 6, 7), SOLVED_ORIENT)
        with self.assertRaises(ValueError):
            solver.solve(bad, self.table)

    def test_non_canonical_state_raises_value_error(self):
        bad = CubeState((6, 1, 2, 3, 4, 5, 0, 7), SOLVED_ORIENT)
        with self.assertRaises(ValueError):
            solver.solve(bad, self.table)

    @staticmethod
    def _brute_shortest(scrambled, max_len=6):
        if scrambled.is_solved():
            return []
        frontier = [(scrambled, [])]
        seen = {scrambled.as_key()}
        for _ in range(max_len):
            nxt = []
            for st, path in frontier:
                for mv in ALL_MOVES:
                    ns = apply_move(st, mv)
                    if ns.is_solved():
                        return path + [mv]
                    k = ns.as_key()
                    if k not in seen:
                        seen.add(k)
                        nxt.append((ns, path + [mv]))
            frontier = nxt
        return None


if __name__ == "__main__":
    unittest.main(verbosity=2)
