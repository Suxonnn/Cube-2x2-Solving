"""
test_color_mapping.py
=================================================================
Unit test untuk color_mapping.py (Step 6).

Jalankan: python3 -m unittest test_color_mapping.py -v
"""

import random
import unittest

import color_mapping as cm
from cube_model import SOLVED_STATE, is_valid_state
from movements import ALL_MOVES, apply_moves
import solver


class TestRotationGroup(unittest.TestCase):

    def test_exactly_24_unique_rotations(self):
        basis = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        sigs = {tuple(r(b) for b in basis) for r in cm.ROTATIONS}
        self.assertEqual(len(cm.ROTATIONS), 24)
        self.assertEqual(len(sigs), 24)

    def test_identity_present(self):
        self.assertTrue(any(r((1, 2, 3)) == (1, 2, 3) for r in cm.ROTATIONS))

    def test_rotations_map_corners_to_corners(self):
        # Tiap rotasi harus memetakan 8 koordinat corner ke 8 koordinat
        # corner lain juga (bukan ke titik di luar himpunan itu).
        coords = set(cm.CORNER_COORD.values())
        for r in cm.ROTATIONS:
            for c in coords:
                self.assertIn(r(c), coords)

    def test_transitivity_every_slot_orient_has_exactly_one_rotation(self):
        random.seed(0)
        for target_slot in range(8):
            for target_orient in range(3):
                perm = [i for i in range(8) if i != 6]
                random.shuffle(perm)
                perm.insert(target_slot, 6)
                others = [p for p in range(8) if p != target_slot]
                orient = [0] * 8
                orient[target_slot] = target_orient
                for p in others[:-1]:
                    orient[p] = random.randint(0, 2)
                total = sum(orient[p] for p in others[:-1]) + target_orient
                orient[others[-1]] = (-total) % 3

                st = solver.CubeState(tuple(perm), tuple(orient))
                self.assertTrue(is_valid_state(st.perm, st.orient))

                matches = [
                    r for r in cm.ROTATIONS
                    if solver.is_encodable(cm.apply_rotation(st, r))
                ]
                self.assertEqual(
                    len(matches), 1,
                    msg=f"slot={target_slot} orient={target_orient} -> {len(matches)} match",
                )

    def test_apply_rotation_identity_is_noop(self):
        identity = next(r for r in cm.ROTATIONS if r((1, 2, 3)) == (1, 2, 3))
        result = cm.apply_rotation(SOLVED_STATE, identity)
        self.assertEqual(result, SOLVED_STATE)


class TestValidation(unittest.TestCase):

    def setUp(self):
        self.solved_colors = cm.state_to_colors(SOLVED_STATE)

    def test_solved_cube_passes(self):
        cm.validate_raw_input(self.solved_colors)  # tidak boleh raise

    def test_rejects_missing_slot(self):
        bad = dict(self.solved_colors)
        del bad[7]
        with self.assertRaises(cm.ColorMappingError):
            cm.validate_raw_input(bad)

    def test_rejects_unknown_color(self):
        bad = dict(self.solved_colors)
        bad[0] = ("Purple", "Red", "Green")
        with self.assertRaises(cm.ColorMappingError):
            cm.validate_raw_input(bad)

    def test_rejects_duplicate_color_within_corner(self):
        bad = dict(self.solved_colors)
        bad[0] = ("White", "White", "Green")
        with self.assertRaises(cm.ColorMappingError):
            cm.validate_raw_input(bad)

    def test_rejects_opposite_color_pair_red_orange(self):
        bad = dict(self.solved_colors)
        bad[0] = ("White", "Red", "Orange")
        with self.assertRaises(cm.ColorMappingError):
            cm.validate_raw_input(bad)

    def test_rejects_opposite_color_pair_white_yellow(self):
        bad = dict(self.solved_colors)
        bad[0] = ("White", "Yellow", "Green")
        with self.assertRaises(cm.ColorMappingError):
            cm.validate_raw_input(bad)

    def test_rejects_duplicate_corner_across_slots(self):
        bad = dict(self.solved_colors)
        bad[0] = self.solved_colors[1]
        with self.assertRaises(cm.ColorMappingError):
            cm.validate_raw_input(bad)

    def test_rejects_wrong_color_count(self):
        bad = dict(self.solved_colors)
        bad[0] = ("White", "Red", "Orange")  # ganti tanpa jaga hitungan warna
        with self.assertRaises(cm.ColorMappingError):
            cm.validate_raw_input(bad)


class TestRawStateFromColors(unittest.TestCase):

    def test_solved_colors_give_solved_state(self):
        colors = cm.state_to_colors(SOLVED_STATE)
        st = cm.raw_state_from_colors(colors)
        self.assertEqual(st, SOLVED_STATE)

    def test_round_trip_state_to_colors_and_back_random_scrambles(self):
        random.seed(1)
        for _ in range(30):
            n = random.randint(0, 15)
            seq = [random.choice(ALL_MOVES) for _ in range(n)]
            st = apply_moves(SOLVED_STATE, " ".join(seq)) if seq else SOLVED_STATE
            colors = cm.state_to_colors(st)
            recovered = cm.raw_state_from_colors(colors)
            self.assertEqual(recovered, st)


class TestCanonicalize(unittest.TestCase):

    def test_canonicalized_state_is_encodable(self):
        random.seed(2)
        for _ in range(30):
            n = random.randint(0, 15)
            seq = [random.choice(ALL_MOVES) for _ in range(n)]
            st = apply_moves(SOLVED_STATE, " ".join(seq)) if seq else SOLVED_STATE
            held = cm.apply_rotation(st, random.choice(cm.ROTATIONS))
            canonical, _, _, _ = cm.canonicalize(held)
            self.assertTrue(solver.is_encodable(canonical))


class TestEndToEndPipeline(unittest.TestCase):
    """Tes paling penting: warna -> solve -> diterapkan -> harus solved."""

    def test_already_solved_detected(self):
        colors = cm.state_to_colors(SOLVED_STATE)
        result = cm.colors_to_solution(colors)
        self.assertTrue(result["already_solved"])
        self.assertEqual(result["moves"], [])

    def test_random_scrambles_with_random_holding_orientation(self):
        random.seed(2024)
        for _ in range(100):
            n = random.randint(0, 20)
            seq = [random.choice(ALL_MOVES) for _ in range(n)]
            scrambled = apply_moves(SOLVED_STATE, " ".join(seq)) if seq else SOLVED_STATE

            # simulasikan user memegang kubus pd orientasi SEMBARANG
            held = cm.apply_rotation(scrambled, random.choice(cm.ROTATIONS))
            colors = cm.state_to_colors(held)

            result = cm.colors_to_solution(colors)

            if result["already_solved"]:
                self.assertTrue(held.is_solved())
                continue

            canonical, _, _, _ = cm.canonicalize(held)
            final = (
                apply_moves(canonical, result["moves_str"])
                if result["moves"] else canonical
            )
            self.assertTrue(final.is_solved(), msg=f"scramble={seq} sol={result['moves']}")

    def test_reference_corner_already_placed_needs_no_instruction(self):
        # Kalau corner id 6 kebetulan SUDAH di slot 6 orient 0 (misal scramble
        # yg hanya pakai U,R,F), tidak perlu instruksi reorientasi.
        st = apply_moves(SOLVED_STATE, "R U2 F' R'")
        colors = cm.state_to_colors(st)
        result = cm.colors_to_solution(colors)
        self.assertFalse(result["needs_reorientation"])
        self.assertEqual(result["reorientation_steps"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
