"""
app.py
=================================================================
Step 7: Controller (Flask).

Tugasnya SEMPIT dan SENGAJA dibuat begitu, sesuai rencana awal:
  - Menyajikan halaman (View) di GET /
  - Menerima input warna dari View, meneruskannya ke Model
    (color_mapping.colors_to_solution -> solver.py -> movements.py)
  - Mengembalikan hasilnya (teks solusi) sebagai JSON ke View

Tidak ada logika kubus di file ini sama sekali -- semua logika sudah
ada & sudah diuji tuntas di Step 1-6 (cube_model, movements, solver,
color_mapping). File ini murni "jembatan" HTTP.

Cara menjalankan:
    pip install flask
    python3 app.py
lalu buka http://127.0.0.1:5000 di browser.
"""

import random

from flask import Flask, jsonify, render_template, request

from cube_model import CORNER_NAMES, DEFAULT_FACE_COLORS, SOLVED_STATE
from color_mapping import ColorMappingError, colors_to_solution, state_to_colors, apply_rotation, ROTATIONS
from movements import ALL_MOVES, apply_moves

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/solve", methods=["POST"])
def api_solve():
    payload = request.get_json(silent=True) or {}
    raw_corners = payload.get("corners")

    if not isinstance(raw_corners, dict):
        return jsonify({"error": "Format input tidak valid: 'corners' harus berupa objek."}), 400

    try:
        corners = {int(slot): tuple(colors) for slot, colors in raw_corners.items()}
    except (TypeError, ValueError):
        return jsonify({"error": "Format input tidak valid: slot harus angka, warna harus daftar 3 item."}), 400

    try:
        result = colors_to_solution(corners)
    except ColorMappingError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # jaring pengaman -- jangan sampai user lihat traceback mentah
        return jsonify({"error": f"Terjadi kesalahan tak terduga: {e}"}), 500

    return jsonify(result)


@app.route("/api/example", methods=["GET"])
def api_example():
    """Isi contoh: kubus diacak + disimulasikan dipegang pd orientasi acak."""
    n = random.randint(3, 12)
    seq = [random.choice(ALL_MOVES) for _ in range(n)]
    scrambled = apply_moves(SOLVED_STATE, " ".join(seq))
    held = apply_rotation(scrambled, random.choice(ROTATIONS))
    colors = state_to_colors(held)
    return jsonify({
        "corners": {str(slot): list(c) for slot, c in colors.items()},
        "scramble_used": " ".join(seq),  # cuma info debug, tdk dipakai logika solve
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
