"""
Grover's Algorithm Quantum Simulator — 4 qubits
Serves the frontend on http://localhost:8080 and streams
measurement results via WebSocket on ws://localhost:8765
"""

import asyncio
import json
import os
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
import numpy as np
import websockets

# ── Quantum simulator ─────────────────────────────────────────────────────────

N_QUBITS = 5
N_STATES = 2 ** N_QUBITS   # 16

_H1 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULIS = [_X, _Y, _Z]


def _kron_n(gate, n):
    """n-fold tensor product of a single-qubit gate."""
    result = gate
    for _ in range(n - 1):
        result = np.kron(result, gate)
    return result


H_N = _kron_n(_H1, N_QUBITS)   # 16×16 Hadamard

# Precompute expanded Pauli operators for each qubit position
_PAULI_OPS = []   # _PAULI_OPS[qubit][pauli_idx] → 16×16 matrix
for _q in range(N_QUBITS):
    _ops = []
    for _P in _PAULIS:
        parts = [np.eye(2, dtype=complex)] * N_QUBITS
        parts[_q] = _P
        mat = parts[0]
        for _p in parts[1:]:
            mat = np.kron(mat, _p)
        _ops.append(mat)
    _PAULI_OPS.append(_ops)


def _apply_depolarizing(state: np.ndarray, noise_p: float) -> np.ndarray:
    """Depolarizing channel: apply a random Pauli to each qubit with prob noise_p."""
    for q in range(N_QUBITS):
        if np.random.random() < noise_p:
            state = _PAULI_OPS[q][np.random.randint(3)] @ state
    return state


def run_circuit(target: int, noise_p: float = 0.0) -> int:
    """
    Run one shot of Grover's circuit with depolarizing noise and return the
    measured basis-state index.  noise_p=0 → ideal (noiseless).
    """
    state = np.zeros(N_STATES, dtype=complex)
    state[0] = 1.0

    # Initial Hadamard layer
    state = H_N @ state
    if noise_p > 0:
        state = _apply_depolarizing(state, noise_p)

    n_iter = max(1, round(np.pi / 4 * np.sqrt(N_STATES)))

    oracle = np.eye(N_STATES, dtype=complex)
    oracle[target, target] = -1.0

    s = H_N[:, 0]
    diffusion = 2.0 * np.outer(s, s.conj()) - np.eye(N_STATES, dtype=complex)

    for _ in range(n_iter):
        state = oracle @ state
        if noise_p > 0:
            state = _apply_depolarizing(state, noise_p)
        state = diffusion @ state
        if noise_p > 0:
            state = _apply_depolarizing(state, noise_p)

    probs = np.abs(state) ** 2
    probs /= probs.sum()
    return int(np.random.choice(N_STATES, p=probs))


def ideal_probs(target: int) -> list[float]:
    """Theoretical (noiseless) probability distribution after Grover's algorithm."""
    state = np.zeros(N_STATES, dtype=complex)
    state[0] = 1.0
    state = H_N @ state
    n_iter = max(1, round(np.pi / 4 * np.sqrt(N_STATES)))
    oracle = np.eye(N_STATES, dtype=complex)
    oracle[target, target] = -1.0
    s = H_N[:, 0]
    diffusion = 2.0 * np.outer(s, s.conj()) - np.eye(N_STATES, dtype=complex)
    for _ in range(n_iter):
        state = oracle @ state
        state = diffusion @ state
    probs = np.abs(state) ** 2
    return (probs / probs.sum()).tolist()


# ── WebSocket handler ─────────────────────────────────────────────────────────

DEFAULT_TARGET  = 10             # |1010>
SHOT_INTERVAL   = 0.06           # seconds between shots
MAX_NOISE       = 0.5            # clamp incoming noise values


async def handler(websocket):
    target  = DEFAULT_TARGET
    noise_p = 0.0
    counts: dict[str, int] = {}
    total   = 0
    status  = "running"   # "running" | "paused" | "stopped"

    async def send_init():
        labels = [format(i, f"0{N_QUBITS}b") for i in range(N_STATES)]
        await websocket.send(json.dumps({
            "type":     "init",
            "labels":   labels,
            "n_qubits": N_QUBITS,
            "noise_p":  noise_p,
            "status":   status,
        }))

    await send_init()

    try:
        while True:
            timeout = SHOT_INTERVAL if status == "running" else None

            try:
                raw  = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                data = json.loads(raw)

                if "cmd" in data:
                    cmd = data["cmd"]
                    if cmd == "start":
                        target  = np.random.randint(N_STATES)
                        counts  = {}
                        total   = 0
                        status  = "running"
                        await send_init()
                    elif cmd == "pause":
                        status = "paused"
                        await websocket.send(json.dumps({"type": "status", "status": status}))
                    elif cmd == "stop":
                        status  = "stopped"
                        counts  = {}
                        total   = 0
                        await send_init()
                    continue

                if "guess" in data:
                    guessed = data["guess"]
                    correct_bs = format(target, f"0{N_QUBITS}b")
                    correct = (guessed == correct_bs)
                    if correct:
                        status = "paused"
                    await websocket.send(json.dumps({
                        "type":    "guess_result",
                        "correct": correct,
                        "guessed": guessed,
                        "target":  correct_bs if correct else None,
                        "status":  status,
                    }))
                    continue

                if "noise" in data:
                    new_noise = float(max(0.0, min(MAX_NOISE, data["noise"])))
                    if new_noise != noise_p:
                        noise_p = new_noise
                        counts  = {}
                        total   = 0
                        await send_init()
                continue

            except asyncio.TimeoutError:
                pass   # interval elapsed — fire a shot

            if status != "running":
                continue

            result    = run_circuit(target, noise_p)
            bitstring = format(result, f"0{N_QUBITS}b")
            counts[bitstring] = counts.get(bitstring, 0) + 1
            total += 1

            await websocket.send(json.dumps({
                "type":   "shot",
                "shot":   bitstring,
                "counts": counts,
                "total":  total,
            }))

    except websockets.exceptions.ConnectionClosed:
        pass


# ── HTTP server (serves index.html) ──────────────────────────────────────────

class _Handler(SimpleHTTPRequestHandler):
    def log_message(self, *_):   # silence request logs
        pass


def _run_http():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    httpd = HTTPServer(("localhost", 8080), _Handler)
    httpd.serve_forever()


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    threading.Thread(target=_run_http, daemon=True).start()
    print("Quantum simulator ready.")
    print("  Frontend  →  http://localhost:8080")
    print("  WebSocket →  ws://localhost:8765")
    print("  Target state: |1010⟩  (Grover iterations:", max(1, round(np.pi / 4 * np.sqrt(N_STATES))), ")")
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
