"""
Grover's Algorithm Quantum Simulator
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

_H1 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULIS = [_X, _Y, _Z]


def _kron_n(gate, n):
    result = gate
    for _ in range(n - 1):
        result = np.kron(result, gate)
    return result


def _make_H_N(n_qubits):
    return _kron_n(_H1, n_qubits)


def _make_pauli_ops(n_qubits):
    ops_list = []
    for q in range(n_qubits):
        ops = []
        for P in _PAULIS:
            parts = [np.eye(2, dtype=complex)] * n_qubits
            parts[q] = P
            mat = parts[0]
            for p in parts[1:]:
                mat = np.kron(mat, p)
            ops.append(mat)
        ops_list.append(ops)
    return ops_list


def _apply_depolarizing(state, noise_p, pauli_ops):
    for q_ops in pauli_ops:
        if np.random.random() < noise_p:
            state = q_ops[np.random.randint(3)] @ state
    return state


def run_circuit(target: int, n_qubits: int, n_iter: int, noise_p: float = 0.0) -> int:
    n_states = 2 ** n_qubits
    H_N = _make_H_N(n_qubits)
    pauli_ops = _make_pauli_ops(n_qubits) if noise_p > 0 else None

    state = np.zeros(n_states, dtype=complex)
    state[0] = 1.0
    state = H_N @ state
    if noise_p > 0:
        state = _apply_depolarizing(state, noise_p, pauli_ops)

    oracle = np.eye(n_states, dtype=complex)
    oracle[target, target] = -1.0
    s = H_N[:, 0]
    diffusion = 2.0 * np.outer(s, s.conj()) - np.eye(n_states, dtype=complex)

    for _ in range(n_iter):
        state = oracle @ state
        if noise_p > 0:
            state = _apply_depolarizing(state, noise_p, pauli_ops)
        state = diffusion @ state
        if noise_p > 0:
            state = _apply_depolarizing(state, noise_p, pauli_ops)

    probs = np.abs(state) ** 2
    probs /= probs.sum()
    return int(np.random.choice(n_states, p=probs))


def optimal_iters(n_qubits):
    return max(1, round(np.pi / 4 * np.sqrt(2 ** n_qubits)))


# ── WebSocket handler ─────────────────────────────────────────────────────────

DEFAULT_N_QUBITS = 5
SHOT_INTERVAL    = 0.06   # seconds between shots
MAX_NOISE        = 0.5


async def handler(websocket):
    n_qubits = DEFAULT_N_QUBITS
    n_states = 2 ** n_qubits
    n_iter   = optimal_iters(n_qubits)
    target   = np.random.randint(n_states)
    noise_p  = 0.0
    counts: dict[str, int] = {}
    total    = 0
    status   = "running"

    async def send_init():
        labels = [format(i, f"0{n_qubits}b") for i in range(n_states)]
        await websocket.send(json.dumps({
            "type":     "init",
            "labels":   labels,
            "n_qubits": n_qubits,
            "n_iter":   n_iter,
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
                        target = np.random.randint(n_states)
                        counts = {}
                        total  = 0
                        status = "running"
                        await send_init()
                    elif cmd == "pause":
                        status = "paused"
                        await websocket.send(json.dumps({"type": "status", "status": status}))
                    elif cmd == "stop":
                        status = "stopped"
                        counts = {}
                        total  = 0
                        await send_init()
                    continue

                if "guess" in data:
                    guessed    = data["guess"]
                    correct_bs = format(target, f"0{n_qubits}b")
                    correct    = (guessed == correct_bs)
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

                if "n_qubits" in data:
                    new_q = int(max(2, min(7, data["n_qubits"])))
                    if new_q != n_qubits:
                        n_qubits = new_q
                        n_states = 2 ** n_qubits
                        n_iter   = optimal_iters(n_qubits)
                        target   = np.random.randint(n_states)
                        counts   = {}
                        total    = 0
                        await send_init()
                    continue

                if "n_iter" in data:
                    new_iter = int(max(1, min(12, data["n_iter"])))
                    if new_iter != n_iter:
                        n_iter = new_iter
                        counts = {}
                        total  = 0
                        await send_init()
                    continue

            except asyncio.TimeoutError:
                pass   # interval elapsed — fire a shot

            if status != "running":
                continue

            result    = run_circuit(target, n_qubits, n_iter, noise_p)
            bitstring = format(result, f"0{n_qubits}b")
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
    def log_message(self, *_):
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
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
