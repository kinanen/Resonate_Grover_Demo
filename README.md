# Quantum Oracle Game

A browser-based game built on a real quantum simulator. The backend runs Grover's algorithm on 5 qubits and streams measurement results live to the frontend. Your job is to figure out which state the quantum oracle is amplifying — just by watching the histogram fill up.

## How it works

Grover's algorithm is a quantum search algorithm that amplifies the probability of one marked ("target") basis state. With 5 qubits there are 32 possible states (`|00000⟩` to `|11111⟩`). After 4 Grover iterations the target state has roughly 96% probability of being measured — so it accumulates counts much faster than the others.

Each round picks a random target. Watch the histogram and click the bar you think is being amplified. Wrong guesses gray out and trigger a 4-second penalty before you can guess again.

## Installation & startup

**Requirements:** Python 3.10+ and pip.

1. Clone or download the project folder.

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the server:
   ```bash
   python3 server.py
   ```

4. Open **http://localhost:8080** in your browser.

The server runs two services on startup:
- HTTP server on port `8080` — serves the frontend
- WebSocket server on port `8765` — streams shot data to the browser

To stop the server, press `Ctrl+C` in the terminal.

## Controls

| Control | Description |
|---|---|
| **Start** | Pick a new random target and begin streaming shots |
| **Pause** | Freeze the stream (counts are preserved) |
| **Stop** | Halt and clear all counts |
| **Gate noise** | Add depolarizing noise to the circuit — degrades the quantum advantage and makes the game harder |

## Noise model

The gate noise slider applies depolarizing noise after every gate layer: each qubit independently receives a random Pauli error (X, Y, or Z) with the given probability. At 0% the target state dominates clearly. Crank it up and the histogram flattens toward uniform — the quantum signal washes out.

## Stack

- **Backend:** Python, NumPy (state-vector simulator), `websockets`
- **Frontend:** Vanilla JS, Chart.js
- **Transport:** WebSocket (`ws://localhost:8765`), HTTP (`http://localhost:8080`)
