# Development environment

Recorded 2026-10-06 on the **target laptop** (the user's local Mac, not a remote container).

## Hardware / OS
| Item | Value |
|---|---|
| Machine | Apple M1 (arm64), 8 CPU cores, 7-core integrated GPU (Metal 3) |
| RAM | 8 GB unified memory (shared between CPU and GPU, so there is no separate VRAM) |
| OS | macOS 14.0 (23A344) |
| Free disk | ~87 GB of 228 GB on the data volume |
| Accelerator | Apple MPS (support verified in Phase 1) |

8 GB of unified memory is a hard constraint. Model precision and batch sizes must be kept small.

## Toolchain
| Tool | Version | Path |
|---|---|---|
| Python (app venv) | 3.12.13 | `/opt/homebrew/bin/python3.12` via uv |
| uv | 0.9.2 | `~/.local/bin/uv` |
| Node | 22.23.1 | nvm |
| npm | bundled with Node 22 | |
| FFmpeg / ffprobe | 7.1.1 | `/opt/homebrew/bin` |
| git | 2.41.0 | Homebrew |

The default `python3` is Anaconda 3.11.5. The project does **not** use it; uv pins 3.12 via `.python-version`.

## Setup commands
```bash
uv python pin 3.12
uv sync                      # creates .venv and installs locked deps (uv.lock)
cd frontend && npm install   # installs locked deps (package-lock.json)
```

## Verify
```bash
uv run pytest -q                       # unit tests
uv run python -m mediaindex &          # backend on 127.0.0.1:8765
curl http://127.0.0.1:8765/api/health
cd frontend && npm run build
```
