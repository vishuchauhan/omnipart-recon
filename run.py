"""Start OmniPart RECON locally:  python run.py   ->  http://localhost:8000"""
import os, sys, webbrowser, threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

def load_env(path=Path(__file__).resolve().parent / ".env"):
    """Read KEY=VALUE lines from .env (kept only on this computer; never shared or zipped)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if v and "paste-your-key" not in v:
            os.environ.setdefault(k.strip(), v)


def free_port(start):
    """Use the first free port from `start` up (an older copy of the server may still hold 8000)."""
    import socket
    for p in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
            except OSError:
                print(f"Port {p} is busy (an older copy may still be running) - trying {p + 1}")
                continue
            return p
    raise SystemExit("No free port found between %d and %d" % (start, start + 19))


def main():
    load_env()
    from server import services
    if not (Path(__file__).parent / "app" / "bundle.json").exists():
        print("First run: generating demo data and running the engine (about 30 seconds)...")
        services.build_demo()
    import uvicorn
    port = free_port(int(os.getenv("PORT", "8000")))
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    print("AI drafting:", "ON" if os.getenv("ANTHROPIC_API_KEY") else "off (add your key to the .env file)")
    print(f"OmniPart RECON running at http://localhost:{port}   (API docs: http://localhost:{port}/docs)   Ctrl+C to stop")
    uvicorn.run("server.api:app", host="127.0.0.1", port=port, log_level="warning")

if __name__ == "__main__":
    main()
