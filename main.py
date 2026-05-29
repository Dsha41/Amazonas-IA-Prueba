"""
main.py - Punto de entrada del servidor SERVER-IA Amazonas.

Arranca uvicorn directamente en foreground exponiendo el WebSocket
/ws/Personal%20de%20Amazonas en el puerto 9000.

Uso:
    python main.py
    python main.py --port 9001 --host 127.0.0.1

Equivalente directo de uvicorn:
    uvicorn src.app.app:app --host 0.0.0.0 --port 9000
"""

import argparse
import sys

import uvicorn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SERVER-IA Amazonas - WebSocket server"
    )
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host de escucha (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=9000,
                        help="Puerto de escucha (default: 9000)")
    parser.add_argument("--reload", action="store_true",
                        help="Recarga automatica en cambios (desarrollo)")
    parser.add_argument("--log-level", default="info",
                        choices=["critical", "error", "warning",
                                 "info", "debug", "trace"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[*] Iniciando SERVER-IA en http://{args.host}:{args.port}")
    print(f"[*] WebSocket: ws://{args.host}:{args.port}/ws/Personal%20de%20Amazonas")
    uvicorn.run(
        "src.app.app:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
