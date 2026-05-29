"""
server.py - Wrapper de uvicorn para arrancar/detener el servidor FastAPI
en un hilo separado. Permite integrar el servidor con una UI (PySide6)
o usarlo directamente desde un script.
"""

import asyncio
import threading
import time

from uvicorn.config import Config
from uvicorn.server import Server as UvicornServer


class Server_services:

    def __init__(self, port: int = 9000, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self.server: UvicornServer | None = None
        self.thread: threading.Thread | None = None
        self.should_exit = False
        self.config = Config(
            app=None,
            host=self.host,
            port=self.port,
            log_level="info",
        )

    def start(self) -> None:
        try:
            self.thread = threading.Thread(target=self._run_server)
            self.thread.daemon = True
            self.should_exit = False
            self.thread.start()
        except Exception as e:
            print(f"Error al iniciar el servidor: {e}")

    def stop(self) -> None:
        if self.server is not None:
            self.server.should_exit = True
            time.sleep(2)

    def _run_server(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            from .app import app
            self.config.app = app
            self.server = UvicornServer(self.config)
            loop.run_until_complete(self.server.serve())
        except Exception as e:
            print(f"Error en el servidor: {e}")
        finally:
            loop.close()
