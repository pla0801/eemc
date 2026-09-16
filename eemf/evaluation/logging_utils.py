from __future__ import annotations

import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_log_line(log_file, message: str) -> None:
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{timestamp()}] {message}\n")


def append_log_header(log_file) -> None:
    append_log_line(log_file, "run started")


def append_log_footer(log_file, status: str) -> None:
    append_log_line(log_file, f"run finished: {status}")


@contextmanager
def redirect_to_log(log_file, tee_stdout: bool = False):
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        out = Tee(sys.stdout, f) if tee_stdout else f
        err = Tee(sys.stderr, f) if tee_stdout else f
        with redirect_stdout(out), redirect_stderr(err):
            yield
