"""Run the MediaIndex server with a Python audit hook that logs every outgoing socket connection and DNS
lookup made by the process (diagnostics for the offline audit; not used in normal operation).

Usage: MEDIAINDEX_SOCKET_LOG=/path/log.txt .venv/bin/python scripts/run_with_socket_audit.py
"""

import os
import runpy
import sys
import time

LOG = open(os.environ.get("MEDIAINDEX_SOCKET_LOG", "socket-audit.log"), "a", buffering=1)


def hook(event, args):
    if event in ("socket.connect", "socket.getaddrinfo", "socket.sendto"):
        target = args[1] if event == "socket.connect" else args
        LOG.write(f"{time.strftime('%H:%M:%S')} {event} {target!r}\n")


sys.addaudithook(hook)
sys.argv = ["mediaindex"]
runpy.run_module("mediaindex", run_name="__main__", alter_sys=True)
