"""
Step-by-step test of the connection to your Qdrant Cloud database (same settings as app.py).
Run:  python check_qdrant.py
It shows exactly which step breaks: address lookup, network connection, secure (TLS) connection,
the API key, or the Qdrant client itself.
"""
import os
import socket
import ssl
import sys
import warnings
from urllib.parse import urlparse

from dotenv import dotenv_values, find_dotenv, load_dotenv

warnings.simplefilter("ignore")

# Remember what the computer's environment already contains BEFORE .env is read:
# load_dotenv() never overwrites an existing variable, so a stale one silently beats .env.
OS_BEFORE = {name: os.environ.get(name) for name in ("QDRANT_URL", "QDRANT_API_KEY")}
ENV_PATH = find_dotenv()  # same lookup the app uses: .env next to this file, then parent folders
if ENV_PATH:
    load_dotenv(ENV_PATH)

TIMEOUT = 15


def step_dns(host):
    return sorted({info[4][0] for info in socket.getaddrinfo(host, None)})


def step_tcp(host, port):
    socket.create_connection((host, port), timeout=TIMEOUT).close()


def step_tls_and_http(host, port, key):
    """Opens a secure connection and asks Qdrant for its collections. Returns the HTTP status line."""
    ctx = ssl.create_default_context()
    with socket.create_connection((host, port), timeout=TIMEOUT) as raw:
        with ctx.wrap_socket(raw, server_hostname=host) as tls:
            request = f"GET /collections HTTP/1.1\r\nHost: {host}\r\napi-key: {key or ''}\r\nConnection: close\r\n\r\n"
            tls.sendall(request.encode())
            return tls.recv(200).decode(errors="replace").split("\r\n")[0] or "(empty reply)"


def run_step(label, fn, *args):
    try:
        result = fn(*args)
        print(f"  OK    {label}" + (f": {result}" if result else ""))
        return True, result
    except Exception as e:
        print(f"  FAIL  {label}: {type(e).__name__}: {e}")
        return False, None


def _host(value):
    return urlparse(value).hostname or value if value else "(not set)"


def _key_hint(value):
    return f"{len(value)} characters, ending in ...{value[-4:]}" if value else "(not set)"


def report_sources():
    """Shows which .env file is read and whether a stale system variable overrides it."""
    print("Where the settings come from:")
    folder = os.path.dirname(ENV_PATH) if ENV_PATH else os.path.dirname(os.path.abspath(__file__))
    if ENV_PATH:
        print(f"  .env file read : {ENV_PATH}")
    else:
        print("  .env file read : NONE FOUND next to this script or in its parent folders")
    lookalikes = [f for f in os.listdir(folder) if f.lower().startswith(".env") and f != ".env"]
    if lookalikes:
        print(f"  Files that look like .env in that folder: {lookalikes}  (Windows may hide '.txt' - the file must be named exactly .env)")

    file_values = dotenv_values(ENV_PATH) if ENV_PATH else {}
    conflict = False
    for name, describe in (("QDRANT_URL", _host), ("QDRANT_API_KEY", _key_hint)):
        in_file, in_os = file_values.get(name), OS_BEFORE[name]
        print(f"  {name}: .env has {describe(in_file)} | computer environment had {describe(in_os)}")
        if in_os and in_file and in_os != in_file:
            conflict = True
    if conflict:
        print("\n  PROBLEM FOUND: your computer/terminal already had its own QDRANT_URL or QDRANT_API_KEY that differs")
        print("  from your .env file. The app uses the computer's OLD value and ignores .env.")
        print("  Fix: remove them (PowerShell), then close and reopen the terminal / VS Code:")
        print("    Remove-Item Env:QDRANT_URL -ErrorAction SilentlyContinue")
        print("    Remove-Item Env:QDRANT_API_KEY -ErrorAction SilentlyContinue")
        print('    [Environment]::SetEnvironmentVariable("QDRANT_URL", $null, "User")')
        print('    [Environment]::SetEnvironmentVariable("QDRANT_API_KEY", $null, "User")')
    print()
    return conflict


def main():
    report_sources()
    url = os.getenv("QDRANT_URL")
    key = os.getenv("QDRANT_API_KEY")
    print("QDRANT_URL     :", url or "(NOT SET -> the app falls back to http://localhost:6333)")
    print("QDRANT_API_KEY :", f"set ({len(key)} characters)" if key else "(NOT SET)")
    proxies = {k: os.environ[k] for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy") if os.environ.get(k)}
    print("Proxy settings :", proxies or "none")
    if not url:
        print("\nProblem: QDRANT_URL is missing, so the app tries your own computer instead of the cloud.")
        print("Put QDRANT_URL=https://<your-cluster>.cloud.qdrant.io in the .env file next to app.py.")
        return 1
    if url.startswith("http://") and "cloud.qdrant.io" in url:
        print("\nNote: Qdrant Cloud needs https://, not http://.")

    parsed = urlparse(url)
    host, port = parsed.hostname, parsed.port or 6333  # the app's client also defaults to 6333
    print(f"\nTesting {host} on port {port} (the port the app uses):")

    ok, _ = run_step("1. address lookup (DNS)", step_dns, host)
    if not ok:
        print("\nMeaning: the address can't be found. Check QDRANT_URL for typos and your internet. "
              "A free cluster that was deleted after inactivity also ends up here.")
        return 1
    ok, _ = run_step(f"2. network connection to port {port}", step_tcp, host, port)
    if not ok:
        print(f"\nMeaning: your computer can't open a connection to port {port}. Likely a firewall, VPN, antivirus or "
              "network (office/college/ISP) blocking it, or the cluster is suspended. Try another network (a phone hotspot) "
              "and check the cluster status in the Qdrant Cloud dashboard.")
        return 1
    ok, status = run_step("3. secure connection + API key", step_tls_and_http, host, port, key)
    if not ok:
        print("\nMeaning: the connection opens but is cut off during the secure (TLS) step or right after.")
        print("Most common causes: (a) the cluster is suspended/paused or being restarted - check the Qdrant Cloud dashboard;")
        print("(b) antivirus/firewall/VPN that inspects HTTPS - try turning it off briefly or use a phone hotspot;")
        print("(c) a proxy. If the above step 2 worked but step 3 fails on every network, the cluster is the likely cause.")
        return 1
    if status and (" 401" in status or " 403" in status):
        print("\nMeaning: the cluster answered but rejected the API key. Copy a fresh key from the Qdrant Cloud dashboard.")
        return 1

    print("\n4. Qdrant client (what the app uses):")
    from qdrant_client import QdrantClient
    ok, names = run_step("list collections", lambda: [c.name for c in QdrantClient(url=url, api_key=key, timeout=30).get_collections().collections])
    if not ok:
        print("\nMeaning: steps 1-3 worked but the Qdrant client failed. A proxy setting (shown at the top) is the usual cause: "
              "try removing HTTPS_PROXY/HTTP_PROXY for this run, e.g. in PowerShell:  Remove-Item Env:HTTPS_PROXY")
        return 1
    print(f"\nSUCCESS: connected to Qdrant. Collections: {names or '(none yet)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())