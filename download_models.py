"""
One-time download of the local search models used by app.py (about 80 MB in total).

Run it once while you have internet access:
    python download_models.py

It ignores HF_HUB_OFFLINE for this run, stores the models in the same folder app.py reads
(FASTEMBED_CACHE_PATH, default: .fastembed_cache next to app.py), and reports clearly if
huggingface.co cannot be reached.
"""
import os
import sys
import urllib.error
import urllib.request

CACHE = os.getenv(
    "FASTEMBED_CACHE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".fastembed_cache"),
)

# Offline flags make fastembed look ONLY in the local cache and never download.
for flag in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    if os.environ.get(flag):
        print(f"Note: {flag}={os.environ[flag]!r} is set; ignoring it for this download.")
        os.environ.pop(flag)

host = os.getenv("HF_ENDPOINT", "https://huggingface.co")
try:
    urllib.request.urlopen(urllib.request.Request(host, method="HEAD"), timeout=15)
    print(f"OK: {host} is reachable.")
except urllib.error.HTTPError:
    print(f"OK: {host} responded (HTTP error codes here are fine).")
except Exception as e:
    print(f"ERROR: cannot reach {host}: {type(e).__name__}: {e}")
    print("Check your internet connection, VPN/proxy/firewall, or set HF_ENDPOINT to a reachable mirror.")
    sys.exit(1)

from fastembed import TextEmbedding, SparseTextEmbedding  # noqa: E402

os.makedirs(CACHE, exist_ok=True)
print(f"Downloading models into {CACHE} ...")
TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir=CACHE)
SparseTextEmbedding(model_name="Qdrant/bm25", cache_dir=CACHE)
print("Done. You can now run the app.")
