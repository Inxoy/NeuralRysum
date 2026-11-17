"""Download a public Nextcloud share (for example tubcloud) with all its subfolders.

    python scripts/fetch_nextcloud.py https://tubcloud.tu-berlin.de/s/<token> data/target [--workers 8]

Files that already exist with the right size are skipped, so rerunning resumes.
Each file is written to <name>.part first and renamed when complete.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

UA = {"User-Agent": "dataset-fetch/1.0"}
PROPFIND = (b'<?xml version="1.0"?><d:propfind xmlns:d="DAV:"><d:prop>'
            b"<d:getcontentlength/><d:resourcetype/></d:prop></d:propfind>")


def listing(base: str, token: str):
    req = urllib.request.Request(f"{base}/public.php/dav/files/{token}/", data=PROPFIND, method="PROPFIND",
                                 headers={**UA, "Depth": "infinity", "Content-Type": "application/xml"})
    with urllib.request.urlopen(req, timeout=120) as r:
        root = ET.fromstring(r.read())
    prefix = f"/public.php/dav/files/{token}/"
    out = []
    for resp in root.findall("{DAV:}response"):
        href = urllib.parse.unquote(resp.find("{DAV:}href").text)
        if resp.find(".//{DAV:}collection") is not None:
            continue
        rel = href.split(prefix, 1)[1]
        size = int(resp.find(".//{DAV:}getcontentlength").text)
        out.append((rel, size))
    return out


def fetch(base: str, token: str, rel: str, size: int, dst: Path) -> int:
    path = dst / rel
    if ".." in Path(rel).parts:
        raise ValueError(f"refusing path {rel}")
    if path.exists() and path.stat().st_size == size:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{base}/public.php/dav/files/{token}/" + urllib.parse.quote(rel)
    part = path.with_name(path.name + ".part")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, part.open("wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if part.stat().st_size != size:
                raise IOError(f"size {part.stat().st_size} != {size}")
            part.replace(path)
            return size
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                raise
            print(f"retry {rel}: {e}", flush=True)
            time.sleep(2 + 4 * attempt)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("share")
    ap.add_argument("dst")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    u = urllib.parse.urlparse(a.share)
    base, token = f"{u.scheme}://{u.netloc}", u.path.rstrip("/").split("/")[-1]
    dst = Path(a.dst)
    files = listing(base, token)
    total = sum(s for _, s in files)
    print(f"{len(files)} files, {total / 1e9:.2f} GB", flush=True)
    done, failed, t0 = 0, [], time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(fetch, base, token, rel, s, dst): (rel, s) for rel, s in files}
        for i, fu in enumerate(cf.as_completed(futs), 1):
            rel, s = futs[fu]
            try:
                fu.result()
                done += s
            except Exception as e:  # noqa: BLE001
                failed.append(rel)
                print(f"FAILED {rel}: {e}", flush=True)
            if i % 200 == 0 or i == len(files):
                mbps = done / 1e6 / max(time.time() - t0, 1e-3)
                print(f"{i}/{len(files)} files, {done / 1e9:.2f} GB, {mbps:.1f} MB/s", flush=True)
    print(f"done, {len(failed)} failed", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
