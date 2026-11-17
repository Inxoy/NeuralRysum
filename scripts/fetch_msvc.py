"""Fetch the MSVC x64 compiler and Windows SDK into a folder, without admin rights.

    python scripts/fetch_msvc.py .toolchain

Reads Microsoft's Visual Studio 2022 release manifest, downloads the compiler
(.vsix) and SDK (.msi) packages from Microsoft's servers, checks every file's
SHA-256 against the manifest and unpacks them. Nothing is installed system-wide;
delete the folder to remove it. Writes ``env.json`` with INCLUDE / LIB / PATH.

Use of these tools is subject to the Visual Studio licence terms. A regular
Visual Studio 2022 install makes this folder unnecessary.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

SDK = "22621"
SDK_MSIS = [
    "Windows SDK for Windows Store Apps Tools-x86_en-us.msi",
    "Windows SDK for Windows Store Apps Headers-x86_en-us.msi",
    "Windows SDK for Windows Store Apps Libs-x86_en-us.msi",
    "Windows SDK Desktop Headers x86-x86_en-us.msi",
    "Windows SDK Desktop Headers x64-x86_en-us.msi",
    "Windows SDK Desktop Libs x64-x86_en-us.msi",
    "Universal CRT Headers Libraries and Sources-x86_en-us.msi",
]

root = Path(sys.argv[1]).resolve()
dl = root / "dl"
dl.mkdir(parents=True, exist_ok=True)


def fetch(url: str, dest: Path, sha256: str | None = None) -> Path:
    if dest.exists() and (sha256 is None or hashlib.sha256(dest.read_bytes()).hexdigest() == sha256.lower()):
        return dest
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
            break
        except Exception as e:  # noqa: BLE001
            print(f"  retry {attempt + 1}: {e}")
    else:
        raise SystemExit(f"download failed: {url}")
    if sha256 and hashlib.sha256(data).hexdigest() != sha256.lower():
        raise SystemExit(f"SHA-256 mismatch for {dest.name}")
    dest.write_bytes(data)
    return dest


chan = json.loads(fetch("https://aka.ms/vs/17/release/channel", dl / "channel.json").read_text("utf-8"))
item = next(i for i in chan["channelItems"] if i["id"] == "Microsoft.VisualStudio.Manifests.VisualStudio")
man = json.loads(fetch(item["payloads"][0]["url"], dl / "vsman.json").read_text("utf-8"))
packages: dict[str, list] = {}
for p in man["packages"]:
    packages.setdefault(p["id"].lower(), []).append(p)

vers = [m.group(1) for i in packages if (m := re.match(r"^microsoft\.vc\.([\d.]+)\.tools\.hostx64\.targetx64\.base$", i))]
v = sorted(vers, key=lambda s: [int(x) for x in s.split(".")])[-1]
print("MSVC package version", v)

# --- compiler, linker, C++ standard library ---------------------------------
msvc_ids = [
    f"microsoft.vc.{v}.tools.hostx64.targetx64.base",
    f"microsoft.vc.{v}.tools.hostx64.targetx64.res.base",
    f"microsoft.vc.{v}.crt.headers.base",
    f"microsoft.vc.{v}.crt.x64.desktop.base",
    f"microsoft.vc.{v}.crt.x64.store.base",
]
for pid in msvc_ids:
    cands = packages[pid]
    pkg = next((p for p in cands if p.get("language") in (None, "en-US")), cands[0])
    for pay in pkg["payloads"]:
        f = fetch(pay["url"], dl / f"{pid}.{pay['fileName'].replace('/', '_')}", pay["sha256"])
        print("  got", pid, f.stat().st_size // 2**20, "MB")
        with zipfile.ZipFile(f) as z:
            for name in z.namelist():
                if name.startswith("Contents/") and not name.endswith("/"):
                    out = root / "msvc" / urllib.request.url2pathname(name[len("Contents/"):])
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(z.read(name))

# --- Windows SDK -------------------------------------------------------------
sdk_pkg = packages[f"win11sdk_10.0.{SDK}"][0]
pays = {p["fileName"].replace("/", "\\").split("\\")[-1].lower(): p for p in sdk_pkg["payloads"]}
sdk_dl = dl / "sdk"
sdk_dl.mkdir(exist_ok=True)
for name in SDK_MSIS:
    pay = pays[name.lower()]
    msi = fetch(pay["url"], sdk_dl / name, pay["sha256"])
    cabs = sorted(set(re.findall(rb"[0-9a-f]{32}\.cab", msi.read_bytes())))
    for cab in cabs:
        c = pays[cab.decode()]
        fetch(c["url"], sdk_dl / cab.decode(), c["sha256"])
    print(f"  got {name} (+{len(cabs)} cabs)")
    r = subprocess.run(["msiexec.exe", "/a", str(msi), "/quiet", "/qn", f"TARGETDIR={root / 'sdk'}"])
    if r.returncode:
        raise SystemExit(f"msiexec failed ({r.returncode}) on {name}")

# --- environment -------------------------------------------------------------
vc = next((root / "msvc" / "VC" / "Tools" / "MSVC").iterdir())
kit = root / "sdk" / "Windows Kits" / "10"
sv = next(d.name for d in (kit / "Include").iterdir() if d.name.startswith("10."))
env = {
    "INCLUDE": [str(vc / "include")] + [str(kit / "Include" / sv / d) for d in ("ucrt", "shared", "um", "winrt", "cppwinrt")],
    "LIB": [str(vc / "lib" / "x64"), str(kit / "Lib" / sv / "ucrt" / "x64"), str(kit / "Lib" / sv / "um" / "x64")],
    "PATH": [str(vc / "bin" / "Hostx64" / "x64"), str(kit / "bin" / sv / "x64")],
}
missing = [p for ps in env.values() for p in ps if not Path(p).exists()]
(root / "env.json").write_text(json.dumps(env, indent=2))
print("compiler", vc.name, "| SDK", sv)
print("missing paths:", missing or "none")
