"""Headless-browser screenshots for the TradingView chart pages.

The alert PNG is a screenshot of the SAME self-contained page the
HTML study shows (``tv_page``) — one render, two outputs, so the
two mediums can never drift apart again. Browsers are found by a
ladder, in order:

  (a) a system browser — PATH lookup (chromium / chrome / edge),
      the standard Edge + Chrome install paths on Windows (EC2),
      or the ``GS_CHART_BROWSER`` env override;
  (b) a self-provisioned Chromium — the Playwright browser cache
      or a tools-dir install; if the Playwright python package is
      present but no browser is, one bounded ``playwright install
      chromium`` attempt is made;
  (c) nothing — the caller falls back to the matplotlib renderer.
      An alert must never fail because the pretty rung is missing.

Every screenshot reports the rung that produced it
(``system:<exe>`` / ``provisioned:<exe>``) and callers stamp it
into the PNG's metadata (``stamp_png_text``), so a fallback render
is never mistaken for the TradingView render.
"""
from __future__ import annotations

import glob
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

_SYSTEM_NAMES = ("chromium", "chromium-browser", "google-chrome",
                 "google-chrome-stable", "microsoft-edge", "msedge",
                 "chrome")

_WINDOWS_KNOWN = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
)

_PLAYWRIGHT_PATTERNS = (
    "chromium_headless_shell-*/chrome-headless-shell-linux64/"
    "chrome-headless-shell",
    "chromium-*/chrome-linux/chrome",
    "chromium-*/chrome-mac/Chromium.app/Contents/MacOS/Chromium",
    "chromium-*/chrome-win/chrome.exe",
    "chromium_headless_shell-*/chrome-headless-shell-win/"
    "headless_shell.exe",
    "chromium-*/chrome-win/headless_shell.exe",
)


# ---------------------------------------------------------------- PNG text
def stamp_png_text(png: bytes, fields: dict) -> bytes:
    """Insert tEXt chunks right after IHDR (provenance record).

    Pure-python chunk surgery — works on any PNG (browser
    screenshots and matplotlib output alike). A non-PNG input is
    returned untouched.
    """
    if not png.startswith(PNG_SIGNATURE) or len(png) < 33:
        return png
    try:
        ihdr_len = struct.unpack(">I", png[8:12])[0]
        insert_at = 8 + 12 + ihdr_len
        chunks = b""
        for key, value in fields.items():
            data = (str(key).encode("latin-1") + b"\x00"
                    + str(value).encode("latin-1"))
            chunks += (struct.pack(">I", len(data)) + b"tEXt" + data
                       + struct.pack(
                           ">I",
                           zlib.crc32(b"tEXt" + data) & 0xFFFFFFFF))
        return png[:insert_at] + chunks + png[insert_at:]
    except Exception:
        return png


def read_png_text(png: bytes) -> dict:
    """The tEXt chunks of a PNG, as a dict (last value wins)."""
    out: dict = {}
    if not png.startswith(PNG_SIGNATURE):
        return out
    pos = 8
    while pos + 12 <= len(png):
        try:
            length = struct.unpack(">I", png[pos:pos + 4])[0]
            ctype = png[pos + 4:pos + 8]
        except Exception:
            break
        data = png[pos + 8:pos + 8 + length]
        if ctype == b"tEXt" and b"\x00" in data:
            key, _, value = data.partition(b"\x00")
            out[key.decode("latin-1")] = value.decode("latin-1")
        pos += 12 + length
        if ctype == b"IEND":
            break
    return out


# ---------------------------------------------------------------- the ladder
def _provisioned_candidates() -> list:
    out: list = []
    cache = Path.home() / ".cache" / "ms-playwright"
    for pattern in _PLAYWRIGHT_PATTERNS:
        out.extend(sorted(glob.glob(str(cache / pattern))))
    for root in (Path.home() / "workspace" / "tools",
                 Path.home() / "tools"):
        if root.is_dir():
            for pattern in ("**/chrome", "**/chrome.exe",
                            "**/headless_shell",
                            "**/chrome-headless-shell"):
                out.extend(sorted(
                    str(p) for p in root.glob(pattern) if p.is_file()))
    return out


def find_browser() -> tuple | None:
    """(exe_path, rung) — rung "system" or "provisioned"."""
    override = os.environ.get("GS_CHART_BROWSER")
    if override and Path(override).is_file():
        return override, "system"
    for name in _SYSTEM_NAMES:
        path = shutil.which(name)
        if path:
            return path, "system"
    for path in _WINDOWS_KNOWN:
        if Path(path).is_file():
            return path, "system"
    for path in _provisioned_candidates():
        if Path(path).is_file():
            return path, "provisioned"
    return None


def ensure_browser() -> tuple | None:
    """find_browser + one bounded self-provisioning attempt."""
    found = find_browser()
    if found is not None:
        return found
    try:
        subprocess.run(
            [sys.executable, "-m", "playwright", "install",
             "chromium"],
            capture_output=True, timeout=600)
    except Exception:
        pass
    return find_browser()


def screenshot_html(html: str, *, width: int, height: int,
                    timeout: float = 30.0) -> tuple | None:
    """Screenshot one self-contained page at an exact geometry.

    Returns (png_bytes, rung_label) or None when no browser rung
    is available or the shot failed — the caller then falls back.
    """
    found = ensure_browser()
    if found is None:
        return None
    exe, kind = found
    with tempfile.TemporaryDirectory(prefix="gs-tv-shot-") as td:
        page = Path(td) / "page.html"
        out = Path(td) / "shot.png"
        page.write_text(html, encoding="utf-8")
        args = [exe]
        if "headless-shell" not in exe and "headless_shell" not in exe:
            # The headless shell is headless by construction; the
            # full browsers take the new-headless flag.
            args.append("--headless=new")
        args += ["--disable-gpu", "--no-sandbox",
                 "--disable-dev-shm-usage", "--hide-scrollbars",
                 "--force-device-scale-factor=1",
                 f"--window-size={int(width)},{int(height)}",
                 "--virtual-time-budget=20000",
                 f"--screenshot={out}", page.as_uri()]
        try:
            subprocess.run(args, capture_output=True, timeout=timeout)
        except Exception:
            return None
        if not out.is_file():
            return None
        png = out.read_bytes()
        if not png.startswith(PNG_SIGNATURE):
            return None
        return png, f"{kind}:{Path(exe).name}"
