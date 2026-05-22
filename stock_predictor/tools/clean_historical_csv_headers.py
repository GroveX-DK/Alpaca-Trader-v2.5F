"""Fjern yfinance/Yahoo-preamblen (Ticker + tom Date-række) fra *_historical.csv i projektroden."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _strip_yfinance_preamble(content: str) -> str | None:
    """
    Genkend:
      linje 1: Price,Close,... (eller Open,High,... uden Date)
      linje 2: Ticker,SYM,...
      linje 3: Date,,,,,
    Returnér ny filtekst med én header-række Date,... + data; ellers None (ingen ændring).
    """
    lines = content.splitlines()
    if len(lines) < 4:
        return None
    l0, l1, l2 = lines[0].strip(), lines[1].strip(), lines[2].strip()
    if not l1.lower().startswith("ticker,"):
        return None
    if not l2.lower().startswith("date,"):
        return None
    if l0.lower().startswith("date,"):
        header = l0
    else:
        header = "Date," + l0
    rest = "\n".join(lines[3:])
    if not rest:
        return header + "\n"
    return (header + "\n" + rest).rstrip("\n") + "\n"


def _process_file(path: Path, dry_run: bool) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    new_content = _strip_yfinance_preamble(raw)
    if new_content is None:
        return "skip (allerede ren eller ukendt format)"
    if new_content == raw:
        return "skip (uændret)"
    if dry_run:
        return f"ville rense ({len(raw)} -> {len(new_content)} tegn)"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(new_content, encoding="utf-8", newline="\n")
    tmp.replace(path)
    return "renset"


def main() -> int:
    p = argparse.ArgumentParser(description="Rens *_historical.csv for Ticker/Date-preambler øverst.")
    p.add_argument(
        "--project-root",
        type=Path,
        default=_ROOT,
        help="Mappe med SYM_historical.csv (default: projektrod).",
    )
    p.add_argument("--dry-run", action="store_true", help="Vis kun hvad der ville ske.")
    args = p.parse_args()
    root: Path = args.project_root.resolve()
    paths = sorted(root.glob("*_historical.csv"))
    if not paths:
        print(f"Ingen *_historical.csv i {root}", flush=True)
        return 0

    n_ok = n_skip = 0
    for path in paths:
        msg = _process_file(path, args.dry_run)
        print(f"{path.name}: {msg}", flush=True)
        if msg.startswith("skip"):
            n_skip += 1
        else:
            n_ok += 1

    print(f"Færdig: {n_ok} opdateret / kørsel, {n_skip} sprunget over.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
