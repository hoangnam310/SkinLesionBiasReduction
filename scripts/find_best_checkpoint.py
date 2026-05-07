"""Find the saved cGAN checkpoint nearest to the lowest-FID epoch in a run.

Parses outputs/<run_dir>/logs/training.log for "FID: <value>" lines, identifies
the epoch with the lowest FID, then returns the path of the saved checkpoint
closest to (but not after) that epoch.

Usage:
    python scripts/find_best_checkpoint.py outputs/wgan_center_20260507_012501

Prints the absolute path of the chosen checkpoint to stdout. Exits non-zero if
no FID lines are found or no matching checkpoint is saved.
"""

import re
import sys
from pathlib import Path


FID_RE = re.compile(r"FID:\s*([0-9]+\.[0-9]+)")
CKPT_RE = re.compile(r"checkpoint_epoch_(\d+)\.pt$")


def find_best_checkpoint(run_dir: Path) -> Path:
    log_path = run_dir / "logs" / "training.log"
    if not log_path.is_file():
        raise SystemExit(f"training.log not found at {log_path}")

    # The log records "Saved checkpoint" before "Computing FID" / "FID: X" at the
    # FID-evaluation epochs (every fid_interval). The most recent checkpoint
    # before each FID line is the one that produced that FID.
    fid_to_epoch = []  # list of (fid, last_saved_epoch)
    last_saved_epoch = None
    saved_re = re.compile(r"checkpoint_epoch_(\d+)\.pt")

    with log_path.open() as f:
        for line in f:
            m_save = saved_re.search(line)
            if m_save:
                last_saved_epoch = int(m_save.group(1))
                continue
            m_fid = FID_RE.search(line)
            if m_fid and last_saved_epoch is not None:
                fid_to_epoch.append((float(m_fid.group(1)), last_saved_epoch))

    if not fid_to_epoch:
        raise SystemExit(f"No FID/checkpoint pairs found in {log_path}")

    best_fid, best_epoch = min(fid_to_epoch, key=lambda x: x[0])

    ckpt_dir = run_dir / "checkpoints"
    candidate = ckpt_dir / f"checkpoint_epoch_{best_epoch:04d}.pt"
    if candidate.is_file():
        chosen = candidate
    else:
        # Fallback: pick the saved checkpoint with the largest epoch <= best_epoch.
        saved = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"))
        candidates = []
        for p in saved:
            m = CKPT_RE.search(p.name)
            if m:
                ep = int(m.group(1))
                if ep <= best_epoch:
                    candidates.append((ep, p))
        if not candidates:
            raise SystemExit(
                f"No checkpoint at or before epoch {best_epoch} in {ckpt_dir}"
            )
        chosen = max(candidates, key=lambda x: x[0])[1]

    print(chosen.resolve(), end="")
    sys.stderr.write(
        f"# best FID {best_fid:.2f} @ epoch {best_epoch} -> {chosen.name}\n"
    )
    return chosen


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: find_best_checkpoint.py <run_dir>")
    find_best_checkpoint(Path(sys.argv[1]))
