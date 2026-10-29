"""Validation loss of sim runs, read from their offline wandb logs, next to their success.

    python scripts/sim/wandb_val_loss.py <run_dir> [...] [--window 25] [--at 150 175 200 250] [--json out.json]

Per run: mean val_loss over the last `window` epochs before each epoch in `--at` (default: the
end of the run), and the 200-scene success of every evaluated checkpoint. The val loss is on
the run's OWN validation clouds (a cut run is scored on cut clouds), so it compares networks
that were each trained on the input they are scored on.
"""
import argparse
import glob
import json
import os

import numpy as np
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore


def history(run):
    files = sorted(glob.glob(os.path.join(run, "wandb", "*", "*.wandb")))
    rows = []
    for f in files:                       # a resumed run can hold several wandb runs
        ds = datastore.DataStore()
        ds.open_for_scan(f)
        while True:
            d = ds.scan_data()
            if d is None:
                break
            r = pb.Record()
            r.ParseFromString(d)
            if r.WhichOneof("record_type") == "history":
                rows.append({i.key or "/".join(i.nested_key): json.loads(i.value_json) for i in r.history.item})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--window", type=int, default=25)
    ap.add_argument("--at", type=int, nargs="*", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out = {}
    for run in a.runs:
        by_epoch = {}
        for r in history(run):
            if "val_loss" in r and "epoch" in r:
                by_epoch[int(r["epoch"])] = float(r["val_loss"])
        if not by_epoch:
            print(f"{os.path.basename(run)}: no val_loss")
            continue
        last = max(by_epoch)
        res = {"val_loss": {}, "success": {}}
        for e in (a.at or [last]):
            v = [by_epoch[k] for k in range(e - a.window + 1, e + 1) if k in by_epoch]
            if len(v) >= 2:
                res["val_loss"][e] = {"mean": float(np.mean(v)), "se": float(np.std(v, ddof=1) / np.sqrt(len(v))), "n": len(v)}
        for f in glob.glob(os.path.join(run, "eval", "*_200ep_block1.json")):
            res["success"][int(os.path.basename(f).split("_")[0])] = json.load(open(f))["success_rate"]
        out[os.path.basename(run)] = res
        vl = "  ".join(f"@{e}: {d['mean'] * 1e3:.3f}+-{d['se'] * 1e3:.3f}" for e, d in sorted(res["val_loss"].items()))
        sc = "  ".join(f"{e}: {s * 100:.1f}%" for e, s in sorted(res["success"].items()))
        print(f"{os.path.basename(run)}\n  val_loss e-3 (last {a.window} ep)  {vl}\n  success  {sc}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
