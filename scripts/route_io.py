#!/usr/bin/env python3
"""Glue for tools/route_dump: workload JSON to TSV in, RDMP binary to npz out.

The npz matches capture_workload.py's, so everything downstream -- the
analysis and export_routing.py -- reads a GGUF capture and an HF capture the
same way.
"""
import argparse, json, struct, sys
import numpy as np

def to_tsv(src, dst):
    rows = json.load(open(src))
    with open(dst, "w") as f:
        for r in rows:
            p = r["prompt"].replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")
            f.write(f"{r['name']}\t{int(r.get('max_new',16))}\t{p}\n")
    print(f"{len(rows)} prompts -> {dst}")

def to_npz(src, dst, workload, n_layers, n_experts):
    b = open(src, "rb").read(); o = 0
    assert b[:4] == b"RDMP"; o = 4
    k, nn = struct.unpack_from("<II", b, o); o += 8
    names = []
    for _ in range(nn):
        l, = struct.unpack_from("<I", b, o); o += 4
        names.append(b[o:o+l].decode()); o += l
    nr, = struct.unpack_from("<I", b, o); o += 4
    rec = np.dtype([("tag","<u4"),("phase","u1"),("layer","<u2"),("pos","<u4"),("ex","<i4",(k,))])
    arr = np.frombuffer(b, dtype=rec, count=nr, offset=o)
    tags = []
    for n in names: tags += [n + "/prefill", n + "/decode"]
    tid = np.array([arr["tag"][i]*2 + arr["phase"][i] for i in range(nr)], dtype=np.int16)
    fam = {}
    if workload:
        for r in json.load(open(workload)): fam[r["name"]] = r["family"]
    np.savez_compressed(dst, tag=tid, layer=arr["layer"].astype(np.int16),
        pos=arr["pos"].astype(np.int32), expert=arr["ex"].astype(np.int16),
        tags=np.array(tags), n_experts=n_experts, topk=k, n_layers=n_layers,
        family=np.array([fam.get(t.split("/")[0], "") for t in tags]))
    print(f"{nr} records, k={k}, {nn} prompts -> {dst}")

ap = argparse.ArgumentParser()
sp = ap.add_subparsers(dest="cmd", required=True)
a1 = sp.add_parser("tsv"); a1.add_argument("src"); a1.add_argument("dst")
a2 = sp.add_parser("npz"); a2.add_argument("src"); a2.add_argument("dst")
a2.add_argument("--workload", default=""); a2.add_argument("--layers", type=int, required=True)
a2.add_argument("--experts", type=int, required=True)
a = ap.parse_args()
if a.cmd == "tsv": to_tsv(a.src, a.dst)
else: to_npz(a.src, a.dst, a.workload, a.layers, a.experts)
