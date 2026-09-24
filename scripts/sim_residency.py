#!/usr/bin/env python3
"""Replay a routing trace against a unit cache and report the decode hit rate
of LRU, LEDGER's value estimate and Belady (future knowledge), so a policy
can be judged against the bound before a three-minute driver run.

usage: sim_residency.py trace.npz cache_fraction [held-out traces...]
"""
import numpy as np, sys
# Replay a trace against a unit cache of C units; report decode hit rate.
# Every policy is "keep the top-C of (resident U passing units) by a score",
# which is exactly LRU when the score is last-use time.
d=np.load(sys.argv[1], allow_pickle=True); frac=float(sys.argv[2])
tags=list(d['tags']); L=int(d['n_layers']); E=int(d['n_experts'])
tag=d['tag']; lay=d['layer'].astype(int); ex=d['expert'].astype(int); pos=d['pos']
reqs={}
for i,t in enumerate(tags):
    n,ph=t.rsplit('/',1); reqs.setdefault(n,{})[ph]=i
N=L*E; C=int(frac*N)
R=[]
for n,ph in reqs.items():
    pm=tag==ph['prefill']; dm=tag==ph['decode']
    u=np.zeros(N,bool); ids=(lay[pm][:,None]*E+ex[pm]).ravel(); u[ids]=True
    pf=np.bincount(ids,minlength=N).astype(float)
    dp=pos[dm]; toks=[]
    for p in np.unique(dp):
        m=dp==p; toks.append(np.unique((lay[dm][m][:,None]*E+ex[dm][m]).ravel()))
    R.append((u,pf,toks))
R=R*2
def run(policy, h0=None, **kw):
    res=np.zeros(N,bool); last=np.full(N,-1e9); hist=(h0.copy() if h0 is not None else np.zeros(N)); cnt_req=np.zeros(N)
    now=0; hit=tot=0; pfn=np.zeros(N)
    def keep(cand, score):
        idx=np.flatnonzero(cand)
        if len(idx)<=C: r=np.zeros(N,bool); r[idx]=True; return r
        top=idx[np.argpartition(-score[idx], C-1)[:C]]
        r=np.zeros(N,bool); r[top]=True; return r
    for (u,pf,toks) in R:
        now+=1
        pfn=pf/pf.sum(); cnt_req[:]=0
        last[u]=now
        res=keep(res|u, policy(last,now,pfn,hist,cnt_req,**kw))
        for t in toks:
            now+=1
            hit+=res[t].sum(); tot+=len(t)
            last[t]=now; hist[t]+=1; cnt_req[t]+=1
            need=np.zeros(N,bool); need[t]=True
            res=keep(res|need, policy(last,now,pfn,hist,cnt_req,**kw))
    return hit/tot
lru=lambda last,now,pfn,hist,cr: last
def pred(last,now,pfn,hist,cr,a=0.5,H=8,wr=1.0):
    rec=np.exp2(-(now-last)/H)
    h=hist/max(hist.sum(),1)
    return wr*rec + a*pfn/ max(pfn.max(),1e-12) + (1-a)*h/max(h.max(),1e-12)
print(f"C={C}/{N}  LRU {run(lru):.3f}", end="")
for a in ():
    for H in (4,16):
        for wr in (0.5,1,2):
            print(f"  a{a}H{H}w{wr} {run(pred,a=a,H=H,wr=wr):.3f}", end="")
print()
# Belady: evict the unit whose next decode use is furthest (future knowledge).
def belady():
    seq=[]  # list of (is_prefill, ids)
    for (u,pf,toks) in R:
        seq.append((True,np.flatnonzero(u)))
        for t in toks: seq.append((False,t))
    T=len(seq); INF=10**9
    nxt=np.full(N,INF); nexts=[None]*T
    for i in range(T-1,-1,-1):
        nexts[i]=nxt.copy()
        if not seq[i][0]: nxt[seq[i][1]]=i
    res=np.zeros(N,bool); hit=tot=0
    for i,(isp,ids) in enumerate(seq):
        if not isp: hit+=res[ids].sum(); tot+=len(ids)
        cand=res.copy(); cand[ids]=True
        idx=np.flatnonzero(cand)
        if len(idx)>C:
            sc=-nexts[i][idx].astype(float)   # after step i
            top=idx[np.argpartition(-sc, C-1)[:C]]
            res=np.zeros(N,bool); res[top]=True
        else: res=cand
    return hit/tot


def dec_counts(path):
    d2=np.load(path, allow_pickle=True); tg=list(d2['tags'])
    dm=np.isin(d2['tag'], [i for i,t in enumerate(tg) if t.endswith('/decode')])
    return np.bincount((d2['layer'][dm].astype(int)[:,None]*E+d2['expert'][dm].astype(int)).ravel(),minlength=N).astype(float)
others=sys.argv[3:]
h0=sum(dec_counts(p) for p in others) if others else None
print(f"  Belady {belady():.3f}  best-fixed {run(pred,a=0.5,H=8,wr=1):.3f}", end="")
if h0 is not None:
    for pw in (1.0, 0.25):
        print(f"  +profile x{pw} {run(pred,h0=h0*pw,a=0.5,H=8,wr=1):.3f}", end="")
print()
