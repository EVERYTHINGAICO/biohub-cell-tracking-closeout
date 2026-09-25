"""FASE 1 (appearance linker): build patch-pair dataset from DoG dets + GT.

Per volume: candidate det pairs between consecutive frames labelled by GT edges
(all positives + neg_ratio sampled negatives). Each endpoint detection's 32^3
image patch is stored ONCE (float16); pairs reference patches by index -> no
duplication. Saved per-volume npz: patches (N,32,32,32) + pairs (M, [i,j,label,
dist_um,dz,dy,dx,score_i,score_j]).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, geff, networkx as nx
from numcodecs import Blosc
from scipy.spatial import cKDTree

CODEC = Blosc(cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE, blocksize=0)
SCALE = np.array([1.625, 0.40625, 0.40625]); MATCH_UM = 7.0; GATE_UM = 12.0
P = 32; R = P // 2


def read_meta(zp):
    m = json.load(open(zp/"0"/"zarr.json"))
    return tuple(int(v) for v in m["shape"]), tuple(int(v) for v in m["chunk_grid"]["configuration"]["chunk_shape"])
def read_frame(zp,t,ch):
    raw=CODEC.decode((zp/"0"/"c"/str(t)/"0"/"0"/"0").read_bytes()); return np.frombuffer(raw,dtype="<u2").reshape(ch)[0]
def norm(f):
    lo,hi=np.percentile(f,[1.0,99.7]); hi=hi if hi>lo else lo+1
    return np.clip((f.astype(np.float32)-lo)/(hi-lo),0,1).astype(np.float32)
def patch(fr,z,y,x):
    Z,Y,X=fr.shape
    z0=min(max(int(round(z))-R,0),Z-P); y0=min(max(int(round(y))-R,0),Y-P); x0=min(max(int(round(x))-R,0),X-P)
    return fr[z0:z0+P,y0:y0+P,x0:x0+P]


def gt_data(gp):
    g,_=geff.read(gp,structure_validation=False)
    if not isinstance(g,nx.DiGraph): g=nx.DiGraph(g)
    nodes={}
    for n,a in g.nodes(data=True): nodes.setdefault(int(a["t"]),[]).append((int(n),a["z"],a["y"],a["x"]))
    return nodes, {(int(u),int(v)) for u,v in g.edges()}
def match(dets_zyx, gt_list):
    if not gt_list or len(dets_zyx)==0: return np.full(len(dets_zyx),-1,np.int64)
    ids=np.array([g[0] for g in gt_list]); xyz=np.array([[g[1],g[2],g[3]] for g in gt_list])*SCALE
    d,idx=cKDTree(xyz).query(dets_zyx*SCALE,k=1); return np.where(d<=MATCH_UM,ids[idx],-1)


def build(zp, dets_v, gt_nodes, gt_edges, neg_ratio, rng):
    sh,ch=read_meta(zp)
    fr_dets={int(t):(g[["z","y","x"]].to_numpy(float),g["score"].to_numpy(float)) for t,g in dets_v.groupby("t")}
    pos, neg = [], []   # each: (t,i,t1,j, feat)
    for t in sorted(fr_dets):
        if (t+1) not in fr_dets: continue
        zi,si=fr_dets[t]; zj,sj=fr_dets[t+1]
        if len(zi)==0 or len(zj)==0: continue
        mi=match(zi,gt_nodes.get(t,[])); mj=match(zj,gt_nodes.get(t+1,[]))
        pi=zi*SCALE; pj=zj*SCALE; treej=cKDTree(pj)
        for i in range(len(zi)):
            for j in treej.query_ball_point(pi[i],GATE_UM):
                dv=pi[i]-pj[j]; feat=[float(np.linalg.norm(dv)),dv[0],dv[1],dv[2],si[i],sj[j]]
                if mi[i]!=-1 and mj[j]!=-1 and (int(mi[i]),int(mj[j])) in gt_edges: pos.append((t,i,t+1,j,feat))
                else: neg.append((t,i,t+1,j,feat))
    if not pos: return None
    if len(neg)>neg_ratio*len(pos):
        neg=[neg[k] for k in rng.choice(len(neg),neg_ratio*len(pos),replace=False)]
    pairs=pos+neg; labels=[1]*len(pos)+[0]*len(neg)
    # collect unique endpoint patches
    need={}  # (t,idx)->patch_row
    for (t,i,t1,j,_) in pairs:
        need.setdefault((t,i),None); need.setdefault((t1,j),None)
    frames_needed=sorted({t for (t,_) in need})
    patches=[];
    for t in frames_needed:
        fr=norm(read_frame(zp,t,ch)); zi,_=fr_dets[t]
        for (tt,idx) in [k for k in need if k[0]==t]:
            need[(tt,idx)]=len(patches); z,y,x=zi[idx]; patches.append(patch(fr,z,y,x).astype(np.float16))
    P_arr=np.stack(patches)
    rows=[]
    for (t,i,t1,j,feat),lb in zip(pairs,labels):
        rows.append([need[(t,i)],need[(t1,j)],lb]+feat)
    return P_arr, np.array(rows,np.float32)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train-dir",required=True); ap.add_argument("--dets",default="detections_train.csv")
    ap.add_argument("--output-dir",default="/tmp/appearance_dataset"); ap.add_argument("--neg-ratio",type=int,default=5)
    ap.add_argument("--limit",type=int,default=0); ap.add_argument("--offset",type=int,default=0)
    args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); train=Path(args.train_dir); rng=np.random.default_rng(0)
    print("loading dets...",flush=True); dets=pd.read_csv(args.dets)
    vols=sorted(dets["volume"].unique())[args.offset:]
    if args.limit: vols=vols[:args.limit]
    tp=tn=0
    for k,v in enumerate(vols,1):
        gp=train/f"{v}.geff"
        if not gp.exists(): continue
        gn,ge=gt_data(gp); res=build(zp:=train/f"{v}.zarr",dets[dets.volume==v],gn,ge,args.neg_ratio,rng)
        if res is None: print(f"[{k}] {v}: no positives"); continue
        Pa,rows=res; np.savez_compressed(out/f"{v}.npz",patches=Pa,pairs=rows)
        npos=int((rows[:,2]==1).sum()); tp+=npos; tn+=len(rows)-npos
        print(f"[{k}/{len(vols)}] {v}: {len(rows)} pairs ({npos} pos), {len(Pa)} patches",flush=True)
    print(f"DONE. {tp} pos + {tn} neg pairs -> {out}")


if __name__=="__main__":
    main()
