#!/usr/bin/env python3
"""Fit a candidate W6NL E4M3-exact codebook without loading/serving a model.
Samples BF16 down-projection weight rows from safetensors in disjoint layer sets.
The table is fitted to the weight distribution; it is not validated on held-out activations.
"""
import argparse, json, struct, pathlib
import numpy as np
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('checkpoint', type=pathlib.Path, help='original BF16 checkpoint directory')
ROOT = parser.parse_args().checkpoint
INDEX = json.loads((ROOT / 'model.safetensors.index.json').read_text())['weight_map']

def sample(layers, seed):
    rng = np.random.default_rng(seed)
    chunks = []
    for layer in layers:
        name = f'model.language_model.layers.{layer}.mlp.experts.down_proj'
        path = ROOT / INDEX[name]
        with path.open('rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            m = json.loads(f.read(n))[name]
            assert m['dtype'] == 'BF16' and m['shape'] == [512, 2560, 640], m
            base = 8 + n + m['data_offsets'][0]
            for e in rng.choice(512, size=5, replace=False):
                start = int(rng.integers(0, 2560 - 128))
                f.seek(base + (int(e) * 2560 + start) * 640 * 2)
                b = f.read(128 * 640 * 2)
                assert len(b) == 128 * 640 * 2
                u = np.frombuffer(b, dtype='<u2').astype(np.uint32) << 16
                x = u.view(np.float32).reshape(-1, 128).copy()
                for width in (1,2,4,8,16,32,64):
                    v = x.reshape(-1, 128//(2*width), 2, width)
                    a, c = v[:,:,0,:].copy(), v[:,:,1,:].copy()
                    v[:,:,0,:] = a+c
                    v[:,:,1,:] = a-c
                v = np.abs(x.reshape(-1,64))
                mx = np.max(v, axis=1, keepdims=True)
                v /= np.maximum(mx, 1e-30)
                chunks.append(v.reshape(-1))
    x = np.concatenate(chunks)
    assert np.all(np.isfinite(x))
    return x

def hist(x, bins=4096):
    # Preserve the amax=1 endpoint in the last bin.
    idx = np.minimum((x*bins).astype(np.int32),bins-1)
    return np.bincount(idx, minlength=bins).astype(np.float64)

def e4(b):
    exponent, mant = b >> 3, b & 7
    if exponent == 0: return mant * 2**-9
    return (1 + mant/8) * 2**(exponent-7)

def score(h, vals, scale, x):
    u = np.asarray(vals, dtype=np.float64)/scale
    i = np.searchsorted((u[1:]+u[:-1])/2, x)
    return float(np.sum(h*(x-u[i])**2)/h.sum())

def fit(h):
    x=(np.arange(len(h))+.5)/len(h)
    centers = np.linspace(0.01, 0.99, 32)
    for _ in range(2500):
        i=np.searchsorted((centers[1:]+centers[:-1])/2,x)
        counts=np.bincount(i, weights=h, minlength=32)
        sums=np.bincount(i, weights=h*x, minlength=32)
        new=np.divide(sums, counts, out=centers.copy(), where=counts>0)
        if np.max(np.abs(new-centers)) < 1e-9: break
        centers=new
    i=np.searchsorted((centers[1:]+centers[:-1])/2,x)
    counts=np.bincount(i, weights=h, minlength=32)
    vals=np.array([e4(b) for b in range(1,127)],dtype=np.float64)
    best=None
    for scale in np.arange(20.,80.01,.5):
        # Find 32 strictly increasing, exactly representable positive fp8 levels.
        cost=counts[:,None]*(centers[:,None]-vals[None,:]/scale)**2
        dp=cost[0].copy(); parents=[]
        for j in range(1,32):
            prev=np.r_[np.inf, dp[:-1]]
            arg=np.minimum.accumulate(prev)
            inds=np.empty(len(vals),np.int32); k=-1; m=np.inf
            for z,p in enumerate(prev):
                if p<m: m=p; k=z-1
                inds[z]=k
            dp=cost[j]+arg; parents.append(inds)
        # Force the outermost magnitude near amax: production scale search starts at amax/peak.
        # Do not spend levels beyond 1 merely because empty high-end Lloyd cells cost nothing.
        eligible=np.where((vals >= scale * .98) & (vals <= scale * 1.02),dp,np.inf)
        z=int(np.argmin(eligible)); chosen=[z]
        for parent in reversed(parents):
            z=int(parent[z]); chosen.append(z)
        chosen=chosen[::-1]
        if min(chosen)<0: continue
        table=vals[chosen]
        err=score(h,table,scale,x)
        if best is None or err<best[0]: best=(err,scale,table.copy())
    return best

a=sample([0,8,16,24,32,40,47],20261116)
b=sample([4,12,20,28,36,44],20261117)
h,hh=hist(a),hist(b); x=(np.arange(len(h))+.5)/len(h)
err,scale,tab=fit(h)
w5=[.8125,2.5,4,5.5,7.5,9,11,12,14,16,18,20,22,24,26,30]
# Optimize one scalar on each sampled distribution, analogous to scale search; not GPTQ.
for label,table in [('W5',w5),('W6',tab)]:
    for split,weights in [('fit',h),('heldout',hh)]:
        e,s=min((score(weights,table,s,x),s) for s in np.arange(20.,80.01,.1))
        print(label,split,'normalized MSE',round(e,9),'scale',round(s,2))
print('fit_count',len(a),'heldout_count',len(b),'candidate_scale',scale)
print('w6 magnitudes',', '.join(f'{v:g}' for v in tab))
print('e4m3 bytes',', '.join(f'0x{next(i for i in range(1,127) if e4(i)==v):02x}' for v in tab))
