"""P24: opposite-sign degradation CHSH test. Aer validation (default) or ibm_fez (BMR_P24_HW=1).
Degradation = RY(+d) on q0 and RY(-d) on q1 after the measurement-basis rotations (ideal S=2.6052 at d=0.2).
Same-sign (old run) leaves S unchanged: null perturbation. Credentials are never printed."""
import math, os, sys, json, numpy as np
from qiskit import QuantumCircuit, transpile
pi=math.pi; D=0.2; SHOTS=8192
ANG={(0,0):(0,pi/8),(0,1):(0,-pi/8),(1,0):(pi/4,pi/8),(1,1):(pi/4,-pi/8)}
SIGN={(0,0):1,(0,1):1,(1,0):1,(1,1):-1}
def circ(a,b,deg):
    qc=QuantumCircuit(2,2); qc.h(0); qc.cx(0,1); qc.ry(2*a,0); qc.ry(2*b,1)
    if deg: qc.ry(D,0); qc.ry(-D,1)
    qc.measure([0,1],[0,1]); return qc
def corr(c):
    n=sum(c.values()); return sum(v*(1 if k[0]==k[1] else -1) for k,v in c.items())/n
def S_se(cs):
    E={k:corr(c) for k,c in cs.items()}
    S=sum(SIGN[k]*E[k] for k in E)
    var=sum((1-E[k]**2)/SHOTS for k in E)
    return S, math.sqrt(var)
def labelled(backend, reps):
    out=[]
    for r in range(reps):
        for deg in (False,True):
            out.append((r,deg,[(k,transpile(circ(*ANG[k],deg),backend,initial_layout=[0,1],optimization_level=1)) for k in ANG]))
    return out
if os.environ.get("BMR_P24_HW")=="1":
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    s=QiskitRuntimeService(name="default-ibm-quantum-platform"); be=s.backend("ibm_fez")
    reps=int(os.environ.get("BMR_P24_REPS","5")); tag="ibm_fez"
    lab=labelled(be,reps); res=[]
    # one job per (rep,condition): 4 circuits
    for r,deg,items in lab:
        job=SamplerV2(mode=be).run([c for _,c in items],shots=SHOTS)
        print("job",job.job_id(),r,deg,flush=True)
        pr=job.result()
        cs={k:pr[i].data.c.get_counts() for i,(k,_) in enumerate(items)}
        res.append((r,deg,job.job_id(),cs))
        S,se=S_se(cs); print(f"  rep{r} {'degraded' if deg else 'native'} S={S:.4f} se={se:.4f}",flush=True)
else:
    from qiskit_aer import AerSimulator
    be=AerSimulator(); reps=5; tag="aer_ideal"
    res=[]
    for r,deg,items in labelled(be,reps):
        cs={}
        for k,c in items: cs[k]=be.run(c,shots=SHOTS,seed_simulator=1000*r+100*int(deg)+10*(2*k[0]+k[1])+7).result().get_counts()
        res.append((r,deg,"aer",cs))
rows=[]
for r,deg,jid,cs in res:
    S,se=S_se(cs); rows.append(dict(rep=r,cond="degraded" if deg else "native",S=S,se=se,job=jid,counts={f"{a}{b}":v for (a,b),v in cs.items()}))
summ={}
for cond in ("native","degraded"):
    v=np.array([x["S"] for x in rows if x["cond"]==cond]); se=np.array([x["se"] for x in rows if x["cond"]==cond])
    summ[cond]=dict(n=len(v),mean=float(v.mean()),sd_between=float(v.std(ddof=1)) if len(v)>1 else None,
                    se_pooled=float(math.sqrt((se**2).sum())/len(se)))
summ["delta_mean_deg_minus_nat"]=summ["degraded"]["mean"]-summ["native"]["mean"]
json.dump(dict(backend=tag,shots=SHOTS,d=D,rows=rows,summary=summ),open(f"../hardware/P24_{tag}.json","w"),indent=1)
print(json.dumps(summ,indent=1))
