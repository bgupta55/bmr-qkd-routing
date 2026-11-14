import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False})
INK="#1f2328"; MUT="#59636e"; BLUE="#0072B2"; ORG="#D55E00"; GRY="#9aa3ad"; LIGHT="#eef1f4"

def save(fig,name):
    fig.savefig(name+".pdf",bbox_inches="tight"); fig.savefig(name+".png",dpi=200,bbox_inches="tight"); plt.close(fig)

def node(ax,x,y,label,kind="honest",r=0.28):
    fc={"honest":"white","bad":ORG,"end":BLUE}[kind]; ec=INK
    hatch="///" if kind=="bad" else None
    c=Circle((x,y),r,fc=fc if kind!="bad" else "white",ec=ORG if kind=="bad" else ec,lw=1.6,hatch=hatch,zorder=3); ax.add_patch(c)
    ax.text(x,y,label,ha="center",va="center",color="white" if kind=="end" else INK,fontsize=8,zorder=4,fontweight="bold")
def link(ax,p,q,style="-",col=INK,lw=1.4):
    ax.plot([p[0],q[0]],[p[1],q[1]],ls=style,color=col,lw=lw,zorder=1)

# ---------- Fig 3' (F2)
fig,(a,b)=plt.subplots(1,2,figsize=(9.2,3.1),gridspec_kw={"width_ratios":[1,1.05]})
for ax in (a,b): ax.set_aspect("equal"); ax.axis("off")
a.set_xlim(-0.6,8.6); a.set_ylim(-1.9,1.5)
xs=[0,2,4,6,8]; labs=["s","v1","v2","v3","d"]
for i in range(4): link(a,(xs[i],0),(xs[i+1],0))
for x,l in zip(xs,labs): node(a,x,0,l,"end" if l in("s","d") else ("bad" if l=="v2" else "honest"))
a.text(4,0.95,"one dishonest relay on the only path",ha="center",color=ORG,fontsize=9)
a.text(4,-1.05,"v2 holds the hop keys on both sides:\nit learns the key K (Lemma A)",ha="center",color=INK,fontsize=8.5)
a.set_title("(a) Single path, one passive relay",loc="left",fontsize=10,color=INK)
b.set_xlim(-0.6,8.6); b.set_ylim(-2.3,2.3)
b.set_title("(b) Three disjoint paths, t = 2",loc="left",fontsize=10,color=INK)
node(b,0,0,"s","end"); node(b,8,0,"d","end")
rows=[(1.5,["a1","a2"],"bad"),(0,["b1","b2"],"bad"),(-1.5,["c1","c2"],"honest")]
for y,names,kind in rows:
    pts=[(0,0),(2.7,y),(5.3,y),(8,0)]
    for p,q in zip(pts[:-1],pts[1:]): link(b,p,q,style="-")
    for x,nm in zip((2.7,5.3),names): node(b,x,y,nm,kind if nm in("a1","b1") else "honest")
b.text(4,2.05,"path 1: relay a1 dishonest",ha="center",color=ORG,fontsize=8)
b.text(4,0.5,"path 2: relay b1 dishonest",ha="center",color=ORG,fontsize=8)
b.text(4,-2.1,"path 3: no dishonest relay → share Y3 stays hidden → K is secure",ha="center",color=BLUE,fontsize=8.5)
save(fig,"fig3prime_cut_vs_disjoint")

# ---------- F1 key delivery
fig,ax=plt.subplots(figsize=(9.2,3.6)); ax.axis("off"); ax.set_xlim(0,12); ax.set_ylim(0,6.4)
def box(x,y,w,h,txt,fc=LIGHT,ec=INK):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.05,rounding_size=0.15",fc=fc,ec=ec,lw=1.2)); ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=8.2,color=INK)
box(0.1,2.0,2.6,2.4,"Source s\n\ndraw M = (K, r1, r2)\nshares Y1, Y2, Y3\nY1⊕Y2⊕Y3 = M")
box(9.3,2.0,2.6,2.4,"Destination d\n\nM = Y1⊕Y2⊕Y3\ncheck tag\nθ = h_r1(K) ⊕ r2")
for y,i in zip((5.0,3.2,1.4),(1,2,3)):
    ax.add_patch(FancyArrowPatch((2.8,3.2),(4.0,y),arrowstyle="-",color=GRY,lw=1))
    for k,x in enumerate((4.0,6.2,8.4)):
        ax.add_patch(Circle((x,y),0.12,fc=INK,ec=INK)) if k==0 else None
    ax.plot([4.0,8.4],[y,y],color=INK,lw=1.3)
    ax.add_patch(FancyArrowPatch((8.4,y),(9.2,3.2),arrowstyle="-",color=GRY,lw=1))
    for x in (4.0,6.2,8.4): ax.add_patch(Circle((x,y),0.13,fc="white",ec=INK,lw=1.3,zorder=3))
    ax.text(5.1,y+0.28,"c = Y%d ⊕ pad (+ MAC)"%i,fontsize=7.6,color=MUT,ha="center")
    ax.text(7.3,y+0.28,"c = Y%d ⊕ pad (+ MAC)"%i,fontsize=7.6,color=MUT,ha="center")
    ax.text(3.35,y+(0.0 if y==3.2 else (0.35 if y>3.2 else -0.35)),"path %d"%i,fontsize=7.6,color=INK,ha="center")
ax.text(6.2,0.35,"Every relay on path i decrypts and re-encrypts Y_i only. (r1, r2) travel inside the shared payload M, never in the clear on every path.",ha="center",fontsize=8,color=INK)
ax.text(6.2,6.15,"Algorithm 4: key delivery by XOR sharing over t+1 disjoint paths",ha="center",fontsize=10,color=INK)
save(fig,"figF1_key_delivery")

# ---------- F7 Lemma A partition
fig,ax=plt.subplots(figsize=(9.2,3.3)); ax.axis("off"); ax.set_xlim(0,12); ax.set_ylim(0,5.6)
def region(x,w,title,sub,fc):
    ax.add_patch(FancyBboxPatch((x,1.2),w,3.4,boxstyle="round,pad=0.05,rounding_size=0.2",fc=fc,ec=INK,lw=1.2))
    ax.text(x+w/2,4.15,title,ha="center",fontsize=10,color=INK,fontweight="bold"); ax.text(x+w/2,3.75,sub,ha="center",va="top",fontsize=8.3,color=INK)
region(0.2,3.4,"A: side of s","nodes reachable from s in (V\\C, H)\nX_A = randomness and\ninternal keys of A","#dfeaf4")
region(4.3,3.4,"C: coalition","passive, meets every s–d path\n\nZ = randomness of C and\nkeys of all links at C","#f6e1d6")
region(8.4,3.4,"B: side of d","all other nodes\nX_B = randomness and\ninternal keys of B","#dfeaf4")
node(ax,0.95,1.62,"s","end"); node(ax,11.05,1.62,"d","end")
ax.add_patch(Rectangle((0.2,0.35),11.6,0.5,fc=LIGHT,ec=INK,lw=1)); ax.text(6,0.6,"public transcript T (visible to everyone)",ha="center",va="center",fontsize=8.5,color=INK)
ax.text(6,5.2,"X_A, X_B, Z independent; no H-link joins A and B  ⇒  X_A ⊥ X_B given (T, Z)  ⇒  K_s, K_d independent given V_C",ha="center",fontsize=9,color=INK)
save(fig,"figF7_lemmaA_partition")

# ---------- F5 decision flow
fig,ax=plt.subplots(figsize=(8.8,3.9)); ax.axis("off"); ax.set_xlim(0,12); ax.set_ylim(0,5.6)
def dbox(x,y,w,h,txt,fc="white"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.05,rounding_size=0.2",fc=fc,ec=INK,lw=1.3)); ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=8.6,color=INK)
def arr(p,q,t=None,off=(0,0.15)):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle="-|>",mutation_scale=10,color=INK,lw=1.2))
    if t: ax.text((p[0]+q[0])/2+off[0],(p[1]+q[1])/2+off[1],t,fontsize=8,color=MUT,ha="center")
dbox(0.2,3.9,3.4,1.2,"Which adversary must\nthe delivered key resist?")
dbox(4.6,4.3,3.0,0.9,"Eavesdropper only\nrequest t = 0: single path\n(budget Σ(ε+δ) ≤ Δ)",LIGHT)
dbox(4.6,2.35,3.0,1.45,"t colluding relays:\nare there t+1 internally\ndisjoint paths, each\nwithin budget?")
dbox(8.6,3.3,3.2,1.0,"Yes: plan of t+1 paths\nXOR sharing (Algorithm 4)","#dfeaf4")
dbox(8.6,1.5,3.2,1.2,"No: BMR refuses\n(no weaker guarantee\nis offered)","#f6e1d6")
dbox(4.6,0.35,3.0,1.1,"Active relays or more than\nt colluders: no secrecy\nstatement (see Table 4)")
arr((3.6,4.5),(4.6,4.75),"only")
arr((3.2,3.9),(4.6,3.2),"collusion",off=(-0.55,0.0))
arr((7.6,3.1),(8.6,3.8),"yes",off=(0,0.2)); arr((7.6,2.8),(8.6,2.2),"no",off=(0.1,-0.35))
arr((1.9,3.9),(4.6,1.0),"beyond t",off=(-0.35,0.0))
save(fig,"figF5_decision_flow")

# ---------- F3 exposure vs adversary
x=["random relay","p = 0.02","p = 0.05","p = 0.10"]
data={"SPF":[10.09,2.92,7.13,13.78],"B2":[17.29,5.39,12.79,23.55],"BMR single path":[22.69,6.62,15.62,28.43],"MP-SPF":[0,0.21,1.22,4.21],"BMR-MP (plans)":[0,0.39,2.14,7.04]}
sty={"SPF":(GRY,"o","-"),"B2":(MUT,"s","-"),"BMR single path":(INK,"D","--"),"MP-SPF":("#56B4E9","^","-"),"BMR-MP (plans)":(BLUE,"o","-")}
fig,ax=plt.subplots(figsize=(6.4,3.5))
for k,v in data.items():
    c,m,ls=sty[k]; ax.plot(range(1,4),v[1:],color=c,marker=m,ls=ls,lw=2 if "BMR-MP" in k else 1.5,ms=6,mfc="white" if k in("SPF","MP-SPF") else c,zorder=3); ax.plot([0],[v[0]],color=c,marker=m,ls="",ms=6,mfc="white" if k in("SPF","MP-SPF") else c,zorder=3)
    ax.text(3.08,v[-1],k,va="center",fontsize=8,color=INK)
ax.set_xticks(range(4)); ax.set_xticklabels(x); ax.set_xlim(-0.2,4.3); ax.set_ylim(0,31)
ax.set_ylabel("delivered keys exposed (%)"); ax.grid(axis="y",color="#e3e6ea",lw=0.8); ax.set_axisbelow(True)
ax.set_title("Exposure against weaker adversaries, t = 1 (ER, BA, hub, ring; in our simulations)",fontsize=9,loc="left")
save(fig,"figF3_exposure_vs_adversary")

# ---------- F4 price
fig,(a,b)=plt.subplots(1,2,figsize=(8.6,3.3),gridspec_kw={"width_ratios":[1.5,1]})
ts=["t = 0\n(B2)","t = 1\n(BMR-MP)","t = 2\n(BMR-MP)"]
topo=[0,3.93,58.27]; rate=[0,10.79,6.23]; bud=[0,13.59,6.73]; b2=[7.7,0,0]
a.bar(range(3),b2,color=GRY,width=0.55,label="B2 refusal (not split)")
a.bar(range(3),topo,bottom=b2,color=BLUE,width=0.55,hatch="",label="fewer than t+1 disjoint paths")
bt=[b2[i]+topo[i] for i in range(3)]
a.bar(range(3),rate,bottom=bt,color="#56B4E9",width=0.55,hatch="//",ec="white",label="rate filter")
bt2=[bt[i]+rate[i] for i in range(3)]
a.bar(range(3),bud,bottom=bt2,color=ORG,width=0.55,hatch="xx",ec="white",label="per-path budget")
for i,tot in enumerate([7.7,28.31,71.23]): a.text(i,tot+1.5,f"{tot:.1f} %",ha="center",fontsize=8.5,color=INK)
a.set_xticks(range(3)); a.set_xticklabels(ts); a.set_ylim(0,84); a.set_ylabel("requests refused (%)"); a.legend(fontsize=7.3,frameon=False,loc="upper left")
a.set_title("(a) Why requests are refused",loc="left",fontsize=9.5)
b.bar(range(3),[0,0.0512,0.1390],color=INK,width=0.5)
for i,v in enumerate([0,0.0512,0.1390]): b.text(i,v+0.004,f"{v:.3f}" if v else "0",ha="center",fontsize=8.5)
b.set_xticks(range(3)); b.set_xticklabels(["t = 0","t = 1","t = 2"]); b.set_ylim(0,0.17); b.set_ylabel("mean-rate cost vs B2")
b.set_title("(b) Mean-rate cost",loc="left",fontsize=9.5)
for ax in (a,b): ax.grid(axis="y",color="#e3e6ea",lw=0.8); ax.set_axisbelow(True)
save(fig,"figF4_price_of_collusion_resistance")

# ---------- F8 achievable collusion level (X7; source: x7_out/f8_source_data.csv)
fam=["WS","ER","grid","BA","SNDlib","ring"]
D={"ER":(1.86,19.69,31.77,46.69),"BA":(2.08,22.80,63.12,12.00),"WS":(0.25,5.38,27.28,67.08),
   "grid":(2.45,19.65,45.77,32.13),"ring":(37.14,26.23,36.63,0.00),"SNDlib":(3.29,26.88,52.22,17.61)}
N={"ER":32000,"BA":32000,"WS":32000,"grid":32000,"ring":32000,"SNDlib":40000}
fig,ax=plt.subplots(figsize=(8.2,3.6))
segs=[("refused",GRY,""),("t = 0 only","#CFE3F2",""),("t = 1",BLUE,"//"),("t ≥ 2",ORG,"xx")]
left=[0]*len(fam)
for k,(lab,col,h) in enumerate(segs):
    vals=[D[f][k] for f in fam]
    ax.barh(range(len(fam)),vals,left=left,color=col,hatch=h,ec="white",lw=1.2,height=0.62,label=lab)
    for i,v in enumerate(vals):
        if v>=6: ax.text(left[i]+v/2,i,f"{v:.1f}",ha="center",va="center",fontsize=7.6,color="white" if col in (BLUE,ORG) else INK,
                         bbox=dict(fc=col,ec="none",pad=0.6) if h else None)
    left=[left[i]+vals[i] for i in range(len(fam))]
ax.set_yticks(range(len(fam))); ax.set_yticklabels([f"{f}  (n={N[f]:,})" for f in fam]); ax.invert_yaxis()
ax.set_xlim(0,100); ax.set_xlabel("share of requests (%)")
ax.legend(ncol=4,fontsize=7.6,frameon=False,loc="lower center",bbox_to_anchor=(0.45,1.0))
ax.grid(axis="x",color="#e3e6ea",lw=0.8); ax.set_axisbelow(True)
save(fig,"figF8_achievable_t_stacked")

# ---------- F3 per-topology supplement (X8; values from results_x7.md Section 4.2, 95% cluster bootstrap)
P=[0.02,0.05,0.10]
S={ # panel: {series: [(val,lo,hi) x3]}
 "ER (N=30)":{"SPF":[(3.47,3.39,3.54),(8.53,8.34,8.71),(16.60,16.25,16.95)],
   "B2":[(7.53,7.32,7.75),(17.73,17.26,18.19),(32.18,31.43,32.95)],
   "BMR single path":[(8.26,8.06,8.48),(19.41,18.97,19.85),(35.11,34.45,35.78)],
   "MP-SPF":[(0.16,0.16,0.17),(0.99,0.95,1.02),(3.66,3.53,3.80)],
   "BMR-MP":[(0.68,0.65,0.71),(3.72,3.57,3.87),(11.97,11.54,12.42)]},
 "BA (N=30)":{"SPF":[(3.20,3.13,3.27),(7.90,7.73,8.07),(15.46,15.15,15.78)],
   "B2":[(6.22,6.04,6.40),(14.85,14.45,15.26),(27.58,26.90,28.26)],
   "MP-SPF":[(0.17,0.16,0.17),(1.00,0.97,1.03),(3.74,3.62,3.86)],
   "BMR-MP":[(0.48,0.46,0.50),(2.69,2.58,2.79),(8.97,8.65,9.30)]},
 "Hub-and-spoke (N=12)":{"SPF = B2":[(2.89,2.89,2.89),(7.14,7.14,7.14),(13.99,13.99,13.99)],
   "BMR-MP = MP-SPF":[(0.04,0.04,0.04),(0.25,0.25,0.25),(1.00,1.00,1.00)]},
 "Ring (N=12)":{"SPF":[(7.47,7.31,7.64),(17.55,17.20,17.90),(31.76,31.21,32.32)],
   "BMR-MP = MP-SPF":[(0.68,0.66,0.70),(3.77,3.67,3.86),(12.41,12.13,12.69)]}}
sty={"SPF":(MUT,"o","-"),"SPF = B2":(MUT,"o","-"),"B2":(GRY,"s","--"),"BMR single path":(ORG,"^","-"),
     "MP-SPF":("#56B4E9","D","--"),"BMR-MP":(BLUE,"o","-"),"BMR-MP = MP-SPF":(BLUE,"o","-")}
fig,axs=plt.subplots(1,4,figsize=(11,3.1),sharey=True)
for ax,(ttl,ser) in zip(axs,S.items()):
    for nm,v in ser.items():
        c,m,ls=sty[nm]; y=[a for a,_,_ in v]; lo=[a-b for a,b,_ in v]; hi=[c_-a for a,_,c_ in v]
        ax.errorbar(P,y,yerr=[lo,hi],color=c,marker=m,ls=ls,lw=1.5,ms=5,capsize=2,label=nm)
    ax.set_title(ttl,loc="left",fontsize=9); ax.set_xticks(P); ax.set_xticklabels(["0.02","0.05","0.10"]); ax.set_xlabel("compromise probability p")
    ax.grid(axis="y",color="#e3e6ea",lw=0.8); ax.legend(fontsize=6.8,frameon=False,loc="upper left")
axs[0].set_ylabel("key exposure (%), t = 1")
save(fig,"figF3s_exposure_per_topology")
