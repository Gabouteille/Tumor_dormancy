#!/usr/bin/env python3
#######################################################################
# memory_horizons.py
# Compare the Bottcher model (eta=0) with the treatment-memory model
# (eta>0) over several simulation lengths (1, 2, 3, 4, 6, 9 treatment cycles).
#######################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import memory_model as mm

outdir=os.path.dirname(os.path.abspath(__file__))

# one treatment cycle = Ts + Tw = 60 time units = 4 segments of 15
segments_list=[4,8,12,16,24,36]

def run(total_segments):
    """ simulate base and memory model for a given number of segments """
    mm.total_segments=total_segments
    mm.Tmax=total_segments*mm.segment_interval+mm.Tn
    tb,xb,lb=mm.simulate(0.,mm.tau_m_memory)
    tm,xm,lm=mm.simulate(mm.eta_memory,mm.tau_m_memory)
    _,_,_,fb,Nb=mm.derived(tb,xb,lb)
    _,_,mem,fm,Nm=mm.derived(tm,xm,lm)
    return dict(t=tb,xP_b=xb[:,1],xP_m=xm[:,1],fb=fb,fm=fm,
                Nb=Nb,Nm=Nm,m=mem,Tmax=mm.Tmax)

results=[run(s) for s in segments_list]

### one figure per horizon
for r in results:
    fig,ax=plt.subplots(5,1,sharex=True,figsize=(7,11))
    t=r["t"]
    ax[0].plot(t,r["xP_b"],"-",linewidth=1.8,color=mm.color_rp,label="P Bottcher (eta=0)")
    ax[0].plot(t,r["xP_m"],"--",linewidth=1.8,color=mm.color_rp,label="P Memoire (eta={})".format(mm.eta_memory))
    ax[1].plot(t,r["fb"],"-",linewidth=1.8,color="black",label="Bottcher (eta=0)")
    ax[1].plot(t,r["fm"],"--",linewidth=1.8,color="black",label="Memoire (eta={})".format(mm.eta_memory))
    ax[2].plot(t,r["Nb"],"-",linewidth=1.8,color=mm.color_number,label="Bottcher (eta=0)")
    ax[2].plot(t,r["Nm"],"--",linewidth=1.8,color=mm.color_number,label="Memoire (eta={})".format(mm.eta_memory))
    ax[3].plot(t,r["m"],"--",linewidth=1.8,color=mm.color_m,label="Memoire (eta={})".format(mm.eta_memory))
    ax[4].plot(t,r["Nm"]/r["Nb"],linewidth=1.8,color=mm.color_number,label="N memoire / N Bottcher")
    ax[4].axhline(1.,linewidth=0.4,color="black")

    ax[0].set_ylabel("Pop. fraction")
    ax[1].set_ylabel("Average fitness")
    ax[1].axhline(0.,linewidth=0.4,color="black")
    ax[2].set_ylabel("# of cells")
    ax[2].set_yscale("log")
    ax[3].set_ylabel("Memory m(t)")
    ax[4].set_ylabel("Ratio to base")
    ax[4].set_xlabel("Time")
    ax[4].set_xlim(0,r["Tmax"])
    for axi in ax:
        axi.legend(fontsize="small")
    fig.suptitle("Horizon: {:.0f} time units ({:.1f} cycles)".format(r["Tmax"],(r["Tmax"]-mm.Tn)/60.))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir,"memory_horizon_T{:.0f}.png".format(r["Tmax"])),dpi=150)
    plt.close(fig)

### summary: ratio of cell numbers at the end of each horizon vs number of cycles
cycles=[(r["Tmax"]-mm.Tn)/60. for r in results]
ratio_end=[r["Nm"][-1]/r["Nb"][-1] for r in results]
fig,ax=plt.subplots(1,1,figsize=(6,4))
ax.plot(cycles,ratio_end,"o-",linewidth=1.8,color=mm.color_number)
for c,q in zip(cycles,ratio_end):
    ax.annotate("{:.2f}".format(q),(c,q),textcoords="offset points",xytext=(0,6),ha="center",fontsize="small")
ax.axhline(1.,linewidth=0.4,color="black")
ax.set_xlabel("Treatment cycles simulated")
ax.set_ylabel("N memoire / N Bottcher at end")
ax.set_yscale("log")
fig.tight_layout()
fig.savefig(os.path.join(outdir,"memory_horizon_ratio_summary.png"),dpi=150)
plt.close(fig)

print("{:>8} {:>8} {:>12} {:>12} {:>10} {:>12}".format("Tmax","cycles","max|dP|","max|dfbar|","ratio_end","max_m"))
for r in results:
    dP=np.max(np.abs(r["xP_m"]-r["xP_b"]))
    dF=np.max(np.abs(r["fm"]-r["fb"]))
    print("{:8.0f} {:8.1f} {:12.4f} {:12.4f} {:10.3f} {:12.3f}".format(
        r["Tmax"],(r["Tmax"]-mm.Tn)/60.,dP,dF,r["Nm"][-1]/r["Nb"][-1],r["m"].max()))

### focus horizons: T=280 (16 segments) and T=580 (36 segments)
focus_segments=[16,36]
etas=[0.25,0.75,0.95]
taus=[0.25,0.75,0.95] # zeta_m read as the memory timescale tau_m

def simulate_horizon(total_segments,eta,tau_m=mm.tau_m_memory,lh=mm.lh):
    """ simulate one horizon with its own eta, tau_m and treatment strength lh """
    mm.total_segments=total_segments
    mm.Tmax=total_segments*mm.segment_interval+mm.Tn
    mm.lh=lh
    return mm.simulate(eta,tau_m)

def rates(m,eta):
    """ K_PD and K_DP along a memory trajectory m(t) """
    return mm.mig_spont*(1+eta*m),mm.mig_spont*(1-eta*m)

for s in focus_segments:
    Tmax=s*mm.segment_interval+mm.Tn

    # 1) K_PD and K_DP: top row for several eta, bottom row for several tau_m
    fig,ax=plt.subplots(2,2,sharex=True,figsize=(10,7))
    for eta in etas:
        t,x,l=simulate_horizon(s,eta)
        kpd,kdp=rates(x[:,2],eta)
        ax[0,0].plot(t,kpd,linewidth=1.8,label="eta={}".format(eta))
        ax[0,1].plot(t,kdp,linewidth=1.8,label="eta={}".format(eta))
    for tau in taus:
        t,x,l=simulate_horizon(s,mm.eta_memory,tau)
        kpd,kdp=rates(x[:,2],mm.eta_memory)
        ax[1,0].plot(t,kpd,linewidth=1.8,label="tau_m={}".format(tau))
        ax[1,1].plot(t,kdp,linewidth=1.8,label="tau_m={}".format(tau))
    ax[0,0].set_title("K_PD, varying eta (tau_m={})".format(mm.tau_m_memory))
    ax[0,1].set_title("K_DP, varying eta (tau_m={})".format(mm.tau_m_memory))
    ax[1,0].set_title("K_PD, varying tau_m (eta={})".format(mm.eta_memory))
    ax[1,1].set_title("K_DP, varying tau_m (eta={})".format(mm.eta_memory))
    for axi in ax.flat:
        axi.legend(fontsize="small")
    ax[1,0].set_xlabel("Time")
    ax[1,1].set_xlabel("Time")
    ax[1,0].set_xlim(0,Tmax)
    fig.suptitle("Transition rates, horizon {:.0f}".format(Tmax))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir,"memory_rates_T{:.0f}.png".format(Tmax)),dpi=150)
    plt.close(fig)

    # 2) xD, xP and m superposed, one line style per eta
    fig,ax=plt.subplots(1,1,figsize=(8,5))
    for eta,style in zip(etas,["-","--",":"]):
        t,x,l=simulate_horizon(s,eta)
        ax.plot(t,x[:,0],style,linewidth=1.8,color=mm.color_d,label="xD eta={}".format(eta))
        ax.plot(t,x[:,1],style,linewidth=1.8,color=mm.color_rp,label="xP eta={}".format(eta))
        ax.plot(t,x[:,2],style,linewidth=1.8,color=mm.color_m,label="m eta={}".format(eta))
    ax.set_xlabel("Time")
    ax.set_xlim(0,Tmax)
    ax.legend(fontsize="small")
    fig.suptitle("xD, xP, m superimposed, horizon {:.0f}".format(Tmax))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir,"memory_xdxpm_T{:.0f}.png".format(Tmax)),dpi=150)
    plt.close(fig)

    # 3) m(t) for eta=0.5 with lh=1 and lh=0.5
    fig,ax=plt.subplots(1,1,figsize=(8,5))
    for lh_val,style in [(1.0,"-"),(0.5,"--")]:
        t,x,l=simulate_horizon(s,mm.eta_memory,mm.tau_m_memory,lh=lh_val)
        ax.plot(t,x[:,2],style,linewidth=1.8,color=mm.color_m,label="m, lh={}".format(lh_val))
    ax.set_xlabel("Time")
    ax.set_ylabel("Memory m(t)")
    ax.set_xlim(0,Tmax)
    ax.legend(fontsize="small")
    fig.suptitle("m(t), eta={}, horizon {:.0f}".format(mm.eta_memory,Tmax))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir,"memory_m_lh_T{:.0f}.png".format(Tmax)),dpi=150)
    plt.close(fig)
