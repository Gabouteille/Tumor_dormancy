#!/usr/bin/env python3
#######################################################################
# memory_model.py
# Compare the Bottcher model (eta=0) with the treatment-memory model
# (eta>0) on the same axes, under the same cyclic treatment schedule.
#######################################################################

import os
import itertools
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

####model parameters
mig_spont=0.010 #sigma: base phenotype switching rate
base_growth=1
comp_growth=1/2.
epsilon=0.0 #fitness of dormant cells
N0=1
lh=1.0 #high treatment strength, also d_H used for normalisation
lw=0.25 #low treatment strength
Tn=40 #duration before any treatment
segment_interval=15
startfrequency_d=0.10 #proportion of dormant cells at start
Ts_steps=2.
Tw_steps=2.
total_segments=12

Ts=segment_interval*Ts_steps
Tw=segment_interval*Tw_steps
Tmax=total_segments*segment_interval+Tn

####memory parameters
eta_memory=0.5 #memory strength, 0 <= eta <= 1
tau_m_memory=10. #memory timescale
m0=0. #memory at start

####plot parameters
color_rp="#ECC000"
color_d="#D24E71"
color_number="#001889"
color_m="gray"

### functions

def rhs(x,t,l,eta,tau_m):
    """ right-hand side dx/dt for x=[xD,xP,m] with treatment strength l """
    xD,xP,m=x
    u=l/lh #normalised treatment exposure
    fP=(base_growth-l-epsilon)*xD+(comp_growth-l)*xP
    fbar=epsilon*xD+fP*xP
    K_PD=mig_spont*(1+eta*m) #rate P -> D
    K_DP=mig_spont*(1-eta*m) #rate D -> P
    dxD=xD*(epsilon-fbar)+K_PD*xP-K_DP*xD
    dxP=xP*(fP-fbar)+K_DP*xD-K_PD*xP
    dm=(u-m)/tau_m
    return [dxD,dxP,dm]

def treatment_schedule():
    """ list of (duration, treatment strength) segments up to Tmax """
    schedule=[(Tn,0.)]
    elapsed=Tn
    cycle=itertools.cycle([(Ts,lh),(Tw,lw)])
    while elapsed<Tmax:
        dur,l=next(cycle)
        dur=min(dur,Tmax-elapsed)
        schedule.append((dur,l))
        elapsed+=dur
    return schedule

def simulate(eta,tau_m,dt=0.05):
    """ integrate the model segment by segment, returns t, x=[xD,xP,m], l """
    x=np.array([startfrequency_d,1.-startfrequency_d,m0])
    t_start=0.
    t_all=[]
    x_all=[]
    l_all=[]
    for dur,l in treatment_schedule():
        t_seg=np.linspace(t_start,t_start+dur,int(round(dur/dt))+1)
        sol=scipy.integrate.odeint(rhs,x,t_seg,args=(l,eta,tau_m),rtol=1e-11,atol=1e-11,mxstep=5000000)
        if t_all: #the first point of a segment is the last point of the previous one
            t_seg=t_seg[1:]
            sol=sol[1:]
        t_all.append(t_seg)
        x_all.append(sol)
        l_all.append(np.full(len(t_seg),l))
        x=sol[-1]
        t_start=t_seg[-1]
    return np.concatenate(t_all),np.concatenate(x_all),np.concatenate(l_all)

def derived(t,x,l):
    """ fractions, average fitness and number of cells along the trajectory """
    xD=x[:,0]
    xP=x[:,1]
    m=x[:,2]
    fP=(base_growth-l-epsilon)*xD+(comp_growth-l)*xP
    fbar=epsilon*xD+fP*xP
    N=N0*np.exp(scipy.integrate.cumulative_trapezoid(fbar,t,initial=0))
    return xD,xP,m,fbar,N

### simulation and plots

models=[("Bottcher (eta=0)",0.,tau_m_memory,"-"),
        ("Memoire (eta={})".format(eta_memory),eta_memory,tau_m_memory,"--")]

fig,ax=plt.subplots(5,1,sharex=True,figsize=(7,11))

results={}
for name,eta,tau_m,style in models:
    t,x,l=simulate(eta,tau_m)
    xD,xP,m,fbar,N=derived(t,x,l)
    results[eta]=(t,xP,fbar,N)
    print(name,"max |xD+xP-1| =",np.max(np.abs(xD+xP-1)))

    ax[0].plot(t,xP,linestyle=style,linewidth=1.8,color=color_rp,label="P "+name)
    ax[0].plot(t,xD,linestyle=style,linewidth=1.8,color=color_d,label="D "+name)
    ax[1].plot(t,fbar,linestyle=style,linewidth=1.8,color="black",label=name)
    ax[2].plot(t,N,linestyle=style,linewidth=1.8,color=color_number,label=name)
    if eta>0:
        ax[3].plot(t,m,linestyle=style,linewidth=1.8,color=color_m,label=name)

# comparison with the base model (eta=0)
t,xP_b,fbar_b,N_b=results[0.]
_,xP_m,fbar_m,N_m=results[eta_memory]
ratio=N_m/N_b
ax[4].plot(t,ratio,linewidth=1.8,color=color_number,label="N memoire / N Bottcher")
ax[4].axhline(1.,linewidth=0.4,color="black")
print("max |xP_mem - xP_base| =",np.max(np.abs(xP_m-xP_b)),"at t =",t[np.argmax(np.abs(xP_m-xP_b))])
print("max |fbar_mem - fbar_base| =",np.max(np.abs(fbar_m-fbar_b)))
print("final N ratio memoire/Bottcher =",ratio[-1])

ax[0].set_ylabel("Pop. fraction")
ax[1].set_ylabel("Average fitness")
ax[1].axhline(0.,linewidth=0.4,color="black")
ax[2].set_ylabel("# of cells")
ax[2].set_yscale("log")
ax[3].set_ylabel("Memory m(t)")
ax[4].set_ylabel("Ratio to base")
ax[4].set_yscale("log")
ax[4].set_xlabel("Time")
ax[4].set_xlim(0,Tmax)

for axi in ax:
    axi.legend(fontsize="small")

fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),"memory_comparison.png"),dpi=150)
