#!/usr/bin/env python3
'''one-at-a-time sweeps of lw, comp_growth, mig_spont and epsilon, using the plot_dynamics
function of plot_multi_egtdyn_fullhelpers.py (its module-level plotting loop is not run).'''
import os
import matplotlib
matplotlib.use("Agg")

SRC = "plot_multi_egtdyn_fullhelpers.py"
OUTDIR = "sweep"

# source of the plotting module, without its final plotting loop
src = open(SRC).read()
body = src[:src.index("#do the actual plotting")]


def sub(old, new):
    global body
    assert body.count(old) == 1, old
    body = body.replace(old, new)


# lw and comp_growth are hard-coded in plot_dynamics: read them from the namespace instead
sub("lw=0.25 #weak treatment strength", "lw=LW #weak treatment strength")
sub("comp_growth=1./2.", "comp_growth=COMP")

ns = {"__name__": "sweep"}
exec(compile(body, SRC, "exec"), ns)
plot_dynamics = ns["plot_dynamics"]

# baseline of egt_cancer_dyn.py, with the cycle duration of that script (Ts_steps=2)
base = dict(LW=0.25, COMP=0.5, lh=1.0, cycle_duration=2, mig=0.01, e=0.0)

sweeps = {
    "lw": [0.0, 0.1, 0.25, 0.4, 0.5],
    "comp_growth": [0.25, 0.375, 0.5, 0.625, 0.75],
    "mig_spont": [0.001, 0.005, 0.01, 0.05, 0.1],
    "epsilon": [0.0, 0.05, 0.1, 0.15, 0.2],
}
keymap = {"lw": "LW", "comp_growth": "COMP", "mig_spont": "mig", "epsilon": "e"}

os.makedirs(OUTDIR, exist_ok=True)
for name, values in sweeps.items():
    for v in values:
        kw = dict(base)
        kw[keymap[name]] = v
        ns["LW"], ns["COMP"] = kw["LW"], kw["COMP"]
        plot_dynamics(lh=kw["lh"], cycle_duration=kw["cycle_duration"], mig=kw["mig"], e=kw["e"],
                      outfname=os.path.join(OUTDIR, f"{name}_{v}.pdf"), label="xy")
        print("done", name, v)
