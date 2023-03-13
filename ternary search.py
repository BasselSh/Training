import numpy as np
import matplotlib.pyplot as plt

eps = 0.001

def ter(f, l, r, ax):
    ll, rr = l, r
    while abs(ll - rr) >= eps:
        fl = f(ll)
        fr = f(rr)
        lm = ll + (rr - ll) / 3.0
        rm = rr - (rr - ll) / 3.0
        flm = f(lm)
        frm = f(rm)
        if flm < frm:
            rr = rm
        else:
            ll = lm
        fig, ax = plt.subplots()
        ax.plot(aa, funnp)
        ax.plot([ll, ll], [f(ll), f(ll) + 1000], color='green')
        ax.plot([rr, rr], [f(rr), f(rr) + 1000], color='green')
        plt.show()
    return (ll + rr) / 2

def sq(x):
    return (x - 5) ** 2 + 10

g = 50
aa = np.linspace(-g, g, 100)
funnp = sq(aa)
fig, ax = plt.subplots()
ax.plot(aa, funnp)
#ax.set_ylim([-40, 40])
minval = ter(sq, -g, g, ax)
plt.show()