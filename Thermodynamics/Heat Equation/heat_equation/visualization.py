import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from .solver import solve

def plot_profiles(u, x, t, steps=(0, 100, 500, 2000)):
    fig, ax = plt.subplots()
    for i in steps:
        if i < len(t):
            ax.plot(x, u[i], label=f"t = {t[i]:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.legend()
    plt.tight_layout()
    return fig

def animate(u, x, t, interval=40):
    fig, ax = plt.subplots()
    line, = ax.plot(x, u[0])
    ax.set_ylim(u.min() - 1, u.max() + 1)

    def update(frame):
        line.set_ydata(u[frame])
        ax.set_title(f"t = {t[frame]:.3f}")
        return (line,)

    return FuncAnimation(fig, update, frames=range(0, len(t), 10),
                         interval=interval, blit=True)