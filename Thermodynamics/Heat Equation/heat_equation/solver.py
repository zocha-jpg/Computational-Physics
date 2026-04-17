import numpy as np

def stability_timestep(dx: float, alpha: float) -> float:
    """dt = dx² / (4α) — gwarantuje stabilność schematu (r ≤ 0.5)."""
    return dx**2 / (4 * alpha)

def initial_condition(N: int, L: float, u_left: float, u_right: float) -> np.ndarray:
    """u₀(x) = 5 + 5·sin(3πx/L) z warunkami brzegowymi."""
    x = np.linspace(0, L, N + 1)
    u0 = 5.0 + 5.0 * np.sin(3 * np.pi * x / L)
    u0[0] = u_left
    u0[-1] = u_right
    return u0

def solve(L=1.0, N=100, alpha=1e-2, tf=10.0,
          u_left=10.0, u_right=0.0):
    """
    Rozwiązuje równanie ciepła: ∂u/∂t = α ∂²u/∂x²
    Metoda: jawne różnice skończone (Euler w czasie, centralne w przestrzeni).
    Zwraca: (u, x, t) gdzie u ma kształt (Nt, N+1).
    """
    dx = L / N
    dt = stability_timestep(dx, alpha)
    r  = alpha * dt / dx**2   # ≤ 0.5 z definicji

    x  = np.linspace(0, L, N + 1)
    Nt = int(tf / dt) + 1
    t  = np.arange(Nt) * dt

    u        = np.zeros((Nt, N + 1))
    u[0]     = initial_condition(N, L, u_left, u_right)
    u[:, 0]  = u_left
    u[:, -1] = u_right

    for n in range(Nt - 1):
        u[n+1, 1:-1] = u[n, 1:-1] + r * (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2])

    return u, x, t