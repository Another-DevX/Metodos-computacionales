import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit
from multiprocessing import cpu_count

# Parámetros globales
mu = 0.0121505856
dt = 0.001
steps = 5000000

# ===========================
# 1. Simulador RTBP (Leap-Frog)
# ===========================
def restricted_three_body_leap_frog(state, mu, dt, softening=0):
    x, y, z, vx, vy, vz = state
    r = np.array([x, y, z])
    v = np.array([vx, vy, vz])
    r1 = np.array([-mu, 0, 0])
    r2 = np.array([1 - mu, 0, 0])

    r13 = r - r1
    r23 = r - r2
    d13 = np.sqrt(np.dot(r13, r13) + softening**2)
    d23 = np.sqrt(np.dot(r23, r23) + softening**2)

    a = (
        - (1 - mu) * r13 / d13**3
        - mu * r23 / d23**3
        + np.array([2 * vy + x, -2 * vx + y, 0])
    )

    v_half = v + 0.5 * a * dt
    r_new = r + v_half * dt

    r13_new = r_new - r1
    r23_new = r_new - r2
    d13_new = np.sqrt(np.dot(r13_new, r13_new) + softening**2)
    d23_new = np.sqrt(np.dot(r23_new, r23_new) + softening**2)

    a_new = (
        - (1 - mu) * r13_new / d13_new**3
        - mu * r23_new / d23_new**3
        + np.array([2 * v_half[1] + r_new[0], -2 * v_half[0] + r_new[1], 0])
    )

    v_new = v_half + 0.5 * a_new * dt

    return np.concatenate((r_new, v_new))

def solve_rtbp(mu, state0, dt, steps, softening=1e-5):
    traj = np.zeros((steps, len(state0)))
    traj[0] = state0
    for i in range(1, steps):
        traj[i] = restricted_three_body_leap_frog(traj[i-1], mu, dt, softening)
    return traj

# ===========================
# 2. Cálculo eficiente de cruces (con numba)
# ===========================
@njit
def extract_poincare_points(x, y, vx, vy):
    points = []
    for i in range(len(y) - 1):
        if y[i] < 0 and y[i+1] > 0 and vy[i] > 0 and vy[i+1] > 0:
            frac = -y[i] / (y[i+1] - y[i])
            x_p = x[i] + frac * (x[i+1] - x[i])
            vx_p = vx[i] + frac * (vx[i+1] - vx[i])
            points.append((x_p, vx_p))
    return points

# ===========================
# 3. Función paralelizable
# ===========================
def compute_single_poincare_rtbp(velocity):
    vx0, vy0 = velocity
    state0 = np.array([0.8, 0.0, 0.0, vx0, vy0, 0.0])
    traj = solve_rtbp(mu, state0, dt, steps)

    y = traj[:, 1]
    if np.all(y > 0) or np.all(y < 0):
        return []

    x = traj[:, 0]
    vx = traj[:, 3]
    vy = traj[:, 4]

    return extract_poincare_points(x, y, vx, vy)

# ===========================
# 4. Paralelización y ejecución
# ===========================
def compute_poincare_section_rtbp_parallel():
    N = 100
    print(f"Simulando {N*N} condiciones iniciales...")
    vx_vals = np.linspace(-0.2, 0.2, N)
    vy_vals = np.linspace(0.1, 0.4, N)
    velocities = [(vx, vy) for vx in vx_vals for vy in vy_vals]

    print(f"Simulando {len(velocities)} condiciones iniciales...")
    results = Parallel(n_jobs=cpu_count())(
        delayed(compute_single_poincare_rtbp)(v) for v in tqdm(velocities)
    )

    x_poincare, vx_poincare = [], []
    for points in results:
        for x, vx in points:
            x_poincare.append(x)
            vx_poincare.append(vx)
    return np.array(x_poincare), np.array(vx_poincare)

# ===========================
# 5. Visualización y guardado
# ===========================
# def plot_poincare_section(x_p, vx_p):
#     plt.figure(figsize=(8, 6))
#     plt.scatter(x_p, vx_p, s=1, alpha=0.5, color='black')
#     plt.xlabel(r"$x$")
#     plt.ylabel(r"$v_x$")
#     plt.title("Sección de Poincaré - Problema de Tres Cuerpos Restringido")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

def save_poincare_to_csv(x_p, vx_p, filename="poincare_rtbp.csv"):
    df = pd.DataFrame({"x": x_p, "vx": vx_p})
    df.to_csv(filename, index=False)
    print(f"Guardado en {filename}")

# ===========================
# Ejecutar todo
# ===========================
x_poincare, vx_poincare = compute_poincare_section_rtbp_parallel()
# plot_poincare_section(x_poincare, vx_poincare)
save_poincare_to_csv(x_poincare, vx_poincare)
