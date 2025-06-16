import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tabulate import tabulate
from matplotlib.backends.backend_pdf import PdfPages

# === PDF REPORT SETUP ===
pdf = PdfPages("Impact_Trade_Report.pdf")

def table_to_figure(table_data, headers, title):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.axis('off')
    table_str = tabulate(table_data, headers=headers, tablefmt="pretty")
    ax.text(0.01, 0.99, f"{title}\n\n{table_str}", va='top', ha='left', fontsize=8, family='monospace')
    return fig

# === CONSTANTS ===
mu_moon = 4.9048695e12  # m^3/s^2
R_moon = 1737.4e3       # m
altitude = 0
r0_mag = R_moon + altitude

# === TRADE STUDY PARAMETERS (USED GLOBALLY) ===
v0_vals = np.arange(100, 2400, 100)
angles_deg = np.arange(1.0, 85.0, 5.0)

# === SHARED FIXED VALUES (AVERAGES FROM PARAMETER RANGE) ===
v0_fixed = np.quantile(v0_vals, 0.89)
theta_deg_fixed = np.mean(angles_deg)
theta_rad_fixed = np.radians(theta_deg_fixed)

# === SHARED FUNCTIONS ===
def two_body_equations(t, y):
    rx, ry, vx, vy = y
    r = np.hypot(rx, ry)
    ax = -mu_moon * rx / r**3
    ay = -mu_moon * ry / r**3
    return [vx, vy, ax, ay]

def impact_event(t, y):
    return np.hypot(y[0], y[1]) - R_moon
impact_event.terminal = True
impact_event.direction = -1

# === SECTION 1: VELOCITY SWEEP (Fixed Angle) ===
colors_v = plt.cm.viridis(np.linspace(0, 1, len(v0_vals)))
table_v = []

a_max = 0 # used for plot scaling.

fig_v, ax_v = plt.subplots(figsize=(6, 6))
for v0_mag, color in zip(v0_vals, colors_v):
    v0_vec = v0_mag * np.array([np.cos(theta_rad_fixed), np.sin(theta_rad_fixed)])
    r0_vec = np.array([0, r0_mag])
    y0 = np.hstack((r0_vec, v0_vec))

    v0_sq = np.dot(v0_vec, v0_vec)
    energy = v0_sq / 2 - mu_moon / np.linalg.norm(r0_vec)
    a = -mu_moon / (2 * energy) if energy < 0 else None
    a_max = a if a > a_max else a_max
    T = 2 * np.pi * np.sqrt(a**3 / mu_moon) if a else None
    t_max = T + 10000 if T else 10000

    sol = solve_ivp(two_body_equations, (0, t_max), y0,
                    events=impact_event, rtol=1e-8, atol=1e-8)

    x, y = sol.y[0], sol.y[1]
    impact_time = sol.t_events[0][0] if sol.t_events[0].size > 0 else None
    impact_pos = sol.y_events[0][0][:2] if sol.t_events[0].size > 0 else None
    flight_distance = np.linalg.norm(impact_pos - r0_vec) if impact_pos is not None else None

    ax_v.plot(x / 1e3, y / 1e3, label=f'v0 = {v0_mag} m/s', color=color)
    if impact_pos is not None:
        ax_v.plot(impact_pos[0] / 1e3, impact_pos[1] / 1e3, 'o', color=color)

    table_v.append([
        v0_mag,
        f"{impact_time:.2f}" if impact_time else "N/A",
        f"{flight_distance / 1e3:.2f}" if flight_distance else "N/A"
    ])

moon = plt.Circle((0, 0), R_moon / 1e3, color='gray', alpha=0.3)
ax_v.add_artist(moon)
ax_v.set_title(f"Velocity Sweep (Fixed Angle = {theta_deg_fixed:.1f}°)")
ax_v.set_xlabel("x [km]")
ax_v.set_ylabel("y [km]")
ax_v.set_aspect('equal')
ax_v.legend()
ax_v.grid(True)
scale = (2*a_max)/R_moon
ax_v.set_xlim(-scale * R_moon / 1e3, scale * R_moon / 1e3)
ax_v.set_ylim(-scale * R_moon / 1e3, scale * R_moon / 1e3)
plt.tight_layout()
pdf.savefig(fig_v)
plt.close(fig_v)

fig_table_v = table_to_figure(table_v, ["v0 [m/s]", "Time to Impact [s]", "Flight Distance [km]"], "Velocity Sweep Results")
pdf.savefig(fig_table_v)
plt.close(fig_table_v)

# === SECTION 2: ANGLE SWEEP (Fixed Speed) ===
colors_a = plt.cm.plasma(np.linspace(0, 1, len(angles_deg)))
table_a = []
a_max = 0

fig_a, ax_a = plt.subplots(figsize=(6, 6))
for angle_deg, color in zip(angles_deg, colors_a):
    angle_rad = np.radians(angle_deg)
    v0_vec = v0_fixed * np.array([np.cos(angle_rad), np.sin(angle_rad)])
    r0_vec = np.array([0, r0_mag])
    y0 = np.hstack((r0_vec, v0_vec))

    v0_sq = np.dot(v0_vec, v0_vec)
    energy = v0_sq / 2 - mu_moon / np.linalg.norm(r0_vec)
    a = -mu_moon / (2 * energy) if energy < 0 else None
    a_max = a if a > a_max else a_max
    T = 2 * np.pi * np.sqrt(a**3 / mu_moon) if a else None
    t_max = T + 10000 if T else 10000

    sol = solve_ivp(two_body_equations, (0, t_max), y0,
                    events=impact_event, rtol=1e-8, atol=1e-8)

    x, y = sol.y[0], sol.y[1]
    impact_time = sol.t_events[0][0] if sol.t_events[0].size > 0 else None
    impact_pos = sol.y_events[0][0][:2] if sol.t_events[0].size > 0 else None
    flight_distance = np.linalg.norm(impact_pos - r0_vec) if impact_pos is not None else None

    ax_a.plot(x / 1e3, y / 1e3, label=f'{angle_deg:.1f}°', color=color)
    if impact_pos is not None:
        ax_a.plot(impact_pos[0] / 1e3, impact_pos[1] / 1e3, 'o', color=color)

    table_a.append([
        f"{angle_deg:.1f}",
        f"{impact_time:.2f}" if impact_time else "N/A",
        f"{flight_distance / 1e3:.2f}" if flight_distance else "N/A"
    ])

moon = plt.Circle((0, 0), R_moon / 1e3, color='gray', alpha=0.3)
ax_a.add_artist(moon)
ax_a.set_title(f"Angle Sweep (Fixed Speed = {v0_fixed:.0f} m/s)")
ax_a.set_xlabel("x [km]")
ax_a.set_ylabel("y [km]")
ax_a.set_aspect('equal')
ax_a.legend(title="Launch Angle")
ax_a.grid(True)
scale = (2*a_max)/R_moon
ax_a.set_xlim(-scale * R_moon / 1e3, scale * R_moon / 1e3)
ax_a.set_ylim(-scale * R_moon / 1e3, scale * R_moon / 1e3)
plt.tight_layout()
pdf.savefig(fig_a)
plt.close(fig_a)

fig_table_a = table_to_figure(table_a, ["Angle [deg]", "Time to Impact [s]", "Flight Distance [km]"], "Angle Sweep Results")
pdf.savefig(fig_table_a)
plt.close(fig_table_a)

# === SECTION 3: TRADE STUDY HEATMAPS ===
theta_vals_rad_trade = np.radians(angles_deg)
shape = (len(angles_deg), len(v0_vals))
impact_time = np.full(shape, np.nan)
flight_distance = np.full(shape, np.nan)
periapsis_distance = np.full(shape, np.nan)

for i, theta_rad in enumerate(theta_vals_rad_trade):
    for j, v0_mag in enumerate(v0_vals):
        v0_vec = v0_mag * np.array([np.cos(theta_rad), np.sin(theta_rad)])
        r0_vec = np.array([0, r0_mag])
        y0 = np.hstack((r0_vec, v0_vec))

        h_vec = np.cross(np.append(r0_vec, 0), np.append(v0_vec, 0))
        h = np.linalg.norm(h_vec)

        v0_sq = np.dot(v0_vec, v0_vec)
        r0_norm = np.linalg.norm(r0_vec)
        energy = v0_sq / 2 - mu_moon / r0_norm
        a = -mu_moon / (2 * energy) if energy < 0 else np.nan
        T = 2 * np.pi * np.sqrt(a**3 / mu_moon) if np.isfinite(a) else 10000
        t_max = T + 10000

        sol = solve_ivp(two_body_equations, (0, t_max), y0,
                        events=impact_event, rtol=1e-8, atol=1e-8)

        if sol.t_events[0].size > 0:
            t_impact = sol.t_events[0][0]
            r_impact = sol.y_events[0][0][:2]
            delta_r = r_impact - r0_vec

            impact_time[i, j] = t_impact
            flight_distance[i, j] = np.linalg.norm(delta_r) / 1e3
            if energy < 0:
                periapsis_distance[i, j] = a * (1 - np.sqrt(1 + (2 * energy * h**2) / (mu_moon**2)))

def plot_heatmap(data, title, cmap='viridis', units='', vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, origin='lower', aspect='auto', cmap=cmap,
                   extent=[v0_vals[0], v0_vals[-1], angles_deg[0], angles_deg[-1]],
                   vmin=vmin, vmax=vmax)
    ax.set_xlabel("Initial Speed [m/s]")
    ax.set_ylabel("Launch Angle [deg]")
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(units)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

plot_heatmap(impact_time, "Time to Impact", units='s')
plot_heatmap(flight_distance, "Flight Distance", units='km')
plot_heatmap(periapsis_distance, "Periapsis Distance", units='m')

# === SECTION 4: SLICE PLOTS ===
def plot_slice_vs_speed(angles_to_plot, metric, metric_name, ylabel):
    fig = plt.figure(figsize=(8, 5))
    for angle in angles_to_plot:
        i_matches = np.where(np.isclose(angles_deg, angle))[0]
        if i_matches.size == 0:
            continue
        i = i_matches[0]
        plt.plot(v0_vals, metric[i], label=f'{angle:.1f}°')
    plt.xlabel("Initial Speed [m/s]")
    plt.ylabel(ylabel)
    plt.title(f"{metric_name} vs Initial Speed at Fixed Angles")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

def plot_slice_vs_angle(speeds_to_plot, metric, metric_name, ylabel):
    fig = plt.figure(figsize=(8, 5))
    for speed in speeds_to_plot:
        if speed in v0_vals:
            j = np.where(v0_vals == speed)[0][0]
            plt.plot(angles_deg, metric[:, j], label=f'{speed:.0f} m/s')
    plt.xlabel("Launch Angle [deg]")
    plt.ylabel(ylabel)
    plt.title(f"{metric_name} vs Launch Angle at Fixed Speeds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

def quantile_sample(array, quantiles):
    indices = [int(round(q * (len(array) - 1))) for q in quantiles]
    return [array[i] for i in indices]

quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
angles_to_plot = quantile_sample(angles_deg, quantiles)
speeds_to_plot = quantile_sample(v0_vals, quantiles)

plot_slice_vs_speed(angles_to_plot, impact_time, "Time to Impact", "Time to Impact [s]")
plot_slice_vs_angle(speeds_to_plot, impact_time, "Time to Impact", "Time to Impact [s]")
plot_slice_vs_speed(angles_to_plot, flight_distance, "Flight Distance", "Flight Distance [km]")
plot_slice_vs_angle(speeds_to_plot, flight_distance, "Flight Distance", "Flight Distance [km]")

# === CLOSE PDF ===
pdf.close()
print("PDF report saved as 'Impact_Trade_Report.pdf'.")
