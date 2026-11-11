"""
Basic Kalman Filter simulation.

Scenario: an object moves along a line at roughly constant velocity.
We only get noisy measurements of its POSITION, and the filter estimates
both position and velocity.

Requires: numpy, matplotlib  (pip install numpy matplotlib)
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---------------------------------------------------------------
# 1. Simulation settings
# ---------------------------------------------------------------
dt = 0.1          # time step (s)
steps = 200       # number of time steps
true_vel = 1.0    # true velocity (m/s)
meas_std = 2.0    # measurement noise std dev (m)
accel_std = 0.2   # random acceleration (process noise) std dev (m/s^2)

# ---------------------------------------------------------------
# 2. Model matrices
#    State x = [position, velocity]^T
# ---------------------------------------------------------------
F = np.array([[1, dt],
              [0, 1]])            # state transition: pos += vel*dt
H = np.array([[1, 0]])            # we measure position only
G = np.array([[0.5 * dt**2],
              [dt]])              # how acceleration affects the state
Q = G @ G.T * accel_std**2        # process noise covariance
R = np.array([[meas_std**2]])     # measurement noise covariance

# ---------------------------------------------------------------
# 3. Generate ground truth and noisy measurements
# ---------------------------------------------------------------
true_states = np.zeros((steps, 2))
measurements = np.zeros(steps)
x_true = np.array([0.0, true_vel])

for k in range(steps):
    accel = np.random.randn() * accel_std
    x_true = F @ x_true + (G * accel).flatten()
    true_states[k] = x_true
    measurements[k] = x_true[0] + np.random.randn() * meas_std

# ---------------------------------------------------------------
# 4. Kalman filter
# ---------------------------------------------------------------
x = np.array([[0.0], [0.0]])      # initial guess (velocity unknown)
P = np.eye(2) * 10.0              # initial uncertainty (large = unsure)

estimates = np.zeros((steps, 2))
pos_std = np.zeros(steps)

for k in range(steps):
    # --- Predict ---
    x = F @ x
    P = F @ P @ F.T + Q

    # --- Update ---
    z = np.array([[measurements[k]]])
    y = z - H @ x                     # innovation (measurement residual)
    S = H @ P @ H.T + R               # innovation covariance
    K = P @ H.T @ np.linalg.inv(S)    # Kalman gain
    x = x + K @ y
    P = (np.eye(2) - K @ H) @ P

    estimates[k] = x.flatten()
    pos_std[k] = np.sqrt(P[0, 0])

# ---------------------------------------------------------------
# 5. Results
# ---------------------------------------------------------------
t = np.arange(steps) * dt
rmse_meas = np.sqrt(np.mean((measurements - true_states[:, 0])**2))
rmse_kf = np.sqrt(np.mean((estimates[:, 0] - true_states[:, 0])**2))
print(f"Position RMSE - raw measurements: {rmse_meas:.3f} m")
print(f"Position RMSE - Kalman estimate:  {rmse_kf:.3f} m")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

ax1.scatter(t, measurements, s=8, color="gray", alpha=0.5, label="Measurements")
ax1.plot(t, true_states[:, 0], "k-", lw=2, label="True position")
ax1.plot(t, estimates[:, 0], "r-", lw=1.5, label="Kalman estimate")
ax1.fill_between(t, estimates[:, 0] - 2 * pos_std, estimates[:, 0] + 2 * pos_std,
                 color="red", alpha=0.15, label="±2σ")
ax1.set_ylabel("Position (m)")
ax1.legend()
ax1.set_title("Kalman Filter: 1D constant-velocity tracking")

ax2.plot(t, true_states[:, 1], "k-", lw=2, label="True velocity")
ax2.plot(t, estimates[:, 1], "b-", lw=1.5, label="Estimated velocity")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Velocity (m/s)")
ax2.legend()

plt.tight_layout()
plt.savefig("kalman_sim.png", dpi=120)
plt.show()
