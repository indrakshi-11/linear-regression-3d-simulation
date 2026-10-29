"""
Linear Regression 3D Simulation
-------------------------------
Requirements:
    pip install numpy matplotlib

Run:
    python linear_regression_3d.py

Controls:
- Data Points slider: change dataset size.
- Noise slider: change how scattered the data is.
- Learning Rate slider: change the gradient descent step size.
- Start / Pause Training: train the regression plane step by step.
- Reset: create new data and reset model coefficients.
- New Point: highlight another data point and show its prediction.
- Drag the 3D plot with the mouse to rotate it.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers 3D projection

# --------------------------- Initial settings ---------------------------
RNG = np.random.default_rng()
TRUE_BETA = np.array([0.5, 0.6, 0.35])  # intercept, x1 coefficient, x2 coefficient
START_BETA = np.array([0.0, 0.0, 0.0])
MAX_STEPS = 250

n_points = 100
noise_level = 0.30
learning_rate = 0.03
beta = START_BETA.copy()
step_number = 0
training = False
timer = None
X1 = X2 = Y = None
selected_index = 0

# --------------------------- Data and maths -----------------------------
def make_data(n, noise):
    """Create example data with a known linear relationship."""
    x1 = RNG.uniform(-5, 5, n)
    x2 = RNG.uniform(-5, 5, n)
    y = TRUE_BETA[0] + TRUE_BETA[1] * x1 + TRUE_BETA[2] * x2
    y = y + RNG.normal(0, noise * 2.2, n)
    return x1, x2, y


def predictions():
    return beta[0] + beta[1] * X1 + beta[2] * X2


def calculate_metrics():
    pred = predictions()
    mse = float(np.mean((Y - pred) ** 2))
    total = float(np.sum((Y - np.mean(Y)) ** 2))
    r2 = 1.0 - float(np.sum((Y - pred) ** 2)) / total if total else 0.0
    return mse, r2


def one_training_step():
    """One batch gradient-descent update for mean squared error."""
    global beta, step_number, training
    pred = predictions()
    error = pred - Y
    n = len(Y)
    gradients = np.array([
        2 * np.mean(error),
        2 * np.mean(error * X1),
        2 * np.mean(error * X2)
    ])
    beta = beta - learning_rate * gradients
    step_number += 1
    redraw()
    if step_number >= MAX_STEPS:
        pause_training()


# ------------------------------ Plot setup -----------------------------
fig = plt.figure(figsize=(13, 8), facecolor="#071126")
fig.canvas.manager.set_window_title("AI/ML Simulator - Linear Regression 3D")
ax = fig.add_axes([0.06, 0.24, 0.60, 0.68], projection="3d")
ax.set_facecolor("#071126")

fig.text(0.06, 0.94, "AI/ML SIMULATOR", color="#6fa3ff",
         fontsize=11, fontweight="bold")
fig.text(0.06, 0.895, "Linear Regression in 3D",
         color="#f0f5ff", fontsize=22, fontweight="bold")
fig.text(0.69, 0.91, "MODEL INFORMATION", color="#8eaef0",
         fontsize=10, fontweight="bold")

equation_text = fig.text(0.69, 0.86, "", color="#f2f5ff", fontsize=12)
metrics_text = fig.text(0.69, 0.72, "", color="#b9c9ec",
                         fontsize=11, linespacing=1.8)
prediction_text = fig.text(0.69, 0.48, "", color="#b9c9ec",
                           fontsize=11, linespacing=1.8)
status_text = fig.text(0.69, 0.34, "", color="#53dfb6",
                       fontsize=10, linespacing=1.5)

# Slider areas
ax_n = fig.add_axes([0.10, 0.155, 0.36, 0.025], facecolor="#15264a")
ax_noise = fig.add_axes([0.10, 0.105, 0.36, 0.025], facecolor="#15264a")
ax_rate = fig.add_axes([0.10, 0.055, 0.36, 0.025], facecolor="#15264a")

slider_n = Slider(ax_n, "Data points", 20, 300, valinit=n_points,
                  valstep=10, color="#477fff")
slider_noise = Slider(ax_noise, "Noise", 0, 1.5, valinit=noise_level,
                      valstep=0.05, color="#477fff")
slider_rate = Slider(ax_rate, "Learning rate", 0.001, 0.1,
                     valinit=learning_rate, valstep=0.001, color="#8c70ff")

# Buttons
ax_train = fig.add_axes([0.53, 0.105, 0.13, 0.055])
ax_reset = fig.add_axes([0.53, 0.035, 0.13, 0.055])
ax_new = fig.add_axes([0.69, 0.20, 0.23, 0.06])
btn_train = Button(ax_train, "Start Training", color="#365ee8", hovercolor="#5679ff")
btn_reset = Button(ax_reset, "Reset", color="#16294d", hovercolor="#294477")
btn_new = Button(ax_new, "Select Another Point", color="#16294d", hovercolor="#294477")

# ------------------------------- Drawing --------------------------------
def redraw():
    ax.clear()
    ax.set_facecolor("#071126")
    ax.grid(True, color="#253b62", alpha=0.65)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.025, 0.06, 0.13, 1.0))
        axis._axinfo["grid"]["color"] = (0.16, 0.25, 0.42, 0.55)

    ax.set_xlabel("X1", color="#dbe7ff", labelpad=8)
    ax.set_ylabel("X2", color="#dbe7ff", labelpad=8)
    ax.set_zlabel("Target Y", color="#dbe7ff", labelpad=8)
    ax.tick_params(colors="#b5c9ee")
    ax.set_title("Data points and current regression plane",
                 color="#dce8ff", pad=14, fontsize=12)

    # Draw fitted plane
    grid = np.linspace(-5, 5, 18)
    gx, gy = np.meshgrid(grid, grid)
    gz = beta[0] + beta[1] * gx + beta[2] * gy
    ax.plot_surface(gx, gy, gz, color="#35d4ae", alpha=0.28,
                    edgecolor="#49d9b7", linewidth=0.25, antialiased=True)

    # Data points
    ax.scatter(X1, X2, Y, color="#328aff", s=20, alpha=0.82,
               depthshade=True, label="Observed data")

    # Selected point and residual line
    x1s, x2s, ys = X1[selected_index], X2[selected_index], Y[selected_index]
    predicted = beta[0] + beta[1] * x1s + beta[2] * x2s
    ax.scatter([x1s], [x2s], [ys], color="#ff4e78", s=85,
               edgecolor="white", linewidth=0.8, label="Selected point")
    ax.plot([x1s, x1s], [x2s, x2s], [ys, predicted],
            color="#ff4e78", linewidth=2, linestyle="--")

    ax.view_init(elev=25, azim=-55)
    ax.legend(loc="upper left", facecolor="#0b1b38", labelcolor="white",
              framealpha=0.9, fontsize=8)

    mse, r2 = calculate_metrics()
    equation_text.set_text(
        f"ŷ = {beta[0]:.3f} + ({beta[1]:.3f}) X1 + ({beta[2]:.3f}) X2"
    )
    metrics_text.set_text(
        f"Training step: {step_number} / {MAX_STEPS}\n"
        f"Mean Squared Error: {mse:.4f}\n"
        f"R² score: {r2:.4f}\n\n"
        f"True coefficients (used to make data):\n"
        f"Intercept = {TRUE_BETA[0]:.2f}\n"
        f"X1 coefficient = {TRUE_BETA[1]:.2f}\n"
        f"X2 coefficient = {TRUE_BETA[2]:.2f}"
    )
    prediction_text.set_text(
        f"SELECTED POINT\n"
        f"X1: {x1s:.2f}     X2: {x2s:.2f}\n"
        f"Actual Y: {ys:.2f}\n"
        f"Predicted Y: {predicted:.2f}\n"
        f"Absolute error: {abs(ys - predicted):.2f}"
    )
    if training:
        status_text.set_text("Training is running…")
    elif step_number >= MAX_STEPS:
        status_text.set_text("Training complete.")
    else:
        status_text.set_text("Adjust the controls, then start training.")
    fig.canvas.draw_idle()


# ------------------------------ Interactions ----------------------------
def pause_training():
    global training, timer
    training = False
    if timer is not None:
        timer.stop()
        timer = None
    btn_train.label.set_text("Resume Training" if step_number else "Start Training")
    fig.canvas.draw_idle()


def toggle_training(_event):
    global training, timer
    if training:
        pause_training()
        return
    if step_number >= MAX_STEPS:
        return
    training = True
    btn_train.label.set_text("Pause Training")
    timer = fig.canvas.new_timer(interval=45)
    timer.add_callback(one_training_step)
    timer.start()
    redraw()


def regenerate(_value=None):
    global X1, X2, Y, beta, step_number, selected_index
    pause_training()
    n = int(slider_n.val)
    noise = float(slider_noise.val)
    X1, X2, Y = make_data(n, noise)
    beta = START_BETA.copy()
    step_number = 0
    selected_index = int(RNG.integers(0, n))
    redraw()


def reset_all(_event):
    slider_n.set_val(100)
    slider_noise.set_val(0.30)
    slider_rate.set_val(0.03)
    regenerate()


def choose_point(_event):
    global selected_index
    selected_index = int(RNG.integers(0, len(Y)))
    redraw()


def update_rate(value):
    global learning_rate
    learning_rate = float(value)


btn_train.on_clicked(toggle_training)
btn_reset.on_clicked(reset_all)
btn_new.on_clicked(choose_point)
slider_n.on_changed(regenerate)
slider_noise.on_changed(regenerate)
slider_rate.on_changed(update_rate)

# Initial dataset and render
X1, X2, Y = make_data(n_points, noise_level)
selected_index = int(RNG.integers(0, n_points))
redraw()
plt.show()
