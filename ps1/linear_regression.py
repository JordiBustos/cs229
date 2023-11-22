import numpy as np
import matplotlib.pyplot as plt


def update_w_and_b(features, target, w, b, alpha, lambda_reg):
    dl_dw = np.zeros_like(w)
    dl_db = 0.0
    N = len(features)

    for i in range(N):
        dl_dw += -2 * features[i] * (target[i] - np.dot(w, features[i]) - b)
        dl_db += -2 * (target[i] - np.dot(w, features[i]) - b)

    w = w - (1 / float(N)) * (dl_dw + 2 * lambda_reg * w) * alpha
    b = b - (1 / float(N)) * dl_db * alpha

    return w, b


def train_gradient_descent(
    features,
    target,
    w,
    b,
    alpha,
    epochs,
    plot_interval=1000,
    lambda_reg=0.01,
    tolerance=1e-5,
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for e in range(epochs):
        w_prev = w.copy()
        w, b = update_w_and_b(features, target, w, b, alpha, lambda_reg)

        # log the progress
        if e % plot_interval == 0:
            print("epoch:", e)
            print("avg_loss:", avg_loss(features, target, w, b, lambda_reg))
            plot_plane(ax, w, b, features, target)

        # check for convergence
        if np.linalg.norm(w - w_prev) <= tolerance:
            print("Converged! Stopping training.")
            break

    return w, b


def avg_loss(features, target, w, b, lambda_reg=0.01):
    N = len(features)
    total_error = np.sum((target - np.dot(features, w) - b) ** 2)
    regularization_term = lambda_reg * np.sum(w**2)
    return (total_error + regularization_term) / float(N)


def predict(x, w, b):
    return np.dot(w, x) + b


def plot_plane(ax, w, b, features, target):
    x_min, x_max = np.min(features[:, 0]), np.max(features[:, 0])
    y_min, y_max = np.min(features[:, 1]), np.max(features[:, 1])

    x, y = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
    z = w[0] * x + w[1] * y + b  # the plane equation

    ax.clear()
    ax.scatter(features[:, 0], features[:, 1], target, c="r", marker="o")
    ax.plot_surface(x, y, z, alpha=0.5, cmap="viridis")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Target")
    plt.pause(0.01)


def generate_points(sample_size=100):
    x = np.random.rand(sample_size, 2)
    noise = np.random.normal(0, 0.5, sample_size)
    y = (
        np.dot(x, np.array([2, 3])) + 5 + noise
    )  # assume that the target is a linear function of the features
    w = np.zeros(2)
    b = 0.0

    return x, y, w, b


def compute_w_normal_equation(features, target):
    ones = np.ones((features.shape[0], 1))
    features_with_bias = np.concatenate((ones, features), axis=1)
    return np.dot(
        np.linalg.inv(np.dot(features_with_bias.T, features_with_bias)),
        np.dot(features_with_bias.T, target),
    )


if __name__ == "__main__":
    x, y, w, b = generate_points()
    w, b = train_gradient_descent(x, y, w, b, 0.001, 10000, plot_interval=1000)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    w_normal_equation = compute_w_normal_equation(x, y)
    plot_plane(ax, w_normal_equation[1:], w_normal_equation[0], x, y)

    plt.show()
    print("w_normal_equation:", w_normal_equation)
    print("w:", w)
    print("b:", b)

    plt.show()
