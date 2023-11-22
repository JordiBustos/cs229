import numpy as np
import matplotlib.pyplot as plt


def update_w_and_b(spendings, sales, w, b, alpha):
    dl_dw = np.zeros_like(w)
    dl_db = 0.0
    N = len(spendings)

    for i in range(N):
        dl_dw += -2 * spendings[i] * (sales[i] - np.dot(w, spendings[i]) - b)
        dl_db += -2 * (sales[i] - np.dot(w, spendings[i]) - b)

    w = w - (1 / float(N)) * dl_dw * alpha
    b = b - (1 / float(N)) * dl_db * alpha

    return w, b


def train(spendings, sales, w, b, alpha, epochs, plot_interval=1000):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for e in range(epochs):
        w, b = update_w_and_b(spendings, sales, w, b, alpha)

        # log the progress
        if e % plot_interval == 0:
            print("epoch:", e, "loss: ", avg_loss(spendings, sales, w, b))
            plot_plane(ax, w, b, spendings, sales)

    return w, b


def avg_loss(spendings, sales, w, b):
    N = len(spendings)
    total_error = np.sum((sales - np.dot(w, spendings.T) - b) ** 2)
    return total_error / float(N)


def predict(x, w, b):
    return np.dot(w, x) + b


def plot_plane(ax, w, b, spendings, sales):
    x_min = np.min(spendings[:, 0])
    x_max = np.max(spendings[:, 0])
    y_min = np.min(spendings[:, 1])
    y_max = np.max(spendings[:, 1])

    x, y = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
    z = w[0] * x + w[1] * y + w[2] * 1 + b

    ax.clear()
    ax.scatter(spendings[:, 0], spendings[:, 1], sales, c="r", marker="o")
    ax.plot_surface(x, y, z, alpha=0.5, cmap="viridis")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Sales")
    plt.pause(0.01)


# Generate 3D random data
x = np.random.rand(200, 3)
noise = np.random.normal(0, 0.1, 1)
y = np.dot(x, np.array([2, 3, 4])) + 5 + noise


# Initialize weights and biases
w = np.zeros(3)
b = 0.0

# Train the model with normalized features
w, b = train(x, y, w, b, 0.001, 15000, plot_interval=1000)

plt.show()
