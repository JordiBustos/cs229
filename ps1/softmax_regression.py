import numpy as np


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def one_hot_encoding(y, num_classes):
    return np.eye(num_classes)[y]


def cross_entropy(output, y_target):
    return -np.sum(np.log(output) * (y_target), axis=1)


def compute_accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))


def gradient_descent(X, y, theta, alpha, num_iterations):
    m = len(y)
    cost_history = []

    for i in range(num_iterations):
        h = softmax(np.dot(X, theta))
        theta = theta - (alpha / m) * np.dot(X.T, (h - y))

        # Compute and store the cost
        cost = cross_entropy(h, y)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}, Cost: {cost}")

    return theta, cost_history


def softmax_regression(X, y, alpha=0.1, num_iterations=2000):
    num_classes = len(np.unique(y))
    y_encoded = one_hot_encoding(y, num_classes)
    theta = np.random.randn(X.shape[1], num_classes) * 0.01

    # Compute mean and std only on the training data
    mean_X = np.mean(X, axis=0)
    std_X = np.std(X, axis=0)

    X = (X - mean_X) / std_X

    theta, cost_history = gradient_descent(X, y_encoded, theta, alpha, num_iterations)
    return theta, mean_X, std_X, cost_history


def predict(X, theta):
    return np.argmax(softmax(np.dot(X, theta)), axis=1)


def generate_points(num_samples_per_class=2000, num_classes=3):
    x = []
    y = []

    for i in range(num_classes):
        mean = np.random.uniform(
            low=-5, high=5, size=2
        )  # Random mean in the range [-5, 5] for each dimension
        covariance = np.random.uniform(
            low=0.5, high=2, size=(2, 2)
        )  # Random covariance matrix
        samples = np.random.multivariate_normal(mean, covariance, num_samples_per_class)

        x.append(samples)
        y.extend([i] * num_samples_per_class)

    x = np.concatenate(x, axis=0)
    y = np.array(y)

    return x, y


# Modify the main function accordingly
def main():
    x, y = generate_points()
    theta, mean_X, std_X, _ = softmax_regression(x, y)
    x_normalized = (x - mean_X) / std_X
    y_pred = predict(x_normalized, theta)
    print(
        "Accuracy:",
        compute_accuracy(one_hot_encoding(y, 3), one_hot_encoding(y_pred, 3)),
    )


if __name__ == "__main__":
    main()
