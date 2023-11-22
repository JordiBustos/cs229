import numpy as np
import matplotlib.pyplot as plt


def generate_points(f, sample_size=100):
    """
    Generate points on a function with some noise.
    Args:
        f: the function
        sample_size: the number of points to generate
    Output:
        x: the x coordinates of the points
        y: the y coordinates of the points
        z: the z coordinates of the points
    """
    x = np.random.uniform(-1, 1, sample_size)
    y = np.random.uniform(-1, 1, sample_size)
    noise = np.random.normal(0, 0.01, sample_size)
    z = f(x, y) + noise

    return x, y, z


def generate_polynomial_features(X, degree=2):
    """
    Generate polynomial features of a given degree.
    Args:
        X: the data matrix
        degree: the degree of the polynomial
    Output:
        X_poly: the polynomial features
    """
    num_samples, num_features = X.shape
    X_poly = np.ones((num_samples, 1))  # Include bias term

    for d in range(1, degree + 1):
        for i in range(num_features):
            feature_power = X[:, i] ** d
            X_poly = np.column_stack((X_poly, feature_power))

    return X_poly


def hypothesis(theta, X):
    """
    Compute the hypothesis function.
    Args:
        theta: the parameters
        X: the data matrix
    Output:
        h: the hypothesis function
    """
    return np.dot(X, theta)


def compute_cost(theta, X, y):
    """
    Compute the cost function.
    Args:
        theta: the parameters
        X: the data matrix
        y: the target vector
    Output:
        cost: the cost function
    """
    m = len(y)
    h = hypothesis(theta, X)
    return (1 / (2 * m)) * np.sum((h - y) ** 2)


def gradient_descent(X, y, theta, alpha, num_iterations):
    """
    Perform gradient descent.
    Args:
        X: the data matrix
        y: the target vector
        theta: the parameters
        alpha: the learning rate
        num_iterations: the number of iterations
    Output:
        theta: the parameters
        cost_history: the history of cost values
    """
    m = len(y)
    cost_history = []

    for _ in range(num_iterations):
        h = hypothesis(theta, X)
        theta = theta - (alpha / m) * np.dot(X.T, (h - y))
        cost = compute_cost(theta, X, y)
        cost_history.append(cost)

    return theta, cost_history


def polynomial_regression(x, y, z, degree=2, alpha=0.01, num_iterations=1000):
    """
    Perform polynomial regression.
    Args:
        x: the x coordinates of the points
        y: the y coordinates of the points
        z: the z coordinates of the points
        degree: the degree of the polynomial
        alpha: the learning rate
        num_iterations: the number of iterations
    Output:
        theta: the parameters
        X_poly: the polynomial features
    """
    X = np.column_stack((x, y))
    X_poly = generate_polynomial_features(X, degree)

    # Initialize parameters
    theta = np.zeros(X_poly.shape[1])

    # Perform gradient descent
    theta, _ = gradient_descent(X_poly, z, theta, alpha, num_iterations)

    return theta, X_poly


def plot_paraboloid_and_fit(x, y, z, theta, degree=2):
    """
    Plot the paraboloid and the quadratic fit.
    Args:
        x: the x coordinates of the points
        y: the y coordinates of the points
        z: the z coordinates of the points
        theta: the parameters
        degree: the degree of the polynomial
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x, y, z, label="Points on the Paraboloid", color="blue")

    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = np.linspace(min(y), max(y), 100)
    x_fit, y_fit = np.meshgrid(x_fit, y_fit)
    data_fit = np.column_stack((x_fit.flatten(), y_fit.flatten()))
    X_fit_poly = generate_polynomial_features(data_fit, degree)
    z_fit = hypothesis(theta, X_fit_poly)
    z_fit = z_fit.reshape(x_fit.shape)
    ax.plot_surface(x_fit, y_fit, z_fit, color="red", alpha=0.5, label="Quadratic Fit")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Quadratic Fit to Points on a Paraboloid")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    f = lambda x, y: x**3 + y**2 + 1
    degree = 2
    x, y, z = generate_points(f)
    theta, X_poly = polynomial_regression(x, y, z, degree)
    plot_paraboloid_and_fit(x, y, z, theta, degree)
