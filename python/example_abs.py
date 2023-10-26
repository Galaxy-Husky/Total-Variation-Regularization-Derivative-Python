from diff_tvr import DiffTVR
import numpy as np
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    np.random.seed(0)
    start = time.time()

    # Data
    def func(x):
        return abs(x - 0.5)

    def deriv_func(x):
        return np.where(x < 0.5, -1, 1)

    n = 1000
    x_coords = np.linspace(0, 1, n+1)
    data = func(x_coords)

    # True derivative
    deriv_true = deriv_func(x_coords)

    # Add noise
    data_noisy = data + np.random.normal(0, 0.05, n+1)

    # Plot true and noisy signal
    fig1 = plt.figure()
    plt.plot(x_coords, data)
    plt.plot(x_coords, data_noisy)
    plt.title("Signal")
    plt.legend(["True", "Noisy"])

    # Derivative with TVR
    dx = np.diff(x_coords)
    diff_tvr = DiffTVR(n, dx, maxiter=None)
    deriv, progress = diff_tvr.get_deriv_tvr(
        data=data_noisy,
        deriv_guess=np.ones(n + 1),
        alpha=0.2,
        no_opt_steps=100,
    )

    # Plot TVR derivative
    fig2 = plt.figure()
    plt.plot(x_coords, deriv_true)
    plt.plot(x_coords, deriv)
    plt.title("Derivative")
    plt.legend(["True", "TVR"])

    fig1.savefig('signal.png')
    fig2.savefig('derivative.png')
    print(f'time: {(time.time() - start):.2f}')
    plt.show()
