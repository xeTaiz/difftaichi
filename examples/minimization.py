import random
import matplotlib.pyplot as plt

import taichi as ti

ti.init(arch=ti.cpu)

n = 10
x = ti.field(dtype=ti.f32, shape=n, needs_grad=True)
y = ti.field(dtype=ti.f32, shape=n)
L = ti.field(dtype=ti.f32, shape=(), needs_grad=True)


@ti.kernel
def compute_loss():
    for i in range(n):
        L[None] += 0.5 * (x[i] - y[i])**2


@ti.kernel
def gradient_descent():
    for i in x:
        x[i] -= x.grad[i] * 0.1

def plot(ax, k):
    c, r = (k // 11) // 5, (k // 11) % 5
    ax[r][c].plot(x.to_numpy(), color='blue', label='Prediction')
    ax[r][c].plot(y.to_numpy(), color='black', label='Target')
    ax[r][c].set_title(f'Step {k}')

def main():
    fig, ax = plt.subplots(5, 2, figsize=(10,25))
    # Initialize vectors
    for i in range(n):
        x[i] = random.random()
        y[i] = random.random()

    # Optimize with 100 gradient descent iterations
    for k in range(100):
        with ti.Tape(loss=L):
            compute_loss()
        gradient_descent()
        if k % 11 == 0:
            plot(ax, k)
            print(f'Iteration {k:03d}: Loss = {L[None]}')

    fig.savefig('minimization_result.png', dpi=200)


if __name__ == '__main__':
    main()
