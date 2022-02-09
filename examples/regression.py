import random

import matplotlib.pyplot as plt
import numpy as np

import taichi as ti

ti.init(arch=ti.cpu)

number_coeffs = 4
learning_rate = 1e-4

N = 32
x = ti.field(ti.f32, shape=N, needs_grad=True)
y = ti.field(ti.f32, shape=N, needs_grad=True)
coeffs = ti.field(ti.f32, shape=number_coeffs, needs_grad=True)
loss = ti.field(ti.f32, shape=(), needs_grad=True)

xs = []
ys = []

for i in range(N):
    v = random.random() * 5 - 2.5
    xs.append(v)
    x[i] = v
    y[i] = (v - 1) * (v - 2) * (v + 2) + random.random() - 0.5
    ys.append(y[i])


@ti.kernel
def regress():
    for i in x:
        v = x[i]
        est = 0.0
        for j in ti.static(range(number_coeffs)):
            est += coeffs[j] * (v**j)
        loss[None] += 0.5 * (y[i] - est)**2

@ti.kernel
def update():
    for i in ti.static(range(number_coeffs)):
        coeffs[i] -= learning_rate * coeffs.grad[i]

def polynomial(x, coeffs):
    y = x * 0
    for i in range(number_coeffs):
        y += coeffs[i] * np.power(x, i)
    return y

def plot(ax, i, coeffs):
    x_linsp = np.linspace(-3.0,3.0,200)
    r, c = (i // 500) % 2, (i // 500) // 2
    ax[r][c].scatter(xs, ys, label='Data', color='r')
    ax[r][c].plot(x_linsp, polynomial(x_linsp, coeffs), label='Prediction', color='b')
    ax[r][c].set_title(f'Step {i}')
    ax[r][c].legend()
    ax[r][c].grid(True)
    ax[r][c].spines['left'].set_position('zero')
    ax[r][c].spines['right'].set_color('none')
    ax[r][c].spines['bottom'].set_position('zero')
    ax[r][c].spines['top'].set_color('none')
    ax[r][c].set_xlim(-3.5,3.5)
    ax[r][c].set_ylim(-4.0,8.0)


if __name__ == '__main__':
    use_tape = True
    fig, ax = plt.subplots(2,3, figsize=(15,10))

    for i in range(2501):
        if use_tape:
            with ti.Tape(loss=loss):
                regress()
        else:
            ti.clear_all_gradients()
            loss[None] = 0
            loss.grad[None] = 1
            regress()
            regress.grad()
        update()
        if i % 500 == 0:
            print('Loss =', loss[None])
            plot(ax, i, coeffs.to_numpy())

    fig.savefig('regression_result.png', dpi=200)
