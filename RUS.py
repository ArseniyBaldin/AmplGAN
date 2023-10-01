from catalyst import qjit, while_loop, cond, measure
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import torch

device = qml.device("default.qubit", wires=3, shots=1)


def layer(x):
    qml.CRY(x, wires=[0, 1])
    qml.CRY(np.pi / 2, wires=[1, 2])
    qml.RZ(-np.pi / 2, wires=1)
    qml.CRY(-x, wires=[0, 1])

@qml.qnode(device)
def make_state():
    qml.RX(np.random.rand() * np.pi, wires=0)
    qml.RY(np.random.rand() * np.pi, wires=0)
    qml.RZ(np.random.rand() * np.pi, wires=0)
    return qml.state()
@qml.qnode(device)
def state_circ(x, state):
    qml.QubitStateVector(state, wires=[0, 1, 2])
    layer(x)
    qml.PauliZ(wires=1)
    qml.RY(-np.pi / 2, wires=2)
    qml.RY(-np.pi, wires=1)
    return qml.state()


@qml.qnode(device)
def measure_circ(x, state):
    qml.QubitStateVector(state, wires=[0, 1, 2])
    layer(x)
    return qml.sample(qml.PauliZ(1))


@qml.qnode(device)
def measure(state):
    qml.QubitStateVector(state, wires=[0, 1, 2])
    return qml.sample(qml.PauliZ(2))


x = 2


def aboba(x):
    k = 0
    for _ in range(1000):
        state = make_state()
        m = measure_circ(x, state=state)
        while m == 1:
            m = measure_circ(x, state=state)
            m0 = measure(state)
            k += (m0 + 1)/2
            if k >= 100:
                break
    print("aboba")

    return k


sigmaZ = torch.tensor([[1, 0], [0, -1]])
projectorZ = torch.kron(torch.kron(torch.eye(2), sigmaZ), torch.eye(2)).to(torch.complex128)

# print(aboba(np.pi / 2))

X = np.linspace(0.2, 1 * np.pi - 0.2, 100)
# print(X)
# Y = [2 * np.arccos(aboba(i)) for i in X]
Y = [aboba(i) for i in X]
# print(Y)
plt.scatter(X, Y)
plt.show()


def projection(vec, projector):
    vec_adj = torch.adjoint(vec)
    res = torch.einsum('ia,ij,aj', vec_adj, projector, vec)
    return res
