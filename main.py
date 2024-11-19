import numpy as np
import sympy as sp

def read(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

        data = []
        for line in lines:
            values = line.strip().split()
            data.append([float(v) for v in values])

        y = np.array(data).T   
        return y
    

def get_A():
    A = [[0, 1, 0, 0, 0, 0,],
         [-(c2 + c1) / m1, 0, c2 / m1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [c2 / m2, 0, -(c2 + c3) / m2, 0, c3 / m2, 0],
         [0, 0, 0, 0, 0, 1],
         [0, 0, c3 / m3, 0, -(c4 + c3) / m3, 0]]
    return sp.Matrix(A)

def print_matrix(matrix, name='Matrix'):
    print(f"\n{name}\n")
    num_rows, num_cols = matrix.shape
    max_widths = [max(len(str(matrix[i, j])) for i in range(num_rows)) for j in range(num_cols)]
    for i in range(num_rows):
        row_str = ""
        for j in range(num_cols):
            element = str(matrix[i, j])
            padding = max_widths[j] - len(element)
            row_str += element + ' ' * (padding + 1) + "\t\t"
        print(row_str)

def derivative_of_vector(y, b):
    derivatives = [sp.diff(Yi, Bi) for Yi in y for Bi in b]
    cols = len(b)
    
    derivative_matrix = [derivatives[i:i + cols] for i in range(0, len(derivatives), cols)]
    return sp.Matrix(derivative_matrix)

def get_f(A, B, U):
    return A @ U + B


def get_U(A, B, U):
    B = np.array(B.subs(beta_0)).tolist()
    k1 = h * get_f(A, B, U)
    k2 = h * get_f(A, B, U + k1 / 2)
    k3 = h * get_f(A, B, U + k2 / 2)
    k4 = h * get_f(A, B, U + k3)

    return U + (k1 + 2 * k2 + 2 * k3 + k4) / 6

def get_f_y(A, y):
    return A @ y

def get_y(A, y):
    k1 = h * get_f_y(A, y)
    k2 = h * get_f_y(A, y + k1 / 2)
    k3 = h * get_f_y(A, y + k2 / 2)
    k4 = h * get_f_y(A, y + k3)

    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

y = read('y9.txt')

h = 0.2

c1, c2, c3, c4, m1, m2, m3 = sp.symbols('c1, c2, c3, c4, m1, m2, m3')
beta_0 = {c2: 0.2, c4: 0.1, m1: 9}
params = {c1: 0.14, c3: 0.2, m2: 28, m3: 18}
beta = [c2, c4, m1]
eps = 1e-16

A_filled = get_A().subs(params)
sp.pprint(A_filled)

beta_vector = np.array([beta_0[c2], beta_0[c4], beta_0[m1]])
while True:
    A = np.array((A_filled.subs(beta_0)).tolist())
    U = np.zeros((6, 3))

    inv_integral = np.zeros((3, 3))
    mult_integral = np.zeros((1, 3))
    distance_integral = 0

    y_approx = y[0]
    for i in range(1, len(y)):
        Ay = A @ sp.Matrix(y_approx)
        B = derivative_of_vector(Ay, beta)

        inv_prod = U.T @ U
        inv_integral = (inv_integral + inv_prod).astype('float64')

        mult_prod = U.T @ (y[i] - y_approx)
        mult_integral = (mult_integral + mult_prod).astype('float64')

        distance = (y[i] - y_approx).T @ (y[i] - y_approx)
        distance_integral += distance

        U = get_U(A, B, U)
        y_approx = get_y(A, y_approx)
    inv_integral = inv_integral * 0.2
    mult_integral = mult_integral * 0.2
    distance_integral = distance_integral * 0.2
    
    inv_integral = np.linalg.pinv(inv_integral)
    mult_integral = mult_integral.flatten()
    delta_beta = inv_integral @ mult_integral
    beta_vector += delta_beta
    beta_0 = {c2: beta_vector[0], c4: beta_vector[1], m1: beta_vector[2]}

    print("Beta", beta_vector)
    print("Distance", distance_integral)
    if distance_integral < eps:
        print("Converged, approx: ", beta_0)
        break