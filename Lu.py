import numpy as np


def lu_decomposition(A):
    n = A.shape[0]

    # Initialize L and U matrices
    L = np.eye(n)
    U = A.astype(float)  # Ensure A is of type float64

    for i in range(n):
        # Perform Gaussian elimination to transform U to row-echelon form
        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j, i:] -= factor * U[i, i:]

    return L, U


def solve_system(L, U, B):
    n = len(B)

    # Solve LY = B for Y
    Y = np.zeros(n)
    for i in range(n):
        Y[i] = B[i] - np.dot(L[i, :i], Y[:i])

    # Solve UX = Y for X
    X = np.zeros(n)
    for i in range(n - 1, -1, -1):
        X[i] = (Y[i] - np.dot(U[i, i + 1 :], X[i + 1 :])) / U[i, i]

    return X


# Example usage
A = np.array([[1, 1, 1], [3, 1, -3], [1, -2, -5]])

B = np.array([1, 5, 10])

# Perform LU decomposition
L, U = lu_decomposition(A)

print("Matrix A:")
print(A)

print("\nMatrix L:")
print(L)

print("\nMatrix U:")
print(U)

# Solve the system of equations
X = solve_system(L, U, B)

# Print the solution
print("\nSolution:")
for i, x in enumerate(X):
    print(f"x{i + 1} = {x}")
