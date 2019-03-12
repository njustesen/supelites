import sympy
import numpy as np


def dist2coords(distances):
    distances = np.array(distances)

    n = len(distances)
    X = sympy.symarray('x', (n, n - 1))

    for row in range(n):
        X[row, row:] = [0] * (n - 1 - row)

    for point2 in range(1, n):

        expressions = []

        for point1 in range(point2):
            expression = np.sum((X[point1] - X[point2]) ** 2)
            expression -= distances[point1,point2] ** 2
            expressions.append(expression)

        X[point2,:point2] = sympy.solve(expressions, list(X[point2,:point2]))[1]

    return X



D = np.ones((30, 30))
coords = dist2coords(D)
print(np.array(coords, dtype=np.float))
# Output:
# [[0.        0.       ]
#  [1.        0.       ]
#  [0.5       0.8660254]]
