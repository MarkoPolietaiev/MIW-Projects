def triangle_inclusion(a, b, c):
    def f(x):
        if x <= a or x > c:
            return 0
        if x <= b:
            return (x - a) / (b - a)
        return (c - x) / (c - b)

    return f


def implication(pre, post):
    return {k: T_norm(pre, v) for k, v in post.items()}


Or = S_norm = lambda x, y: x + y - x*y
And = T_norm = lambda x, y: x*y

denom = range(10, 51, 10)

A = {0: 0.507, 25: 0.434, 50: 0.182, 75: 0.330, 100: 0.461}
B = {1: 0.414, 2: 0.178, 3: 0.534}
C = triangle_inclusion(20, 50, 100)

I = [0.885, 0.875, 0.785, 0.821, 0.272]
I = {a: b for a, b in zip(denom, I)}

J = [0.942, 0.629, 0.303, 0.392, 0.828]
J = {a: b for a, b in zip(denom, J)}

K = [0.780, 0.206, 0.078, 0.995, 0.097]
K = {a: b for a, b in zip(denom, K)}

x = 100
y = 3
z = 93.000

x_A = A[x]
x_n_A = 1 - x_A

y_B = B[y]
y_n_B = 1 - y_B

z_C = C(z)
z_n_C = 1 - z_C


if __name__ == '__main__':
    pre = Or(x_A, And(y_n_B, z_C))
    I_Hat = implication(pre, I)
    pre = And(x_n_A, Or(y_B, z_C))
    J_Hat = implication(pre, J)
    pre = Or(x_A, Or(y_B, z_n_C))
    K_Hat = implication(pre, K)
    values = zip(*(s.values() for s in (I_Hat, J_Hat, K_Hat)))
    all_or = [Or(Or(v1, v2), v3) for v1, v2, v3 in values]
    numerator = sum((a * b for a, b in zip(denom, all_or)))
    denominator = sum(all_or)
    result = numerator / denominator
    print(result)
