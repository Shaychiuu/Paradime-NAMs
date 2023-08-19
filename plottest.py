import random
import matplotlib.pyplot as plt

xs = [random.randint(0, 10) for _ in range(100)]
ys = [random.randint(0, 10) for _ in range(100)]

plt.scatter(xs, ys)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Random plot")
plt.show()