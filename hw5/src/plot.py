import numpy as np
import matplotlib.pyplot as plt

Xs = list()
Ys = list()
with open("../result/hist_cpu.dat") as fin:
    fin.readline()
    for line in fin:
        x, y = map(float, line.split())
        Xs.append(x)
        Ys.append(y)
plt.figure()
plt.title("Histogram")
plt.bar(Xs, Ys)
plt.savefig("../result/histogram.png")

Xs = np.linspace(0, 30, 100)
Ys = np.exp(-Xs)
plt.figure()
plt.title("Theoretical Probability Distribution")
plt.plot(Xs, Ys)
plt.savefig("../result/probability_distribution.png")