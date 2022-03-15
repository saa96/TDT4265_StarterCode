import matplotlib.pyplot as plt
import numpy as np

Precision1 = [1.0, 1.0, 1.0, 0.5, 0.20]
Recall1 = [0.05, 0.1, 0.4, 0.7, 1.0]

Precision2 = [1.0, 0.80, 0.60, 0.5, 0.20]
Recall2 = [0.3, 0.4, 0.5, 0.7, 1.0]

plt.scatter(Recall1,Precision1)
plt.title("1")
plt.show()

plt.scatter(Recall2,Precision2)
plt.title("2")
plt.show()