import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

input_csv = pd.read_csv('./output/VQE_H2_rel_sample0.csv')
S0_data = input_csv[input_csv.keys()[0]]
length = len(S0_data)

x = np.arange(length) + 1.0
exact_S0 = np.zeros(length)  -1.10844849

plt.xlabel('Iteration',size=15)
plt.ylabel('Energy / a.u.',size=15)

plt.plot(x, exact_S0, linestyle='--',linewidth=3, label='exact $S_0$')
plt.plot(x, S0_data,linewidth=3, label='VQE $S_0$')
plt.legend()
plt.savefig('H2.pdf')
plt.show()

