import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

input_csv = pd.read_csv('./outputs/VQE_O2_rel_sample0.csv')
GS_4tet_data1 = input_csv[input_csv.keys()[0]]
#GS_4tet_data2 = input_csv[input_csv.keys()[1]]
length = len(GS_4tet_data1)

x = np.arange(length) + 1.0
exact_GS = np.zeros(length) -149.52905903

plt.figure(10)
plt.xlabel('Iteration',size=15)
plt.ylabel('Energy / a.u.',size=15)

plt.plot(x, exact_GS, linestyle='--',linewidth=3, label='exact GS')
plt.plot(x, GS_4tet_data1,linewidth=3, label='VQE $GS$')
#plt.plot(x, GS_4tet_data2,linewidth=3, label='VQE $GS(4tet)2$')
plt.legend()
plt.savefig('O2.pdf')
plt.show()

