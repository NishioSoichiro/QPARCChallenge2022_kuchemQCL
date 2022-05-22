import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

input_csv = pd.read_csv('./output/VQE_CoOPh4_rel_sample0.csv')
#input_csv = pd.read_csv('./output/ssVQE_CoOPh4_rel_sample0.csv')
GS_4tet_data1 = input_csv[input_csv.keys()[0]]
#GS_4tet_data2 = input_csv[input_csv.keys()[1]]
length = len(GS_4tet_data1)

x = np.arange(length) + 1.0
exact_GS = np.zeros(length) -2612.79811958

plt.figure(10)
plt.xlabel('Iteration',size=15)
plt.ylabel('Energy / a.u.',size=15)

plt.plot(x, exact_GS, linestyle='--',linewidth=3, label='exact GS')
plt.plot(x, GS_4tet_data1,linewidth=3, label='VQE $GS(0)$')
#plt.plot(x, GS_4tet_data2,linewidth=3, label='VQE $GS(1)$')
#plt.xlim(0,1000)
plt.legend()
plt.tight_layout()
plt.savefig('VQE_CoOPh4.pdf')
plt.show()

