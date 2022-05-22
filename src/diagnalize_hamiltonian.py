from fill_fcidump import read_FCIDUMP
import numpy as np
from scipy.linalg import eigh
import scipy
import util_openfermion
import openfermion

hoo = read_FCIDUMP("./BAGEL_data/FCIDUMP.H2_rel")
onee = hoo['H1']
twoe = hoo['H2']
print('ECORE:',hoo['ECORE'])
n_qubit = onee.shape[0]
ham_pauli = util_openfermion.construct_ham_pauli(onee, twoe, spatial2spinorb=False)
ham_sparse = openfermion.get_sparse_operator(ham_pauli, n_qubits=n_qubit)#.real
#print(ham_sparse)
eigvals, eigvecs = scipy.linalg.eigh(ham_sparse.todense())
#eigvals, eigvecs = scipy.sparse.linalg.eigsh(ham_sparse, k=3, which='SA')
print('RelCASCI energy by this program -> ',np.sort(eigvals[:3])+hoo['ECORE'])
print('RelCASCI energy by BAGEL     -> \n'+
     '1   0  *      -1.10844849\n'+
     '1   1  *      -0.83268531\n'+
     '1   2  *      -0.83268531')
