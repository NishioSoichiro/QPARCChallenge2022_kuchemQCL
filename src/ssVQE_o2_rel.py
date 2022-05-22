import qulacs
from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import CZ, RY, RZ, merge
from qulacs import Observable
from qulacs.observable import create_observable_from_openfermion_text

from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermion.linalg import get_sparse_operator #エラーが出る場合は openfermion を version 1.0.0 以上にしてみてください
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf

from scipy.optimize import minimize
from pyscf import fci
import numpy as np
#import matplotlib.pyplot as plt
import util_openfermion
from fill_fcidump import read_FCIDUMP

hoo = read_FCIDUMP("./BAGEL_data/FCIDUMP.O2_rel")


onee_real = hoo['H1'].real
twoe_real = hoo['H2'].real
ecore = hoo['ECORE']

fermionic_hamiltonian_real = util_openfermion.construct_ham_pauli(onee_real, twoe_real, 
    return_ham_fermi=True, spatial2spinorb=False)
jw_hamiltonian_real = jordan_wigner(fermionic_hamiltonian_real)

onee_imag = np.array(hoo['H1'].imag,dtype='float64')
twoe_imag = np.array(hoo['H2'].imag,dtype='float64')

fermionic_hamiltonian_imag = util_openfermion. construct_ham_pauli(onee_imag, twoe_imag, 
    return_ham_fermi=True, spatial2spinorb=False)
jw_hamiltonian_imag = jordan_wigner(fermionic_hamiltonian_imag)


qulacs_hamiltonian_real = create_observable_from_openfermion_text(str(jw_hamiltonian_real))
qulacs_hamiltonian_imag = create_observable_from_openfermion_text(
    str(jw_hamiltonian_imag).replace('j',''))

n_qubit =  onee_real.shape[0]
depth = n_qubit

def he_ansatz_circuit(n_qubit, depth, theta_list):
    """he_ansatz_circuit
    Returns hardware efficient ansatz circuit.

    Args:
        n_qubit (:class:`int`):
            the number of qubit used (equivalent to the number of fermionic modes)
        depth (:class:`int`):
            depth of the circuit.
        theta_list (:class:`numpy.ndarray`):
            rotation angles.
    Returns:
        :class:`qulacs.QuantumCircuit`
    """
    circuit = QuantumCircuit(n_qubit)
    circuit.add_gate(RY(0, theta_list[-2]))
    circuit.add_gate(RZ(0, theta_list[-1]))
    for d in range(depth):
        for i in range(n_qubit):
            circuit.add_gate(merge(RY(i, theta_list[2*i+2*n_qubit*d]), RZ(i, theta_list[2*i+1+2*n_qubit*d])))
        for i in range(n_qubit//2):
            circuit.add_gate(CZ(2*i, 2*i+1))
        for i in range(n_qubit//2-1):
            circuit.add_gate(CZ(2*i+1, 2*i+2))
    for i in range(n_qubit):
        circuit.add_gate(merge(RY(i, theta_list[2*i+2*n_qubit*depth]), RZ(i, theta_list[2*i+1+2*n_qubit*depth])))

    return circuit

def get_exp(state, theta_list):
    circuit = he_ansatz_circuit(n_qubit, depth, theta_list) #量子回路を構成
    circuit.update_quantum_state(state) #量子回路を状態に作用
    return qulacs_hamiltonian_real.get_expectation_value(state) - qulacs_hamiltonian_imag.get_expectation_value(state) #ハミルトニアンの期待値を計算

def cost(theta_list):
    state0 = QuantumState(n_qubit) #|00000> を準備
    state1 = QuantumState(n_qubit); state1.set_computational_basis(1) #|00001> を準備
    state2 = QuantumState(n_qubit); state2.set_computational_basis(2) #|0010> を準備
    return 3*get_exp(state0, theta_list)+2*get_exp(state1, theta_list)+1*get_exp(state2, theta_list)

for i_sample in range(3):
    #init_theta_list = np.random.random(2*n_qubit*(depth+1))*1e-1
    #init_theta_list = np.random.random(2*n_qubit*(depth+2))*2*np.pi
    exp_history0 = []
    exp_history1 = []
    exp_history2 = []
    def callback(theta_list):
        state0 = QuantumState(n_qubit) #|0000> を準備
        state1 = QuantumState(n_qubit); state1.set_computational_basis(1) #|0001> を準備
        state2 = QuantumState(n_qubit); state2.set_computational_basis(2) #|0010> を準備
        exp_history0.append(get_exp(state0, theta_list))
        exp_history1.append(get_exp(state1, theta_list))
        exp_history2.append(get_exp(state2, theta_list))
    init_theta_list = np.random.random(2*n_qubit*(depth+1)+3)*2*np.pi

    method = "BFGS"
    options = {"disp": True, "maxiter": 1000, "gtol": 1e-8}
    opt = minimize(cost, init_theta_list,
                method=method,
                options= options,
                callback=callback)

    data = np.array(np.hstack((np.array(exp_history0).reshape(-1,1),
                            np.array(exp_history1).reshape(-1,1),
                            np.array(exp_history2).reshape(-1,1)))).real + ecore.real
    print(f'sample={i_sample}')
    print('BAGEL\n'      
        +'8   0  *    -149.52905903\n'
        +'8   1  *    -149.52353159\n'
        +'8   2  *    -149.52353157\n')
    np.savetxt(f'./output/ssVQE_O2_rel_sample{i_sample}.csv', data, delimiter=",")
    print(f'ssVQE {data[-1]}')
    #print(f'delta_conv {data[-2]-data[-1]}')
