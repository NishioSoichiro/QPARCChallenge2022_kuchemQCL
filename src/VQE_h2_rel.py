import qulacs
from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import CZ, RY, RZ, merge
from qulacs import Observable
from qulacs.observable import create_observable_from_openfermion_text

from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermion.linalg import get_sparse_operator
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf

from scipy.optimize import minimize
from pyscf import fci
import numpy as np
import util_openfermion
from fill_fcidump import read_FCIDUMP

hoo = read_FCIDUMP("./BAGEL_data/FCIDUMP.H2_rel")


onee_real = hoo['H1'].real
twoe_real = hoo['H2'].real
ecore = hoo['ECORE']

fermionic_hamiltonian_real = util_openfermion.construct_ham_pauli(onee_real, twoe_real, 
        return_ham_fermi=True, spatial2spinorb=False)
jw_hamiltonian_real = jordan_wigner(fermionic_hamiltonian_real)

onee_imag = np.array(hoo['H1'].imag,dtype='float64')
twoe_imag = np.array(hoo['H2'].imag,dtype='float64')

fermionic_hamiltonian_imag = util_openfermion.construct_ham_pauli(onee_imag, twoe_imag, 
        return_ham_fermi=True, spatial2spinorb=False)
jw_hamiltonian_imag = jordan_wigner(fermionic_hamiltonian_imag)


qulacs_hamiltonian_real = create_observable_from_openfermion_text(str(jw_hamiltonian_real))
qulacs_hamiltonian_imag = create_observable_from_openfermion_text(
        str(jw_hamiltonian_imag).replace('j','')) # divided by imaginary unit j

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

def cost(theta_list):
    state = QuantumState(n_qubit)
    circuit = he_ansatz_circuit(n_qubit, depth, theta_list)
    circuit.update_quantum_state(state)
    return qulacs_hamiltonian_real.get_expectation_value(state)\
            - qulacs_hamiltonian_imag.get_expectation_value(state)

for i in range(3):
    cost_history = []
    init_theta_list = np.random.random(2*n_qubit*(depth+1))*2*np.pi
    cost_history.append(cost(init_theta_list))
    method = "BFGS"
    options = {"disp": True, "maxiter": 3000, "gtol": 1e-6}
    opt = minimize(cost, init_theta_list,
                method=method,
                options= options,
                callback=lambda x: cost_history.append(cost(x)))

    data = np.array(cost_history).real + ecore.real
    print('BAGEL\n'      
        +'1   0  *      -1.10844849     1.57e-14      0.00\n'
        +'1   1  *      -0.83268531     3.65e-13      0.00\n'
        +'1   2  *      -0.83268531     4.41e-13      0.00\n')
    np.savetxt(f'output/VQE_H2_rel_sample{i}.csv', data, delimiter=",")
    print(f'VQE {data[-1]}')
