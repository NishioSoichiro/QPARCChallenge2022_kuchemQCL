author : HINO Kentaro
# Execute steps
## VQE calculation
```
$ python VQE_${mol}_rel.py
```
where `${mol}`=`H2`, `O2`, or `CoOPh4`. 
- `H2` result is `VQE_H2_rel.pyout`.
- Note that `CoOPh4`(10 qubit) calculation requires long time.(~12 h)
- Convergence behavior will be saved as `.csv` in  `output` directory.

## ssVQE calculation
```
$ python ssVQE_${mol}_rel.py
```
where `${mol}`=`H2`, `O2`, or `CoOPh4`. 
- `H2` result is `ssVQE_H2_rel.pyout`.
- Note that `CoOPh4`(10 qubit) calculation requires long time.
- Convergence behavior will be saved as `.csv` in  `output` directory. (~12 h)
- Cost function weigts are set to $[n,...,2,1]$ where $n$ is the number of states.
- ssVQE ansatz may be not suitable for degenerate system. (In our experience)

## Plot convergence
```
$ python plot_csv_${mol}.py
```
where `${mol}`=`H2`, `O2`, or `CoOPh4`. 
- `H2` result is `H2.pdf`.

---

# Script explanations
## Dirac Hartree-Fock by BAGEL
- Output files are in `./output`.
- Energies are in `.baout` file.
- Complex value core energy $E_{\rm core}$, 1-electron integrals $(p|q)$, and 2-electron intagrals $(pq|rs)$ are in `FCIDUMP` file.  

## Fill symmetry
- `fill_fcidump.read_FCIDUMP` read the `FCIDUMP` file and fill the **relativistic** symmetry.

## Construct Quantum Circuit by Hamiltonian
1. `util_openfermion.construct_ham_pauli()` transform `FCIDUMP` to second quantization form. 
    - Note that qulacs does **not** support complex value `FCIDUMP`, thus we separate real part and imginary part.
1. `openfermion.transforms.jordan_wigner()` transform second quantization Hamiltonian to sum of Pauli operator products(JW Hamiltonian).
1. `create_observable_from_openfermion_text()` transform JW Hamiltonian to qulacs Hamiltonian.
1. Define cost function. This method is implemented with reference to Quantum Native Dojo https://dojo.qulacs.org/ja/latest/notebooks/6.2_qulacs_VQE.html.
---

# For debug
## Construct CI Hamiltonian
- `util_openfermion.construct_ham_pauli` gives CI Hamiltonian matirix $\langle\Psi_i|\hat{H}|\Psi_j\rangle$

## Diagonalize CI Hamiltonian and get energy
```
$ python diagonalize_hamiltonian.py
```
gives exact energies (eigenvalues) by diagnalizing Hamiltonian which is constructed by filled symmetry FCIDUMP.