import numpy as np
import scipy.linalg
import openfermion
from itertools import product, combinations
import math


def alph_qubit(iorb):
    return iorb*2
def beta_qubit(iorb):
    return iorb*2+1

def _number_operator_fermi(norb, *, fac_alph=1.0, fac_beta=1.0):
    n_alph_fermi = openfermion.FermionOperator()
    n_beta_fermi = openfermion.FermionOperator()
    for i in range(norb):
        n_alph_fermi += openfermion.FermionOperator(((alph_qubit(i), 1),(alph_qubit(i), 0)), fac_alph)
        n_beta_fermi += openfermion.FermionOperator(((beta_qubit(i), 1),(beta_qubit(i), 0)), fac_beta)
    return n_alph_fermi, n_beta_fermi

def _s_plus_minus_operator_fermi(norb, *, fac=1.0):
    s_plus_minus_fermi = openfermion.FermionOperator()
    for i,j in product(range(norb),repeat=2):
        s_plus_minus_fermi += openfermion.FermionOperator(((alph_qubit(i), 1),(beta_qubit(i), 0), \
                                                           (beta_qubit(j), 1),(alph_qubit(j), 0)), fac)
    return s_plus_minus_fermi

def _spatial2spinorb(onee_spatial, twoe_spatial=None):
    if onee_spatial is not None:
        norb = onee_spatial.shape[0]
        onee_spinorb = np.zeros([norb*2]*2, dtype=onee_spatial.dtype)
        onee_spinorb[0::2, 0::2] = onee_spatial
        onee_spinorb[1::2, 1::2] = onee_spatial
    else:
        onee_spinorb = None

    if twoe_spatial is not None:
        norb = twoe_spatial.shape[0]
        twoe_spinorb = np.zeros([norb*2]*4, dtype=twoe_spatial.dtype)
        twoe_spinorb[0::2, 0::2, 0::2, 0::2] = twoe_spatial
        twoe_spinorb[1::2, 1::2, 0::2, 0::2] = twoe_spatial
        twoe_spinorb[0::2, 0::2, 1::2, 1::2] = twoe_spatial
        twoe_spinorb[1::2, 1::2, 1::2, 1::2] = twoe_spatial
    else:
        twoe_spinorb = None

    if onee_spinorb is None:
        return twoe_spinorb
    if twoe_spinorb is None:
        return onee_spinorb
    return onee_spinorb, twoe_spinorb

def _spatial2spinorb_int3e(int3e_spatial):
    '''
    the code is equivalent to:
    norb = int3e_spatial.shape[0]
    int3_spinorb = np.zeros([norb*2]*6, dtype=int3_spatial.dtype)
    int3_spinorb[0::2, 0::2, 0::2, 0::2, 0::2, 0::2] = int3_spatial #aa aa aa
    int3_spinorb[0::2, 0::2, 0::2, 0::2, 1::2, 1::2] = int3_spatial #aa aa bb
    int3_spinorb[0::2, 0::2, 1::2, 1::2, 0::2, 0::2] = int3_spatial #aa bb aa
    int3_spinorb[1::2, 1::2, 0::2, 0::2, 0::2, 0::2] = int3_spatial #bb aa aa
    int3_spinorb[0::2, 0::2, 1::2, 1::2, 1::2, 1::2] = int3_spatial #aa bb bb
    int3_spinorb[1::2, 1::2, 0::2, 0::2, 1::2, 1::2] = int3_spatial #bb aa bb
    int3_spinorb[1::2, 1::2, 1::2, 1::2, 0::2, 0::2] = int3_spatial #bb bb aa
    int3_spinorb[1::2, 1::2, 1::2, 1::2, 1::2, 1::2] = int3_spatial #bb bb bb
    '''
    norb = int3e_spatial.shape[0]
    int3e_spinorb = np.zeros([norb*2]*6, dtype=int3e_spatial.dtype)
    for i0,i1,i2 in product([0,1], repeat=3):
        int3e_spinorb[i0::2,i0::2, i1::2,i1::2, i2::2,i2::2] = int3e_spatial
    return int3e_spinorb

def _spatial2spinorb_int4e(int4e_spatial):
    norb = int4e_spatial.shape[0]
    int4e_spinorb = np.zeros([norb*2]*8, dtype=int4e_spatial.dtype)
    for i0,i1,i2,i3 in product([0,1], repeat=4):
        int4e_spinorb[i0::2,i0::2, i1::2,i1::2, i2::2,i2::2, i3::2,i3::2] = int4e_spatial
    return int4e_spinorb

def _spatial2spinorb_int5e(int5e_spatial):
    norb = int5e_spatial.shape[0]
    int5e_spinorb = np.zeros([norb*2]*10, dtype=int5e_spatial.dtype)
    for i0,i1,i2,i3,i4 in product([0,1], repeat=5):
        int5e_spinorb[i0::2,i0::2, i1::2,i1::2, i2::2,i2::2, i3::2,i3::2, i4::2,i4::2] = int5e_spatial
    return int5e_spinorb

def _construct_ham_1body(hint_spinorb, fac):
    orb_qubit_map = list(range(hint_spinorb.shape[0]))

    ham_1body = openfermion.FermionOperator()
    for p, p_qubit in enumerate(orb_qubit_map):
        for q, q_qubit in enumerate(orb_qubit_map):
            ham_1body += openfermion.FermionOperator(((p_qubit, 1),(q_qubit, 0)), hint_spinorb[p,q] * fac)
    return ham_1body

def _construct_ham_2body(gint_spinorb, fac):
    orb_qubit_map = list(range(gint_spinorb.shape[0]))

    ham_2body = openfermion.FermionOperator()
    for (p,p_qubit), (q,q_qubit), (r,r_qubit), (s,s_qubit) in product(enumerate(orb_qubit_map), repeat=4):
        if p!=r and q!=s:#e.g. a_q a_s = 0
            ham_2body += openfermion.FermionOperator(\
                ((p_qubit, 1),(r_qubit, 1),(s_qubit, 0),(q_qubit, 0)), gint_spinorb[p,q,r,s] * fac)
    return ham_2body

def _construct_ham_3body(int3e_spinorb, fac):
    '''
    the code is equivalent to:
    orb_qubit_map = list(range(int3e_spinorb.shape[0]))

    ham_3body = openfermion.FermionOperator()
    for (p,p_qubit), (q,q_qubit), (r,r_qubit), (s,s_qubit), (t,t_qubit), (u,u_qubit) in product(enumerate(orb_qubit_map), repeat=6):
        if len(set([p,r,t]))==3 and len(set([u,s,q]))==3:
            ham_3body += openfermion.FermionOperator(\
                ((p_qubit, 1),(r_qubit, 1),(t_qubit, 1),(u_qubit, 0),(s_qubit, 0),(q_qubit, 0)), int3e_spinorb[p,q, r,s, t,u] * fac)
    return ham_3body
    '''
    orb_qubit_map = list(range(int3e_spinorb.shape[0]))

    ham_3body = openfermion.FermionOperator()
    for index_pairs in product(enumerate(orb_qubit_map), repeat=6):
        orbs  = [x[0] for x in index_pairs]
        qubit = [x[1] for x in index_pairs]
        if len(set([orbs[0],orbs[2],orbs[4]]))==3 and len(set([orbs[1],orbs[3],orbs[5]]))==3:
            moint = int3e_spinorb[orbs[0],orbs[1],orbs[2], orbs[3],orbs[4],orbs[5]]
            if abs(moint*fac) > 1.e-8:
                ham_3body += openfermion.FermionOperator(((qubit[0],1),(qubit[2],1),(qubit[4],1), \
                                                          (qubit[5],0),(qubit[3],0),(qubit[1],0)), moint*fac)
    return ham_3body

def _construct_ham_4body(int4e_spinorb, fac):
    orb_qubit_map = list(range(int4e_spinorb.shape[0]))

    ham_4body = openfermion.FermionOperator()
    for index_pairs in product(enumerate(orb_qubit_map), repeat=8):
        orbs  = [x[0] for x in index_pairs]
        qubit = [x[1] for x in index_pairs]
        if len(set([orbs[0],orbs[2],orbs[4],orbs[6]]))==4 and len(set([orbs[1],orbs[3],orbs[5],orbs[7]]))==4:
            ham_4body += openfermion.FermionOperator(\
                ((qubit[0],1),(qubit[2],1),(qubit[4],1),(qubit[6],1), \
                 (qubit[7],0),(qubit[5],0),(qubit[3],0),(qubit[1],0),), int4e_spinorb[orbs[0],orbs[1],orbs[2],orbs[3],\
                                                                                      orbs[4],orbs[5],orbs[6],orbs[7]] * fac)
    return ham_4body

def _construct_ham_5body(int5e_spinorb, fac):
    orb_qubit_map = list(range(int5e_spinorb.shape[0]))

    ham_5body = openfermion.FermionOperator()
    for index_pairs in product(enumerate(orb_qubit_map), repeat=10):
        orbs  = [x[0] for x in index_pairs]
        qubit = [x[1] for x in index_pairs]
        if len(set([orbs[0],orbs[2],orbs[4],orbs[6],orbs[8]]))==5 and len(set([orbs[1],orbs[3],orbs[5],orbs[7],orbs[9]]))==5:
            ham_5body += openfermion.FermionOperator(\
                ((qubit[0],1),(qubit[2],1),(qubit[4],1),(qubit[6],1),(qubit[8],1), \
                 (qubit[9],0),(qubit[7],0),(qubit[5],0),(qubit[3],0),(qubit[1],0),), \
                                                     int5e_spinorb[orbs[0],orbs[1],orbs[2],orbs[3],orbs[4],\
                                                                   orbs[5],orbs[6],orbs[7],orbs[8],orbs[9]] * fac)
    return ham_5body


def construct_ham_pauli(onee, twoe, int3e=None, int4e=None, int5e=None, *, return_ham_fermi=False, spatial2spinorb=True):
    if spatial2spinorb:
        onee_spinorb, twoe_spinorb = _spatial2spinorb(onee, twoe)
    else:
        onee_spinorb, twoe_spinorb = onee, twoe
    ham_fermi  = _construct_ham_1body(onee_spinorb, 1.0)
    ham_fermi += _construct_ham_2body(twoe_spinorb, 1/2)

    if int3e is not None:
        int3e_spinorb = _spatial2spinorb_int3e(int3e)
        ham_fermi += _construct_ham_3body(int3e_spinorb, 1/(2*3))

    if int4e is not None:
        int4e_spinorb = _spatial2spinorb_int4e(int4e)
        ham_fermi += _construct_ham_4body(int4e_spinorb, 1/(2*3*4))

    if int5e is not None:
        int5e_spinorb = _spatial2spinorb_int5e(int5e)
        ham_fermi += _construct_ham_5body(int5e_spinorb, 1/(2*3*4*5))

    if not return_ham_fermi:
        ham_pauli = openfermion.jordan_wigner(ham_fermi)
        return ham_pauli
    else:
        return ham_fermi
