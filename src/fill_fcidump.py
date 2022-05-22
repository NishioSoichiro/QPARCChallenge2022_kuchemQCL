import re
from functools import reduce
import numpy as np
import pprint as pp

def bar(n):
    if n%2 == 0:
        return n - 1
    else:
        return n + 1

def family_1(i,j,k,l,val):
    ''' Eq. (8-13)'''
    bi, bj, bk, bl = bar(i), bar(j), bar(k), bar(l)
    cval = np.conjugate(val)
    return [(  i,  j,  k,  l,  val), 
            ( bi, bj,  l,  k, cval), 
            (  j,  i, bk, bl, cval), 
            ( bj, bi, bl, bk, val)]

def family_2(i,j,k,bl,val):
    ''' Eq. (8-14) '''
    bi, bj, bk, l = bar(i), bar(j), bar(k), bar(bl)
    cval = np.conjugate(val)
    return [(  i,  j,  k, bl,  val), 
            (  k, bl,  i,  j,  val), 
            (  j,  i, bk,  l,-cval), 
            ( bk,  l,  j,  i,-cval),
            (  l, bk, bj, bi,-cval), 
            ( bj, bi,  l, bk,-cval),
            ( bl,  k, bi, bj, cval), 
            ( bi, bj, bl,  k, cval)]

def family_3(i,bj,k,bl,val):
    ''' Eq. (8-15) '''
    bi, j, bk, l = bar(i), bar(bj), bar(k), bar(bl)
    cval = np.conjugate(val)
    return [(  i, bj,  k, bl,  val), 
            ( bj,  i, bl,  k, cval)]

def family_4(i,bj,bk,l,val):
    ''' Eq. (8-15) '''
    bi, j, k, bl = bar(i), bar(bj), bar(bk), bar(l)
    cval = np.conjugate(val)
    return [(  i, bj,  bk,  l,  val), 
            ( bi,  j,   k, bl, cval)]

# family_1
def mmmm(i,j,k,l,val):
    return family_1(i,j,k,l,val)
def ppmm(bi,bj,l,k,cval):
    i,j = bar(bi), bar(bj)
    val = np.conjugate(cval)
    return family_1(i,j,k,l,val)
def mmpp(j,i,bk,bl,cval):
    k,l = bar(k), bar(bl)
    val = np.conjugate(cval)
    return family_1(i,j,k,l,val)
def pppp(bj,bi,bl,bk,val):
    i,j,k,l = bar(bi), bar(bj), bar(bk), bar(bl)
    return family_1(i,j,k,l,val)

# family_2
def mmmp(i,j,k,bl,val):
    return family_2(i,j,k,bl,val)
def mpmm(k,bl,i,j,val):
    return family_2(i,j,k,bl,val)
def mmpm(j,i,bk,l,mcval):
    k,bl = bar(bk), bar(l)
    val = -np.conjugate(mcval)
    return family_2(i,j,k,bl,val)
def pmmm(bk,l,j,i,mcval):
    k,bl = bar(bk), bar(l)
    val = -np.conjugate(mcval)
    return family_2(i,j,k,bl,val)
def mppp(l,bk,bj,bi,mcval):
    i,j,k,bl = bar(bi), bar(bj), bar(bk), bar(l)
    val = -np.conjugate(mcval)
    return family_2(i,j,k,bl,val)
def ppmp(bj,bi,l,bk,mcval):
    i,j,k,bl = bar(bi), bar(bj), bar(bk), bar(l)
    val = -np.conjugate(mcval)
    return family_2(i,j,k,bl,val)
def pmpp(bl,k,bi,bj,cval):
    i,j = bar(bi), bar(bj)
    val = np.conjugate(cval)
    return family_2(i,j,k,bl,val)
def pppm(bi,bj,bl,k,cval):
    i,j = bar(bi), bar(bj)
    val = np.conjugate(cval)
    return family_2(i,j,k,bl,val)


# family_3
def mpmp(i,bj,k,bl,val):
    return family_3(i,bj,k,bl,val)
def pmpm(bj,i,bl,k,cval):
    val = np.conjugate(cval)
    return family_3(i,bj,k,bl,val)


# family_4
def mppm(i,bj,bk,l,val):
    return family_4(i,bj,bk,l,val)
def pmmp(bi,j,k,bl,cval):
    i, bj, bk, l = bar(bi), bar(j), bar(k), bar(bl)
    val = np.conjugate(cval)
    return family_4(i,bj,bk,l,val)

def swap_orb(i,j,k,l,val):
    """ (ij|kl) = (ji|lk)*  """
    return [(i,j,k,l,val), (j,i,l,k, np.conjugate(val))]

def swap_ele(i,j,k,l,val):
    """ (ij|kl) = (kl|ij)  """
    return [(i,j,k,l,val),(k,l,i,j,val)]

def swap_ele_orb(i,j,k,l,val):
    lis = []
    for tup in swap_ele(i,j,k,l,val):
        for tup2 in swap_orb(*tup):
            lis.append(tup2)
    return lis


def read_FCIDUMP(filename, KramersSym=False):
    '''Parse FCIDUMP.  Return a dictionary to hold the integrals and
    parameters with keys:  H1, H2, ECORE, NORB, NELEC, MS, ORBSYM, ISYM
    '''

    print('Parsing %s **** modified by nishio for FCIDUMP from zfci module of BAGEL ****' % filename)

    finp = open(filename, 'r')

    data = []
    for i in range(10):
        line = finp.readline().upper()
        data.append(line)
        if '&END' in line:
            break
    else:
        raise RuntimeError('Problematic FCIDUMP header')

    result = {}
    tokens = ','.join(data).replace('&FCI', '').replace('&END', '')
    tokens = tokens.replace(' ', '').replace('\n', '').replace(',,', ',')
    for token in re.split(',(?=[a-zA-Z])', tokens):
        key, val = token.split('=')
        if key in ('NORB', 'NELEC', 'MS2', 'ISYM'):
            result[key] = int(val.replace(',', ''))
        elif key in ('ORBSYM',):
            result[key] = [int(x) for x in val.replace(',', ' ').split()]
        else:
            result[key] = val

    norb = result['NORB']
    norb_pair = norb * (norb+1) // 2
    h1e = np.zeros((norb,norb), dtype=np.complex128)
    h2e = np.zeros((norb,norb,norb,norb), dtype=np.complex128)
    dat = finp.readline().split()
    while dat:
        tp = tuple(map(float, dat[0].strip('()').split(',')))
        zint = complex(tp[0],tp[1])

        def fill(i,j,k,l,val):
            if abs(h2e[i-1,j-1,k-1,l-1] - val) > 1.e-16 \
            and abs(h2e[i-1,j-1,k-1,l-1]) > 1.e-16:
                #print(f'({i}{j}|{k}{l})={h2e[i-1,j-1,k-1,l-1]}, but {val}')
                pass
            if abs(val.imag) < 1.e-14:
                val = val.real + 0.0j
            h2e[i-1,j-1,k-1,l-1] = val

        i, j, k, l = [int(x) for x in dat[1:5]]
        if k != 0:
            if KramersSym:
                family = (i%2, j%2, k%2, l%2)
                if family == (0,0,0,0):
                    _iter = mmmm
                elif family == (0,0,0,1):
                    _iter = mmmp
                elif family == (0,0,1,0):
                    _iter = mmpm
                elif family == (0,0,1,1):
                    _iter = mmpp
                elif family == (0,1,0,0):
                    _iter = mpmm
                elif family == (0,1,0,1):
                    _iter = mpmp
                elif family == (0,1,1,0):
                    _iter = mppm
                elif family == (0,1,1,1):
                    _iter = mppp
                elif family == (1,0,0,0):
                    _iter = pmmm
                elif family == (1,0,0,1):
                    _iter = pmmp
                elif family == (1,0,1,0):
                    _iter = pmpm
                elif family == (1,0,1,1):
                    _iter = pmpp
                elif family == (1,1,0,0):
                    _iter = ppmm
                elif family == (1,1,0,1):
                    _iter = ppmp
                elif family == (1,1,1,0):
                    _iter = pppm
                elif family == (1,1,1,1):
                    _iter = pppp
                else:
                    assert False, f'no family found {family}'
            else:
                _iter = swap_ele_orb

            for p,q,r,s,val in _iter(i,j,k,l,zint):#[(i,j,k,l,zint)]:#
                if p == r or q == s:
                    ''' different electron in the same orbnital '''
                    #print(f'skip, ({p}{q}|{r}{s})')
                    continue

                fill(p,q,r,s,val)
        else:
            if j != 0:
                def fill(i,j,val):
                    if abs(h1e[i-1,j-1] - val) > 1.e-16 \
                        and abs(h1e[i-1,j-1]) > 1.e-16:
                        print(f'({i}|{j})={h1e[i-1,j-1]}, but {val}')
                    h1e[i-1,j-1] = zint
                fill(i,j,zint)
            else:
                result['ECORE'] = zint
        dat = finp.readline().split()

    result['H1'] = h1e
    result['H2'] = h2e
    finp.close()
    return result
