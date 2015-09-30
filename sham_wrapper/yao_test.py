import os
from urllib import urlopen, urlretrieve
import numpy as np
from AbundanceMatching import AbundanceFunction, LF_SCATTER_MULT

test_file = urlretrieve('http://slac.stanford.edu/~yymao/tmp/am_deconv_test.npz')[0]
tests = np.load(test_file)

M, phi_log = np.loadtxt(urlopen('http://arxiv.org/src/1304.7778v2/anc/LF_SerExp.dat'), usecols=(0,1)).T
phi = (10.**phi_log)*0.4/(0.7**3)

af = AbundanceFunction(M, phi, ext_range=(-30.0, -5.0))

for scatter in (0.05, 0.1, 0.15, 0.2, 0.25):
    scatter *= LF_SCATTER_MULT
    af.deconvolute(scatter)


    assert np.allclose(af._x_deconv[scatter], tests['{0:g}'.format(scatter)], rtol=0.01)

    # assert (af._x_deconv[scatter] == tests['{0:g}'.format(scatter)]).all()
    # diff = (af._x_deconv[scatter] == tests['{0:g}'.format(scatter)])
    # for val in diff:
    #   if val != 0:
    #       print(val)
    # ineq = np.where(diff != 0)[0]
    # print(len(diff[ineq]), len(diff))

os.remove(test_file)