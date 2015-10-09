===========================================
Homotopy algorithm for l1-norm minimization
===========================================

This is a Python package that implement the homotopy algorithm described in 
'Sparsity-Based Space-Time Adaptive Processing Using Complex-Valued Homotopy 
Technique For Airborne Radar' by Zhaocheng Yang et al. 2013. It can be applied to 
both real-valued and complex-valued l1-norm minimization problem.

Installation
============

To install this package, you need a Python version of 2.7.5 and `numpy` installed.

This package is installable by the usual methods, either the standard ::

    $ python setup.py install [--user]

or to develop the package ::

    $ python setup.py develop [--user]

It should also be installable directly with `pip` using the command::

	$ pip install [-e] git+ssh://git@github.com/zuoshifan/homotopy
