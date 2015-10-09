import numpy as np


def sgn(z):
    "Sign function. Return z/|z| for z != 0 and 0 else."
    if z == 0:
        return 0
    else:
        return z / np.abs(z)

def asgn(x):
    "Sign of an 1d numpy array."
    y = np.zeros_like(x)
    for i in range(len(x)):
        y[i] = sgn(x[i])
    return y


class Homotopy:
    """A class implements the homotopy algorithm to do fast l1-norm minimization.

    This class implements the algorithm described in 'Sparsity-Based Space-Time
    Adaptive Processing Using Complex-Valued Homotopy Technique For Airborne Radio'
    by Zhaocheng Yang et al. 2013. It can applied to both real-valued and
    complex-valued l1-norm minimization problem.
    """

    def __init__(self, A, y):
        """Initialize a Homotopy object.

        Parameters
        ----------
        A : np.ndarray [M, N]
            Coefficients matrix.
        y : np.ndarray [M]
            Right-hand vector.
        """
        assert A.ndim ==2 and y.ndim == 1 and A.shape[0] == y.shape[0], 'Invalid arguments'
        self.A = A
        self.nrows = A.shape[0]
        self.ncols = A.shape[1]
        self.y = y
        self.dtype = np.promote_types(A.dtype, y.dtype)

    def solve(self, max_iters, epsilon=1.0e-12, precision=1.0e-10, max_non_zero=None, return_lambda=False, bisection=False, warnings=True, verbose=0):
        """Solve the l1-norm minimization problem.

        This will follow the solution path until the given conditions are satisfied
        and the results will be returned, or error occurs and it will print error
        information and exit.

        Parameters
        ----------
        max_iters : integer
            Maximum number of iterations.
        epsilon : float, optional
            Desired l2 error of y - A x. Default: 1.0e-12.
        precision : float, optional
            Numerical precision in computations such as treat abs(number) < presion
            as zero. Default: 1.0e-10.
        max_non_zero : None or integer, optional
            The maximum number of non zero elements in the final solution. Default: None,
            means no limit for number of non zero elements.
        return_lambda : boolean, optional
            Also return the final lambda value is True. Default: False.
        bisection : boolean, optional
            Do a iteratively middle-point bisection to find optimal gamma value and x.
            Default: False.
        warnings : boolean, optional
            Print warning information when the solution may not be optimal. Default: True.
        verbose : integer, optional
            Verbosity level, when 0 no output info at all, higher value for higher verbosity.
            Default: 0.

        Returns
        -------
        x : np.ndarray
            The solution.
        lambda : float, optional
            The final lambda value. only provided if `return_lambda` is True.

        """

        is_bad = False
        is_error = False
        if max_non_zero is None:
            max_non_zero = self.ncols
        else:
            assert max_non_zero >= 0, 'Invalid number of maximum non-zeros'
            max_non_zero = min(max_non_zero, self.ncols)

        # initialize x, lambda and active set
        self.c = np.dot(A.T.conj(), y)
        self.lbd = np.max(np.abs(self.c)) # lambda
        on_ind = np.argmax(np.abs(self.c))
        self.on_indices = [ on_ind ]
        self.off_indices = range(self.ncols)
        self.off_indices.remove(on_ind)
        x = np.zeros(self.ncols, dtype=self.dtype)
        cur_error = self.l2Error(x)
        if cur_error <= epsilon:
            if verbose > 0:
                print 'Return initial value as the solution with error %f (target is %f)' % (cur_error, epsilon)
            if return_lambda:
                return x, self.lbd
            else:
                return x

        # iteration
        iters = 0
        while ((cur_error > epsilon) and (len(self.on_indices) <= max_non_zero) and (iters < max_iters)):
            if verbose > 0:
                print 'Iteration %d start: num_on = %d, lambda = %f, l2Error = %f' % (iters, len(self.on_indices), self.lbd, cur_error)

            try:
                d = self.computeD()
            except np.linalg.LinAlgError:
                is_error = True
                break

            result = self.computeG(x, d, precision)
            if result is None:
                break
            else:
                gamma, ind, add = result
            x += gamma * d
            if verbose > 1:
                print 'd = ', d
                print 'gamma = ', gamma
                print 'x = ', x
            self.lbd -= gamma
            cur_error = self.l2Error(x)
            if cur_error > epsilon:
                if add:
                    self.on_indices.append(ind)
                    self.off_indices.remove(ind)
                else:
                    self.on_indices.remove(ind)
                    self.off_indices.append(ind)
                self.computeC(x)

            if verbose > 0:
                print 'Iteration %d end: num_on = %d, lambda = %f, l2Error=%f' % (iters, len(self.on_indices), self.lbd, cur_error)
            iters += 1

        num_on = len(self.on_indices)
        if num_on >= max_non_zero:
            is_bad = True
            if warnings:
                print 'Maximum non-zero value reached, solution is not optimal.'
                print '  %d elements in the on set,' % num_on
                print '  %d iterations performed,' % iters
                print '  current lambda is %f,' % self.lbd
                print '  current error is %f (target is %f)\n.' % (cur_error, epsilon)

        if iters == max_iters:
            is_bad = True
            if warnings:
                print 'Maximum number of iterations reached, solution is not optimal.'
                print '  %d elements in the on set,' % num_on
                print '  %d iterations performed,' % iters
                print '  current lambda is %f,' % self.lbd
                print '  current error is %f (target is %f)\n.' % (cur_error, epsilon)

        if (not is_bad) and bisection:
            # Mid-point bisection to find optimal gamma value
            low = 0.0
            high = gamma
            while high - low > precision:
                mid = 0.5 * (low + high)
                tmp = x - mid * d
                if self.l2Error(tmp) > epsilon:
                    high = mid
                else:
                    low = mid
            x = tmp
            self.lbd += mid

        if is_error:
            print 'Get ERROR while the solving process...'
            x = np.zeros(self.ncols, dtype=self.dtype)
            self.lbd = 0

        if return_lambda:
            return x, self.lbd
        else:
            return x


    def computeC(self, x):
        """Compute c = A^H y - A^H A x given a `x` vector."""
        # self.c = np.dot(self.A.T.conj(), self.y - np.dot(self.A, x))
        # more effective way
        self.c = np.dot(self.A.T.conj(), self.y - np.dot(self.A[:, self.on_indices], x[self.on_indices]))

    def computeD(self):
        """Compute the update direction vector."""
        c_I = self.c[self.on_indices]
        sgnc_I = asgn(c_I)
        A_I = self.A[:, self.on_indices]
        d_I = np.dot(np.linalg.inv(np.dot(A_I.T.conj(), A_I)), sgnc_I)
        d = np.zeros(self.ncols, dtype=self.dtype)
        d[self.on_indices] = d_I
        return d

    def computeG(self, x, d, precision):
        """Compute the gamma value and the corresponding index given `x` and direction
        vector `d`.
        """
        G_minus, ind_minus = self.computeG_minus(x, d, precision)
        G_plus, ind_plus = self.computeG_plus(x, d, precision)
        # no G_minus and G_plus can be get, so self.lbd should decrease to 0^+
        if (G_minus is None and G_plus is None):
            return None
        if G_minus is None:
            return G_plus, ind_plus, True
        if G_plus is None:
            return G_minus, ind_minus, False
        if G_minus < G_plus:
            return G_minus, ind_minus, False
        else:
            return G_plus, ind_plus, True

    def computeG_minus(self, x, d, precision):
        """Compute gamma_minus and its index."""
        num_on = len(self.on_indices)
        G_minus, ind_minus = None, None
        for ind, i in enumerate(self.on_indices):
            tmp = -x[i] / d[i]
            assert tmp.imag < precision, 'Not so small imaginary part of G_minus: %f' % tmp
            tmp = -(x[i] / d[i]).real
            # not exactly larger than 0 due to numerical error
            if tmp > precision:
                if ind == 0:
                    G_minus, ind_minus = tmp, i
                else:
                    if tmp < G_minus:
                        G_minus, ind_minus = tmp, i

        return G_minus, ind_minus

    def computeG_plus(self, x, d, precision):
        """Compute gamma_plus and its index."""
        A_I = self.A[:, self.on_indices]
        d_I = d[self.on_indices]
        num_off = len(self.off_indices)
        G_plus, ind_plus = None, None
        l = self.lbd
        c = self.c
        for ind, i in enumerate(self.off_indices):
            b = np.dot(self.A[:, i].T.conj(), np.dot(A_I, d_I))
            rzb = (np.conj(c[i]) * b).real
            c2 = np.abs(c[i])**2
            if np.abs(np.abs(b) - 1.0) < precision:
                # for |b| = 1
                tmp = 0.5 * (l**2 - c2) / (l - rzb)
            else:
                # for |b| != 1
                b2 = np.abs(b)**2
                delta2 = (l - rzb)**2 - (1.0 - b2) * (l**2 - c2)
                # small negative value due to numerical error
                if delta2 < 0:
                    delta = 0
                else:
                    delta = np.sqrt(delta2)
                tmp = (l - rzb - delta) / (1.0 - b2)
            # not exactly larger than 0 due to numerical error
            if tmp > precision:
                if ind == 0:
                    G_plus, ind_plus = tmp, i
                else:
                    if tmp < G_plus:
                        G_plus, ind_plus = tmp, i

        return G_plus, ind_plus


    def l2Error(self, x):
        """Compute the l2Error = |y - A x|^2 given a `x` vector."""
        # delta = self.y - np.dot(self.A, x)
        # more effective way
        delta = self.y - np.dot(self.A[:, self.on_indices], x[self.on_indices])
        return np.sum(np.abs(delta)**2)


if __name__ == "__main__":

    A = np.array([[1.0,2.0,3.0],[1.0,3.0,1.5]], dtype=np.float64)
    # A = np.array([[1.0,2.0,3.0],[1.0,3.0,1.5]], dtype=np.complex128)
    y = np.array([6,6])
    h = Homotopy(A, y)
    # case 0
    print 'case 0...'
    x = h.solve(1)
    print x
    # case 1
    print 'case 1...'
    x = h.solve(6)
    print x
    # case 2
    print 'case 2...'
    x, lbd = h.solve(6, epsilon=1.0e-8, return_lambda=True)
    print x, lbd
    # case 3
    print 'case 3...'
    x, lbd = h.solve(6, epsilon=1.0e-8, max_non_zero=1, return_lambda=True, warnings=True)
    print x, lbd
    # case 4
    print 'case 4...'
    x, lbd = h.solve(6, epsilon=1.0e-8, bisection=True, return_lambda=True)
    print x, lbd
    # case 5
    print 'case 5...'
    x, lbd = h.solve(6, epsilon=1.0e-8, verbose=1, return_lambda=True)
    print x, lbd
    # case 6
    print 'case 6...'
    x, lbd = h.solve(6, epsilon=1.0e-8, verbose=2, return_lambda=True)
    print x, lbd
