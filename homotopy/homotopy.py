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
    # def __init__(self, A, y, max_non_zero=None, epsilon=0.0):
    def __init__(self, A, y):
        self.A = A
        self.nrows = A.shape[0]
        self.ncols = A.shape[1]
        self.y = y
        self.dtype = np.promote_types(A.dtype, y.dtype)

        self.c = np.dot(A.T.conj(), y)
        self.lbd = np.max(np.abs(self.c)) # lambda
        on_ind = np.argmax(np.abs(self.c))
        self.on_indices = [ on_ind ]
        self.off_indices = range(self.ncols)
        self.off_indices.remove(on_ind)

    def solve(self, max_iters, epsilon=0.0, precision=1.0e-8, max_non_zero=None, return_lambda=False, verbose=False):
        is_bad = False
        if max_non_zero is None:
            max_non_zero = self.ncols
        else:
            assert max_non_zero >= 0, 'Invalid number of maximum non-zeros'
            max_non_zero = np.min(max_non_zero, self.ncols)

        x = np.zeros(self.ncols, dtype=self.dtype)
        cur_error = self.l2Error(x)
        if cur_error <= epsilon:
            if return_lambda:
                return x, self.lbd
            else:
                return x

        # iteration
        iters = 0
        while ((cur_error > epsilon) and (len(self.on_indices) <= max_non_zero) and (iters < max_iters)):
            if verbose:
                print 'Iteration %d start: num_on = %d, lambda = %f, l2Error = %f' % (iters, len(self.on_indices), self.lbd, cur_error)

            d = self.computeD()
            print 'd = ', d
            result = self.computeG(x, d, precision)
            print result
            if result is None:
                break
            else:
                gamma, ind, add = result
            x += gamma * d
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

            if verbose:
                print 'Iteration %d end: num_on = %d, lambda = %f, l2Error=%f' % (iters, len(self.on_indices), self.lbd, cur_error)
            iters += 1

        num_on = len(self.on_indices)
        if num_on >= max_non_zero:
            is_bad = True
            print 'Maximum non-zero value reached, solution is not optimal.\n'
            print '  %d elements in the on set\n' % num_on
            print '  %d iterations performed\n' % iters
            print '  current lambda is %f\n' % self.lbd
            print '  current error is %f (target is %f)\n\n' % (cur_error, epsilon)

        if iters == max_iters:
            is_bad = True
            print 'Maximum number of iterations reached, solution is not optimal.\n'
            print '  %d elements in the on set\n' % num_on
            print '  %d iterations performed\n' % iters
            print '  current lambda is %f\n' % self.lbd
            print '  current error is %f (target is %f)\n\n' % (cur_error, epsilon)

        # if not is_bad:
        #     # Mid-point bisection to find optimal gamma value
        #     low = 0.0
        #     high = self.lambda

        if return_lambda:
            return x, self.lbd
        else:
            return x


    def computeC(self, x):
        # self.c = np.dot(self.A.T.conj(), self.y - np.dot(self.A, x))
        # more effective way
        self.c = np.dot(self.A.T.conj(), self.y - np.dot(self.A[:, self.on_indices], x[self.on_indices]))

    def computeD(self):
        c_I = self.c[self.on_indices]
        sgnc_I = asgn(c_I)
        A_I = self.A[:, self.on_indices]
        d_I = np.dot(np.linalg.inv(np.dot(A_I.T.conj(), A_I)), sgnc_I)
        d = np.zeros(self.ncols, dtype=self.dtype)
        d[self.on_indices] = d_I
        return d

    def computeG(self, x, d, precision):
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
        num_on = len(self.on_indices)
        G_minus, ind_minus = None, None
        for ind, i in enumerate(self.on_indices):
            tmp = -(x[i] / d[i]).real
            # not exactly larger than 0 due to numerical error
            if tmp > precision:
                if ind == 0:
                    G_minus, ind_minus = tmp, i
                else:
                    if tmp < G_minus:
                        G_minus, ind_minus = tmp, i

        return G_minus, i

    def computeG_plus(self, x, d, precision):
        A_I = self.A[:, self.on_indices]
        d_I = d[self.on_indices]
        num_off = len(self.off_indices)
        G_plus, ind_plus = None, None
        l = self.lbd
        c = self.c
        for ind, i in enumerate(self.off_indices):
            b = np.dot(self.A[:, i].T.conj(), np.dot(A_I, d_I))
            tmp = (l - (np.conj(c[i]) * b).real)**2 - (1.0 - np.abs(b)**2) * (l**2 - np.abs(c[i])**2)
            # small negative value due to numerical error
            if tmp < 0:
                delta = 0
            else:
                delta = np.sqrt(tmp)
            if np.abs(b) != 1.0:
                tmp = (l - (np.conj(c[i]) * b).real - delta) / (1.0 - np.abs(b)**2)
            else:
                tmp = 0.5 * (l**2 - np.abs(c[i])**2) / (l - (np.conj(c[i]) * b).real)
            # not exactly larger than 0 due to numerical error
            if tmp > precision:
                if ind == 0:
                    G_plus, ind_plus = tmp, i
                else:
                    if tmp < G_plus:
                        G_plus, ind_plus = tmp, i

        return G_plus, ind_plus


    def l2Error(self, x):
        delta = self.y - np.dot(self.A, x)
        return np.sum(delta**2)


if __name__ == "__main__":

    A = np.array([[1.0,2.0,3.0],[1.0,3.0,1.5]], dtype=np.float64)
    y = np.array([6,6])
    h = Homotopy(A, y)
    x, lbd = h.solve(6, epsilon=1.0e-8, verbose=True, return_lambda=True)
    print lbd
    print x
