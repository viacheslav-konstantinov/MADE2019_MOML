import sys
from io import StringIO
import unittest
import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_equal, assert_array_almost_equal, assert_allclose

import oracles
from oracles import QuadraticOracle, create_log_reg_oracle
from methods import LBFGS, Newton, BFGS, LBFGS, LineSearchTool

from nose.tools import assert_almost_equal, ok_, eq_


# Check if it's Python 3
if not sys.version_info > (3, 0):
    print('You should use only Python 3!')
    sys.exit()


def check_equal_histories(test_history, reference_history, atol=1e-3):
    if test_history is None or reference_history is None:
        assert_equal(test_history, reference_history)
        return

    for key in reference_history.keys():
        assert_equal(key in test_history, True)
        if key != 'time':
            assert_allclose(test_history[key], reference_history[key], atol=atol)
        else:
            # Cannot check time properly :(
            # At least, make sure its length is correct and its values are non-negative and monotonic
            assert_equal(len(test_history[key]), len(reference_history[key]))
            test_time = np.asarray(test_history['time'])
            assert_equal(np.all(test_time >= 0), True)
            assert_equal(np.all(test_time[1:] - test_time[:-1] >= 0), True)


@unittest.skip
def test_line_search():
    oracle = get_quadratic()
    x = np.array([100, 0, 0])
    d = np.array([-1, 0, 0])

    # Constant line search
    ls_tool = LineSearchTool(method='Constant', c=1.0)
    assert_almost_equal(ls_tool.line_search(oracle, x, d, ), 1.0)
    ls_tool = LineSearchTool(method='Constant', c=10.0)
    assert_almost_equal(ls_tool.line_search(oracle, x, d), 10.0)

    # Armijo rule
    ls_tool = LineSearchTool(method='Armijo', alpha_0=100, c1=0.9)
    assert_almost_equal(ls_tool.line_search(oracle, x, d), 12.5)

    ls_tool = LineSearchTool(method='Armijo', alpha_0=100, c1=0.9)
    assert_almost_equal(ls_tool.line_search(oracle, x, d, previous_alpha=1.0), 1.0)

    ls_tool = LineSearchTool(method='Armijo', alpha_0=100, c1=0.95)
    assert_almost_equal(ls_tool.line_search(oracle, x, d), 6.25)
    ls_tool = LineSearchTool(method='Armijo', alpha_0=10, c1=0.9)
    assert_almost_equal(ls_tool.line_search(oracle, x, d), 10.0)

    # Wolfe rule
    ls_tool = LineSearchTool(method='Wolfe', c1=1e-4, c2=0.9)
    assert_almost_equal(ls_tool.line_search(oracle, x, d), 16.0)
    ls_tool = LineSearchTool(method='Wolfe', c1=1e-4, c2=0.8)
    assert_almost_equal(ls_tool.line_search(oracle, x, d), 32.0)


class TestNewton(unittest.TestCase):
    def get_quadratic(self):
        # Quadratic function:
        #   f(x) = 1/2 x^T x - [1, 2, 3]^T x
        A = np.eye(3)
        b = np.array([1, 2, 3])
        return oracles.QuadraticOracle(A, b)


    def test_newton_ideal_step(self):
        oracle = self.get_quadratic()
        x0 = np.ones(3) * 10.0
        opt = Newton(oracle, x0, tolerance=1e-5)
        opt.run(10)
        ok_(np.allclose(opt.hist['x_star'], [1.0, 2.0, 3.0]))
        check_equal_histories(opt.hist, {'func': [90.0, -7.0],
                                        'grad_norm': [13.928388277184119, 0.0],
                                        'time': [0, 0]  # dummy timestamps
                                        })


    def get_1d(self, alpha):
        # 1D function:
        #   f(x) = exp(alpha * x) + alpha * x^2
        class Func(oracles.BaseSmoothOracle):
            def __init__(self, alpha):
                self.alpha = alpha

            def func(self, x):
                return np.exp(self.alpha * x) + self.alpha * x ** 2

            def grad(self, x):
                return np.array(self.alpha * np.exp(self.alpha * x) +
                                2 * self.alpha * x)

            def hess(self, x):
                return np.array([self.alpha ** 2 * np.exp(self.alpha * x) +
                                 2 * self.alpha])

        return Func(alpha)


    def test_newton_1d(self):
        oracle = self.get_1d(0.5)
        x0 = np.array([1.0])
        FUNC = [
            np.array([2.14872127]),
            np.array([0.9068072]),
            np.array([0.89869455]),
            np.array([0.89869434])]
        GRAD_NORM = [
            1.8243606353500641,
            0.14023069594489929,
            0.00070465169721295462,
            1.7464279966628027e-08]
        TIME = [0] * 4  # Dummy values.
        X = [
            np.array([1.]),
            np.array([-0.29187513]),
            np.array([-0.40719141]),
            np.array([-0.40777669])]
        TRUE_HISTORY = {'func': FUNC,
                        'grad_norm': GRAD_NORM,
                        'time': TIME,
                        'x': X}
        # Constant step size.
        opt = Newton(
            oracle, x0,
            tolerance=1e-10,
            line_search_options={
                'method': 'Constant',
                'c': 1.0}
        )
        opt.run(10)
        ok_(np.allclose(opt.hist['x_star'], [-0.4077777], atol=1e-4))
        check_equal_histories(opt.hist, TRUE_HISTORY)



class TestBFGS(unittest.TestCase):
    # Define a simple quadratic function for testing
    A = np.array([[1, 0], [0, 2]], dtype=np.float64)
    b = np.array([1, 6], dtype=np.float64)
    oracle = QuadraticOracle(A, b)

    f_star = -9.5
    x0 = np.array([0, 0], dtype=np.float64)
    # For this func |nabla f(x)| < tol ensures |f(x) - f(x^*)| < tol^2

    def test_default(self):
        """Check if everything works correctly with default parameters."""
        opt = BFGS(self.oracle, self.x0)
        opt.run()

    def test_tolerance(self):
        """Check if argument `tolerance` is supported."""
        BFGS(self.oracle, self.x0, tolerance=1e-5)

    def test_max_iter(self):
        """Check if argument `max_iter` is supported."""
        opt = BFGS(self.oracle, self.x0)
        opt.run(max_iter=0)

    def test_line_search_options(self):
        """Check if argument `line_search_options` is supported."""
        BFGS(self.oracle, self.x0, line_search_options={'method': 'Wolfe', 'c1': 1e-4, 'c2': 0.9})

    def test_quality(self):
        opt = BFGS(self.oracle, self.x0, tolerance=1e-5)
        opt.run(5)
        x_min = opt.hist['x_star']
        # x_min, message, _ = LBFGS(self.oracle, self.x0, tolerance=1e-5)
        f_min = self.oracle.func(x_min)

        g_k_norm_sqr = norm(self.A.dot(x_min) - self.b, 2)**2
        g_0_norm_sqr = norm(self.A.dot(self.x0) - self.b, 2)**2
        self.assertLessEqual(g_k_norm_sqr, 1e-5 * g_0_norm_sqr)
        self.assertLessEqual(abs(f_min - self.f_star), 1e-5 * g_0_norm_sqr)

    def test_history(self):
        x0 = -np.array([1.3, 2.7])
        opt = BFGS(self.oracle, x0, line_search_options={'method': 'Constant', 'c': 1.0}, tolerance=1e-6)
        opt.run(10)
        x_min = opt.hist['x_star']
        func_steps = [25.635000000000005,
                      22.99,
                      -9.48707349065929,
                      -9.5]
        grad_norm_steps = [11.629703349613008,
                           11.4,
                           0.22738961577617722,
                           0.0]
        time_steps = [0.0] * 4  # Dummy values
        x_steps = [np.array([-1.3, -2.7]),
                   np.array([1.0, 8.7]),
                   np.array([1.0, 2.88630519]),
                   np.array([1., 3.])]
        true_history = dict(grad_norm=grad_norm_steps, time=time_steps, x=x_steps, func=func_steps)
        check_equal_histories(opt.hist, true_history)


class TestLBFGS(unittest.TestCase):
    # Define a simple quadratic function for testing
    A = np.array([[1, 0], [0, 2]])
    b = np.array([1, 6])
    oracle = QuadraticOracle(A, b)

    f_star = -9.5
    x0 = np.array([0, 0])
    # For this func |nabla f(x)| < tol ensures |f(x) - f(x^*)| < tol^2

    def test_default(self):
        """Check if everything works correctly with default parameters."""
        opt = LBFGS(self.oracle, self.x0)
        opt.run()

    def test_tolerance(self):
        """Check if argument `tolerance` is supported."""
        LBFGS(self.oracle, self.x0, tolerance=1e-5)

    def test_max_iter(self):
        """Check if argument `max_iter` is supported."""
        opt = LBFGS(self.oracle, self.x0)
        opt.run(max_iter=0)

    def test_memory_size(self):
        """Check if argument `memory_size` is supported."""
        LBFGS(self.oracle, self.x0, memory_size=1)

    def test_line_search_options(self):
        """Check if argument `line_search_options` is supported."""
        LBFGS(self.oracle, self.x0, line_search_options={'method': 'Wolfe', 'c1': 1e-4, 'c2': 0.9})

    def test_quality(self):
        opt = LBFGS(self.oracle, self.x0, tolerance=1e-5)
        opt.run(5)
        x_min = opt.hist['x_star']
        # x_min, message, _ = LBFGS(self.oracle, self.x0, tolerance=1e-5)
        f_min = self.oracle.func(x_min)

        g_k_norm_sqr = norm(self.A.dot(x_min) - self.b, 2)**2
        g_0_norm_sqr = norm(self.A.dot(self.x0) - self.b, 2)**2
        self.assertLessEqual(g_k_norm_sqr, 1e-5 * g_0_norm_sqr)
        self.assertLessEqual(abs(f_min - self.f_star), 1e-5 * g_0_norm_sqr)

    def test_history(self):
        x0 = -np.array([1.3, 2.7])
        opt = LBFGS(self.oracle, x0, memory_size=10, line_search_options={'method': 'Constant', 'c': 1.0}, 
                    tolerance=1e-6)
        opt.run(10)
        x_min = opt.hist['x_star']
        func_steps = [25.635000000000005,
                      22.99,
                      -9.3476294733722725,
                      -9.4641732176886055,
                      -9.5]
        grad_norm_steps = [11.629703349613008,
                           11.4,
                           0.55751193505619512,
                           0.26830541958992876,
                           0.0]
        time_steps = [0.0] * 5  # Dummy values
        x_steps = [np.array([-1.3, -2.7]),
                   np.array([1.0, 8.7]),
                   np.array([0.45349973, 3.05512941]),
                   np.array([0.73294321, 3.01292737]),
                   np.array([0.99999642, 2.99998814])]
        true_history = dict(grad_norm=grad_norm_steps, time=time_steps, x=x_steps, func=func_steps)
        check_equal_histories(opt.hist, true_history)


if __name__ == '__main__':
    unittest.main()
