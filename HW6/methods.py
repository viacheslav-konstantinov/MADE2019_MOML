import numpy as np
import numpy.linalg as npla
import scipy
from collections import defaultdict, deque
from datetime import datetime
from numpy.linalg import LinAlgError
from scipy.optimize.linesearch import scalar_search_wolfe2
from scipy.linalg import cho_solve, cho_factor
from time import perf_counter


class LineSearchTool(object):
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        if self._method == 'Constant':
            return self.c
        elif self._method == 'Armijo':
            alpha_0 = previous_alpha if previous_alpha is not None else self.alpha_0
            return self.armijo_search(oracle, x_k, d_k, alpha_0)
        elif self._method == 'Wolfe':
            alpha = scalar_search_wolfe2(
                phi=lambda step: oracle.func_directional(x_k, d_k, step),
                derphi=lambda step: oracle.grad_directional(x_k, d_k, step),
                c1=self.c1,
                c2=self.c2
            )[0]
            if alpha is None:
                return self.armijo_search(oracle, x_k, d_k, self.alpha_0)
            else:
                return alpha

        return None

    def armijo_search(self, oracle, x_k, d_k, alpha_0):
        phi = lambda step: oracle.func_directional(x_k, d_k, step)
        alpha = alpha_0
        coef = self.c1 * oracle.grad_directional(x_k, d_k, 0)
        while phi(alpha) > phi(0) + alpha * coef:
            alpha = alpha / 2
        return alpha


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


class Newton(object):
    def __init__(self, oracle, x_0, tolerance=1e-4, line_search_options=None):
        self.oracle = oracle
        self.tolerance = tolerance
        self.line_search_tool = get_line_search_tool(line_search_options)
        self.hist = defaultdict(list)
        self.x_0 = x_0.copy()
        self.dim = x_0.size
        # maybe more of your code here

    def run(self, max_iter=100):
        # your code here
        x_k = self.x_0
        f_0_grad_norm = self.oracle.grad(self.x_0).T @ self.oracle.grad(self.x_0)
        t_start = perf_counter()

        for it in np.arange(max_iter):

            f_k_grad = self.oracle.grad(x_k)
            f_k_grad_norm = f_k_grad.T @ f_k_grad
                        
            self.hist['time'].append(perf_counter() - t_start)
            self.hist['func'].append(self.oracle.func(x_k))
            self.hist['grad_norm'].append(np.sqrt(f_k_grad_norm))

            if(self.dim <= 2):
                self.hist['x'].append(x_k)

            if(f_k_grad_norm / f_0_grad_norm <= self.tolerance):
                break

            d_k = - cho_solve(cho_factor(self.oracle.hess(x_k)), self.oracle.grad(x_k))
            alpha_k = self.line_search_tool.line_search(self.oracle, x_k, d_k)
            x_k = x_k + alpha_k * d_k


        self.hist['x_star'] = x_k


class BFGS(object):
    def __init__(self, oracle, x_0, tolerance=1e-4, line_search_options=None):
        self.oracle = oracle
        self.tolerance = tolerance
        self.line_search_tool = get_line_search_tool(line_search_options)
        self.hist = defaultdict(list)
        self.x_0 = x_0.copy()
        self.dim = x_0.size
        # maybe more of your code here

    def run(self, max_iter=100):
        # your code here
        x_k = self.x_0
        f_0_grad_norm = self.oracle.grad(self.x_0).T @ self.oracle.grad(self.x_0)

        t_start = perf_counter()

        I = np.eye(self.dim)
        H_k = I

        for _ in np.arange(max_iter):
            f_k_grad = self.oracle.grad(x_k)
            f_k_grad_norm = f_k_grad.T @ f_k_grad

            self.hist['time'].append(perf_counter() - t_start)
            self.hist['func'].append(self.oracle.func(x_k))
            self.hist['grad_norm'].append(np.sqrt(f_k_grad_norm))

            if(self.dim <= 2):
                self.hist['x'].append(x_k)

            if(f_k_grad_norm / f_0_grad_norm <= self.tolerance):
                break
            
            d_k = - H_k @ f_k_grad
            alpha_k = self.line_search_tool.line_search(self.oracle, x_k, d_k)
            x_k += alpha_k * d_k

            s_k = alpha_k * d_k
            y_k = self.oracle.grad(x_k) - f_k_grad
            
            s_k = s_k.reshape(-1, 1)
            y_k = y_k.reshape(-1, 1)
            
            rho_k = 1 / (y_k.T @ s_k)
            U1 = (I - rho_k * s_k @ y_k.T)
            U2 = (I - rho_k * y_k @ s_k.T)
            U3 = s_k @ s_k.T
            
            H_k = U1 @ H_k @ U2 + rho_k * U3
        
        self.hist['x_star'] = x_k

class LBFGS(object):
    def __init__(self, oracle, x_0, tolerance=1e-4, memory_size=10, 
                 line_search_options=None):
        
        self.oracle = oracle
        self.tolerance = tolerance
        self.line_search_tool = get_line_search_tool(line_search_options)
        self.hist = defaultdict(list)
        self.memory_size = memory_size
        self.x_0 = x_0.copy()
        self.dim = x_0.size
        # maybe more of your code here


    def run(self, max_iter=100):
        # your code here
        x_k = self.x_0
        f_0_grad_norm = self.oracle.grad(self.x_0).T @ self.oracle.grad(self.x_0)
        
        cur_history = deque([], maxlen=self.memory_size)
        
        t_start = perf_counter()
        gamma_k = 1.
        
        for k in np.arange(max_iter):
            f_k_grad = self.oracle.grad(x_k)
            f_k_grad_norm = f_k_grad.T @ f_k_grad

            self.hist['time'].append(perf_counter() - t_start)
            self.hist['func'].append(self.oracle.func(x_k))
            self.hist['grad_norm'].append(np.sqrt(f_k_grad_norm))

            if(self.dim <= 2):
                self.hist['x'].append(x_k)

            if(f_k_grad_norm / f_0_grad_norm <= self.tolerance):
                break

            q = f_k_grad
            alphas = []
            
            if k and self.memory_size:
                hist = cur_history.copy()
                si, yi = hist.popleft()
                gamma_k = (si * yi).sum() / (yi * yi).sum()
                rho_i = 1 / (yi * si).sum()
                alpha_i = rho_i * (si * q).sum()
                alphas.append(alpha_i)
                q = q - alpha_i*yi
                
                for _ in range(min(k - 1, self.memory_size - 1)):
                    si, yi = hist.popleft()
                    rho_i = 1 / (yi * si).sum()
                    alpha_i = rho_i * (si * q).sum()
                    alphas.append(alpha_i)
                    q = q - alpha_i*yi
            
            z = gamma_k*q

            if k and self.memory_size:
                hist = cur_history.copy()
                for i in np.arange(min(k, self.memory_size) - 1, -1, -1):
                    si, yi = hist.pop()
                    rho_i = 1 / (yi * si).sum()
                    beta_i = rho_i * (yi * z).sum()
                    z = z + (alphas[i] - beta_i) * si
            d_k = - z
            
            alpha_k = self.line_search_tool.line_search(self.oracle, x_k, d_k)
            
            x_k = x_k + alpha_k * d_k
            s_k = alpha_k * d_k
            y_k = self.oracle.grad(x_k) - f_k_grad
            cur_history.appendleft((s_k, y_k))
            gamma_k = (s_k * y_k).sum() / (y_k * y_k).sum()

        self.hist['x_star'] = x_k


class GradientDescent(object):
    """
    Gradient descent optimization algorithm.
    
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    """
    def __init__(self, oracle, x_0, tolerance=1e-10, line_search_options=None):
        self.oracle = oracle
        self.x_0 = x_0.copy()
        self.tolerance = tolerance
        self.line_search_tool = get_line_search_tool(line_search_options)
        self.hist = defaultdict(list)
        self.dim = x_0.size
        # maybe more of your code here
    
    def run(self, max_iter=100):
        """
        Runs gradient descent for max_iter iterations or until stopping 
        criteria is satisfied, starting from point x_0. Saves function values 
        and time in self.hist
        
        self.hist : dictionary of lists
        Dictionary containing the progress information
        Dictionary has to be organized as follows:
            - self.hist['time'] : list of floats, containing time in seconds passed from the start of the method
            - self.hist['func'] : list of function values f(x_k) on every step of the algorithm
            - self.hist['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - self.hist['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

        """
        # your code here
        x_k = self.x_0
        f_0_grad_norm = self.oracle.grad(self.x_0).T @ self.oracle.grad(self.x_0)

        self.hist['time'] = [0]
        self.hist['func'] = [self.oracle.func(self.x_0)]
        self.hist['grad_norm'] = [np.sqrt(f_0_grad_norm)]

        if(self.dim <= 2):
            self.hist['x'] = [self.x_0]

        t_start = perf_counter()

        for _ in np.arange(max_iter):
            f_k = self.oracle.func(x_k)
            f_k_grad = self.oracle.grad(x_k)
            f_k_grad_norm = f_k_grad.T @ f_k_grad
            
            if(f_k_grad_norm / f_0_grad_norm <= self.tolerance):
                break

            d_k = - f_k_grad
            alpha_k = self.line_search_tool.line_search(self.oracle, x_k, d_k)
            x_k = x_k + alpha_k * d_k

            self.hist['time'].append(perf_counter() - t_start)
            self.hist['func'].append(self.oracle.func(x_k))
            self.hist['grad_norm'].append(np.sqrt(f_k_grad_norm))

            if(self.dim <= 2):
                self.hist['x'].append(x_k)

        self.hist['x_star'] = x_k
