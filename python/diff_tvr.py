import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse.linalg import LinearOperator, aslinearoperator, cg
from scipy.sparse import diags, spdiags
from typing import Tuple


class DiffTVR:

    def __init__(self, n: int, dx: np.array, tol: float = 1e-4, maxiter: int = 100):
        """Differentiate with TVR.

        Args:
            n (int): Number of points in data.
            dx (np.array): Spacing of data.
        """

        self.n = n
        self.dx = dx
        self.tol = tol
        self.maxiter = maxiter

        self.d_mat = self._make_d_mat()
        self.a_mat = self._make_a_mat()
        self.a_mat_t = self._make_a_mat_t()
        self.row_sums = self.sum_rows()

    def _make_d_mat(self):
        """Make differentiation matrix with central differences.

        Returns:
            np.array: N x N+1
        """

        c = 1. / self.dx
        return diags([-c, c], [0, 1], (self.n, self.n + 1))

    def _make_a_mat(self) -> LinearOperator:
        """Make integration matrix with trapezoidal rule.

        Returns:
            LinearOperator: N x N+1
        """

        def trapezoid_integrate(vector):
            return np.cumsum(0.5 * self.dx * (vector[:-1] + vector[1:]))

        return LinearOperator(shape=(self.n, self.n + 1), matvec=trapezoid_integrate)

    def _make_a_mat_t(self) -> LinearOperator:
        """Transpose of the integration matrix with trapezoidal rule.

        Returns:
            LinearOperator: N+1 x N
        """

        def trapezoid_integrate_transpose(vector):
            result = np.append(0.5 * self.dx * np.flip(np.cumsum(np.flip(vector))), [0], axis=0)
            result[1:] = result[1:] + result[:-1]
            return result

        return LinearOperator(shape=(self.n + 1, self.n), matvec=trapezoid_integrate_transpose)

    def sum_rows(self) -> np.array:
        """Row sums of A.T @ A

        Returns:
            np.array: Vector of length N+1
        """

        product_mat_row_sums = np.zeros(self.n + 1)
        product_mat_row_sums[0] = (np.arange(self.n, 0, -1) * self.dx).sum()
        product_mat_row_sums[1:-1] = product_mat_row_sums[0] - np.cumsum(np.cumsum(self.dx[:-1]))
        product_mat_row_sums[:-1] = self.dx * product_mat_row_sums[:-1]
        product_mat_row_sums[1:] = product_mat_row_sums[1:] + product_mat_row_sums[:-1]
        return 0.5 * product_mat_row_sums

    def make_en_mat(self, deriv_curr: np.array) -> np.array:
        """Diffusion matrix

        Args:
            deriv_curr (np.array): Current derivative of length N+1

        Returns:
            np.array: N x N
        """

        eps = 1e-6
        vec = 1. / np.sqrt(pow(self.d_mat @ deriv_curr, 2) + eps)
        return spdiags(self.dx * vec, 0, self.n, self.n)

    def make_ln_mat(self, alpha: float, en_mat: np.array) -> np.array:
        """Diffusivity term

        Args:
            alpha (float): Regularization parameter
            en_mat (np.array): Result from make_en_mat

        Returns:
            np.array: N+1 x N+1
        """

        return alpha * self.d_mat.T @ en_mat @ self.d_mat

    def make_gn_vec(self, deriv_curr: np.array, data: np.array, ln_mat: np.array) -> np.array:
        """Negative right hand side of linear problem

        Args:
            deriv_curr (np.array): Current derivative of size N+1
            data (np.array): Data of size N
            ln_mat (np.array): Diffusivity term from make_ln_mat

        Returns:
            np.array: Vector of length N+1
        """

        return self.a_mat_t.matvec(self.a_mat.matvec(deriv_curr)) - self.a_mat_t.matvec(data) + ln_mat @ deriv_curr

    def make_hn_mat(self, ln_mat: np.array) -> LinearOperator:
        """Matrix in linear problem

        Args:
            ln_mat (np.array): Diffusivity term from make_ln_mat

        Returns:
            np.array: N+1 x N+1
        """

        return self.a_mat_t @ self.a_mat + aslinearoperator(ln_mat)

    def precondition(self, row_sums: np.array, ln_mat: np.array) -> LinearOperator:
        approx_hn_mat = np.diag(row_sums) + ln_mat
        bands = np.zeros((3, self.n + 1))
        bands[0, 1:] = np.diagonal(approx_hn_mat, 1)
        bands[1] = np.diagonal(approx_hn_mat, 0)
        bands[2, :-1] = np.diagonal(approx_hn_mat, -1)

        def inverse_approx_hn_mat(vector):
            return solve_banded((1, 1), bands, vector)

        return LinearOperator(shape=(self.n + 1, self.n + 1), matvec=inverse_approx_hn_mat)

    def get_deriv_tvr_update(self, data: np.array, deriv_curr: np.array, alpha: float) -> np.array:
        """Get the TVR update

        Args:
            data (np.array): Data of size N
            deriv_curr (np.array): Current deriv of size N+1
            alpha (float): Regularization parameter

        Returns:
            np.array: Update vector of size N+1
        """

        en_mat = self.make_en_mat(deriv_curr)
        ln_mat = self.make_ln_mat(alpha, en_mat)
        hn_mat = self.make_hn_mat(ln_mat)
        gn_vec = self.make_gn_vec(deriv_curr, data, ln_mat)
        p = self.precondition(self.row_sums, ln_mat)
        x, exit_code = cg(hn_mat, -gn_vec, tol=self.tol, maxiter=self.maxiter, M=p, atol='legacy')
        return x

    def get_deriv_tvr(self, data: np.array, deriv_guess: np.array, alpha: float, no_opt_steps: int,
                      return_progress: bool = False, return_interval: int = 1) -> Tuple[np.array, np.array]:
        """Get derivative via TVR over optimization steps

        Args:
            data (np.array): Data of size N+1
            deriv_guess (np.array): Guess for derivative of size N+1
            alpha (float): Regularization parameter
            no_opt_steps (int): No. opt steps to run
            return_progress (bool, optional): True to return derivative progress during optimization. Defaults to False.
            return_interval (int, optional): Interval at which to store derivative if returning. Defaults to 1.

        Returns:
            Tuple[np.array,np.array]: First is the final derivative of size N+1,
            second is the stored derivatives if return_progress=True of size no_opt_steps+1 x N+1, else [].
        """
        data = data[1:] - data[0]
        deriv_curr = deriv_guess
        deriv_st = []
        for opt_step in range(no_opt_steps):
            update = self.get_deriv_tvr_update(data, deriv_curr, alpha)
            deriv_curr += update
            if return_progress:
                if opt_step % return_interval == 0:
                    deriv_st.append(deriv_curr)
        return deriv_curr, deriv_st
