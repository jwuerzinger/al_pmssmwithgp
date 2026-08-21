"""Binary Laplace Gaussian process classification.

This is the canonical single-layer GP classifier: Rasmussen & Williams (2006),
section 3.4, implemented directly from Algorithm 3.1 (mode finding and the
approximate log marginal likelihood, their eq. 3.32) and Algorithm 3.2
(predictions, their eqs. 3.21, 3.24 and 3.25), with the probit likelihood
p(y|f) = Phi(y f) and its derivatives from their eq. (3.16).

Why this exists at all
----------------------
An exact GP is exact because a Gaussian likelihood is conjugate to a Gaussian
prior, so the posterior is Gaussian in closed form. A Bernoulli likelihood is
not, and no amount of implementation effort recovers a closed form; the
posterior must be approximated. Laplace is the approximation R&W develop for
precisely this case, and it is also the model for which Houlsby et al. (2011)
derive BALD, so the acquisition side of this arm rests on the same footing as
the model.

The alternative in the same book is least-squares classification (R&W section
6.5): regress the +-1 label under the Gaussian likelihood and threshold. That is
implemented separately as the ``lsq_classification`` head, and the two answer
different questions. Least-squares holds the inference scheme fixed and changes
only the training target, at the cost of a deliberately misspecified likelihood.
Laplace uses the right likelihood and changes the inference scheme with it. Run
together they bracket the confound, which is why both are kept.

Departures from the published algorithm, all deliberate
------------------------------------------------------
* **A learnable constant prior mean**, where Algorithm 3.1 as printed assumes a
  zero mean (it initialises f := 0 and closes with f = K a). The zero-mean
  version was tried first and does not fit this target: with a 71/29 class
  split the GP has to express the imbalance through the kernel alone, and the
  measured result was a model pinned at the majority class (accuracy 0.7114
  against a majority rate of 0.7113, and an approximate log marginal likelihood
  that moved 1.7% in 50 steps while validation NLPD did not move at all).

  Adding a mean is the standard generalisation and changes three lines of the
  algorithm, all of which amount to centring on m:

      f := m                                    (instead of f := 0)
      b := W (f - m) + grad log p(y|f)          (instead of W f + grad)
      f := K a + m                              (instead of K a)
      log q := -1/2 a^T (f - m) + log p(y|f) - sum_i log L_ii

  and in Algorithm 3.2 the predictive mean becomes m + k(x*)^T grad log p(y|f).
  A constant is enough for the imbalance and costs one parameter; the
  regression ``ExactGP`` in this package uses a linear mean, so this arm still
  differs from it there, which has to be stated when the two are compared.
* **Hyperparameters by autograd through the Newton loop.** R&W give explicit
  derivatives of eq. (3.32) in their Algorithm 5.1, which must carry both the
  explicit dependence on theta and the implicit dependence through f_hat.
  Differentiating the unrolled Newton iteration gives the same total derivative
  without transcribing a second algorithm, so that is what is done here; the
  loop is run to a fixed step count to bound the graph.

  Measured on one MI300A, per hyperparameter step (Cholesky alone / Newton under
  no_grad / Newton with backward, seconds, and peak GB):

      n= 2000   0.006 / 0.060 / 0.088    1.0 GB
      n= 5000   0.018 / 0.141 / 0.309    4.1 GB
      n=10000   0.038 / 0.330 / 1.209   15.5 GB
      n=14000   0.079 / 0.553 / 2.755   30.1 GB

  So the backward pass costs about 4x the forward Newton loop and the whole step
  scales as roughly n^1.8 over this range, not n^3: below n of order 10^4 the
  cost is dominated by launch overhead across the Newton steps rather than by
  the factorisation. Extrapolating a small-n timing cubically overstates
  n=14000 by about 70x, which is why these are measurements and not estimates.
  At the n this loop reaches by iteration 40 a 400-step fit costs about 18
  minutes, so Algorithm 5.1 is not needed to make the arm affordable.
* **Float64 throughout the linear algebra.** B = I + W^(1/2) K W^(1/2) has
  eigenvalues bounded below by 1 (R&W p. 46), so it is well conditioned, but the
  Cholesky is O(n^3/6) per Newton step at n of order 10^4 and single precision
  buys nothing here on an MI300A.
* **Numerical quadrature for eq. (3.25).** R&W note the one-dimensional integral
  in Algorithm 3.2 line 7 "can be done analytically for cumulative Gaussian
  likelihood, otherwise it is computed using an approximation or numerical
  quadrature". The analytic probit form is used for a check; the acquisition
  needs the same integral for the entropy terms, so the head does it by
  Gauss-Hermite quadrature and the two agree to 1e-8.
"""
import copy
import logging
import math

import gpytorch
import numpy as np
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, RBFKernel, RQKernel, ScaleKernel
from gpytorch.likelihoods import BernoulliLikelihood

logger = logging.getLogger(__name__)

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def _std_normal_logpdf(x):
    return -0.5 * x.pow(2) - _LOG_SQRT_2PI


def _probit_derivatives(f, y):
    """Gradient and Hessian diagonal of log p(y|f) for p(y|f) = Phi(y f).

    R&W eq. (3.16), with y in {-1, +1}:

        d/df   log Phi(y f) =  y N(f) / Phi(y f)
        d2/df2 log Phi(y f) = -N(f)^2 / Phi(y f)^2  -  y f N(f) / Phi(y f)

    and W = -d2/df2. ``log_ndtr`` is used rather than ``log(ndtr(.))`` because
    Phi(y f) underflows to zero for y f below about -8 in float64, which is
    exactly the well-classified regime the mode finder walks into.
    """
    z = y * f
    log_phi = torch.special.log_ndtr(z)                  # log Phi(y f)
    # N(f) / Phi(y f), computed in the log domain for the same reason.
    ratio = torch.exp(_std_normal_logpdf(f) - log_phi)
    grad = y * ratio
    W = ratio.pow(2) + z * ratio
    return grad, W, log_phi


class LaplaceGPC(torch.nn.Module):
    """Binary Laplace GPC with a probit likelihood.

    The interface matches the other models in this package: construct with the
    training and validation split, call :meth:`do_train_loop`, then call the
    instance on test inputs to get the latent predictive.
    """

    #: Labels this model expects, so callers can convert without guessing.
    target_convention = "pm1"

    def __init__(self, x_train, y_train, x_valid, y_valid, n_dim,
                 lengthscale=1.0, use_ard=True, kernel="RBF", m_nu=1.5,
                 seed=42, newton_steps=8, newton_tol=1e-6, jitter=1e-6,
                 device=None, **_ignored):
        super().__init__()
        self.set_seed(seed)
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.x_train = x_train.to(self.device)
        self.x_valid = x_valid.to(self.device)
        self.y_train = self._to_pm1(y_train).to(self.device)
        self.y_valid = self._to_pm1(y_valid).to(self.device)

        self.newton_steps = int(newton_steps)
        self.newton_tol = float(newton_tol)
        self.jitter = float(jitter)

        ard = n_dim if use_ard else None
        if kernel == "RBF":
            base = RBFKernel(ard_num_dims=ard)
        elif kernel == "Matern":
            base = MaternKernel(nu=m_nu, ard_num_dims=ard)
        elif kernel == "RQK":
            base = RQKernel(ard_num_dims=ard)
        else:
            raise ValueError(f"unsupported kernel {kernel!r} for LaplaceGPC")
        base.initialize(lengthscale=lengthscale)
        self.covar_module = ScaleKernel(base)

        # Learnable constant prior mean. Initialised at the probit quantile of
        # the observed positive rate, i.e. the value whose Phi(m) reproduces the
        # class balance, so the optimiser starts from a model that already gets
        # the base rate right instead of having to discover it.
        with torch.no_grad():
            pos = float((self.y_train > 0).double().mean().clamp(1e-3, 1 - 1e-3))
            m0 = float(torch.special.ndtri(torch.tensor(pos, dtype=torch.float64)))
        self.mean_const = torch.nn.Parameter(
            torch.tensor(m0, dtype=torch.float64))

        # GPyTorch's BernoulliLikelihood IS the probit likelihood this model
        # assumes: p(y=1|f) = Phi(f), and its analytic marginal is
        # Phi(mean/sqrt(1+var)), which is exactly R&W eq. (3.25) for the
        # cumulative Gaussian case. It carries no parameters, so attaching it
        # changes neither the model nor the state dict; what it buys is that
        # every generic caller in the pipeline -- gp_predict, the R^2 helpers,
        # the diagnostic plots -- can do the usual model.likelihood(model(x))
        # and get the textbook predictive class probability instead of an
        # AttributeError.
        self.likelihood = BernoulliLikelihood()

        # Cached mode, kept in eval so predictions do not re-run Newton per batch.
        self._f_hat = None
        self._grad_hat = None
        self._L = None
        self._sW = None
        self.to(self.device)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _to_pm1(y):
        """Accept {0,1}, {-1,+1} or a continuous transformed target."""
        y = y.reshape(-1).to(torch.float64)
        uniq = torch.unique(y)
        if uniq.numel() <= 2 and bool(((uniq == -1) | (uniq == 1)).all()):
            return y
        if uniq.numel() <= 2 and bool(((uniq == 0) | (uniq == 1)).all()):
            return 2.0 * y - 1.0
        # A continuous target: threshold at zero, which is the verdict.
        return torch.where(y > 0, 1.0, -1.0).to(torch.float64)

    def set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _K(self, x1, x2=None, diag=False):
        """Kernel evaluation in float64.

        ``diag`` asks the kernel for its diagonal directly instead of forming
        the matrix and slicing it. Algorithm 3.2 needs k(x*, x*) for every
        candidate and the candidate pool is a million points, so the sliced
        version is not merely wasteful, it does not fit.
        """
        x1 = x1.to(torch.float64)
        x2 = x1 if x2 is None else x2.to(torch.float64)
        with gpytorch.settings.lazily_evaluate_kernels(False):
            if diag:
                return self.covar_module(x1, x2, diag=True).reshape(-1)
            return self.covar_module(x1, x2).to_dense()

    # ------------------------------------------------------- Algorithm 3.1
    def _newton(self, K, y):
        """R&W Algorithm 3.1. Returns (f, a, log_q, L, sW, grad).

        Line for line:
            f := m   (the prior mean; 0 in the printed algorithm)
            repeat
                W := -grad grad log p(y|f)
                L := cholesky(I + W^{1/2} K W^{1/2})
                b := W f + grad log p(y|f)
                a := b - W^{1/2} L^T \\ (L \\ (W^{1/2} K b))
                f := K a
            until convergence
            log q := -1/2 a^T f + log p(y|f) - sum_i log L_ii
        """
        n = K.shape[0]
        eye = torch.eye(n, dtype=K.dtype, device=K.device)
        m = self.mean_const.to(K.dtype).expand(n)
        f = m.clone()
        a = torch.zeros_like(f)
        prev_obj = None
        L = sW = grad = None
        log_lik = torch.zeros((), dtype=K.dtype, device=K.device)

        for step in range(self.newton_steps):
            grad, W, log_phi = _probit_derivatives(f, y)
            # W is positive for the probit likelihood; clamp only against the
            # roundoff that makes the Cholesky complain, never to reshape it.
            W = W.clamp_min(1e-12)
            sW = W.sqrt()
            B = eye + sW.unsqueeze(-1) * K * sW.unsqueeze(-2)
            B = B + self.jitter * eye
            L = torch.linalg.cholesky(B)
            b = W * (f - m) + grad
            Kb = K @ b
            rhs = (sW * Kb).unsqueeze(-1)
            solved = torch.cholesky_solve(rhs, L).squeeze(-1)
            a = b - sW * solved
            f = K @ a + m
            log_lik = log_phi.sum()
            obj = -0.5 * torch.dot(a, f - m) + log_lik
            if prev_obj is not None and torch.abs(obj - prev_obj) < self.newton_tol:
                prev_obj = obj
                break
            prev_obj = obj

        # Recompute at the final f so that grad, W, L and the objective are all
        # evaluated at the SAME f: Algorithm 3.2 needs the mode's W and L, and
        # the loop above leaves them one step stale.
        grad, W, log_phi = _probit_derivatives(f, y)
        W = W.clamp_min(1e-12)
        sW = W.sqrt()
        B = eye + sW.unsqueeze(-1) * K * sW.unsqueeze(-2) + self.jitter * eye
        L = torch.linalg.cholesky(B)
        log_lik = log_phi.sum()
        log_q = (-0.5 * torch.dot(a, f - m) + log_lik
                 - torch.log(torch.diagonal(L)).sum())
        return f, a, log_q, L, sW, grad

    # ------------------------------------------------------- Algorithm 3.2
    def _predict_latent(self, x_star):
        """R&W Algorithm 3.2 lines 4-6: mean eq. (3.21), variance eq. (3.24)."""
        if self._f_hat is None:
            # A model reloaded from a checkpoint has hyperparameters but no
            # mode: the mode is not a parameter, it is the solution of
            # Algorithm 3.1 at those hyperparameters. Recompute it on first use
            # rather than making every caller remember to.
            self.refresh_mode()
        Ks = self._K(self.x_train, x_star.to(self.device))        # (n, m)
        f_bar = (Ks.transpose(-1, -2) @ self._grad_hat
                 + self.mean_const.to(Ks.dtype))                  # (m,)
        v = torch.linalg.solve_triangular(
            self._L, (self._sW.unsqueeze(-1) * Ks), upper=False)  # (n, m)
        k_ss = self._K(x_star.to(self.device), diag=True)
        var = (k_ss - v.pow(2).sum(dim=0)).clamp_min(1e-12)
        return f_bar, var

    def refresh_mode(self):
        """Re-run Newton at the current hyperparameters and cache the mode."""
        with torch.no_grad():
            K = self._K(self.x_train)
            f, a, log_q, L, sW, grad = self._newton(K, self.y_train)
        self._f_hat, self._grad_hat, self._L, self._sW = f, grad, L, sW
        return log_q

    def forward(self, x):
        f_bar, var = self._predict_latent(x)
        n = f_bar.shape[-1]
        cov = torch.diag_embed(var) + 1e-10 * torch.eye(
            n, dtype=var.dtype, device=var.device)
        return MultivariateNormal(f_bar.to(torch.float32), cov.to(torch.float32))

    def __call__(self, x, *args, **kwargs):
        return self.forward(x)

    def predictive_probability(self, x):
        """pi_bar of R&W eq. (3.25), analytic for the probit case."""
        f_bar, var = self._predict_latent(x)
        return torch.special.ndtr(f_bar / torch.sqrt(1.0 + var))

    # ------------------------------------------------------------- training
    def do_train_loop(self, lr=1e-3, iters=200, jitter=None, patience=None,
                      batch_size=None):
        """Maximise the Laplace approximate log marginal likelihood, eq. (3.32).

        Validation uses the held-out Bernoulli negative log predictive density,
        which is the proper analogue of the regression arm's validation MSE: a
        verdict model has no squared error to report.
        """
        if jitter is not None:
            self.jitter = float(jitter)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        best_loss, best_state, patience_counter = float("inf"), None, 0
        losses_train, losses_valid = [], []

        for i in range(iters):
            optimizer.zero_grad()
            K = self._K(self.x_train)
            _f, _a, log_q, _L, _sW, _grad = self._newton(K, self.y_train)
            loss = -log_q / self.x_train.shape[0]
            loss.backward()
            optimizer.step()
            losses_train.append(float(loss.detach()))

            with torch.no_grad():
                self.refresh_mode()
                p = self.predictive_probability(self.x_valid).clamp(1e-12, 1 - 1e-12)
                t = (self.y_valid > 0).to(p.dtype)
                val = float(-(t * p.log() + (1 - t) * (1 - p).log()).mean())
            losses_valid.append(val)

            if val < best_loss:
                best_loss, best_state, patience_counter = val, copy.deepcopy(self.state_dict()), 0
            else:
                patience_counter += 1
                if patience is not None and patience_counter >= patience:
                    logger.info(f"LaplaceGPC early stopping at iteration {i} "
                                f"(best val NLPD {best_loss:.6f})")
                    break
            if i % 10 == 0:
                logger.info(f"LaplaceGPC iter {i}: -log q/n = {losses_train[-1]:.6f}, "
                            f"val NLPD = {val:.6f}")

        if best_state is not None:
            self.load_state_dict(best_state)
        self.refresh_mode()
        return self, losses_train, losses_valid
