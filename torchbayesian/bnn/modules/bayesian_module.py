"""
    @file:              bayesian_module.py
    @Author:            Raphael Brodeur

    @Creation Date:     07/2025
    @Last modification: 02/2026

    @Description:       This file contains the 'BayesianModule' class, a Torch 'nn.Module' container used to
                        reparametrize the parameters of any torch model or module with a variational posterior. The
                        wrapped module or model is thus converted into a Bayesian neural network (BNN) via variational
                        inference (VI) through Bayes by Backprop (BBB).
"""

from typing import (
    Any,
    Dict,
    Optional,
    Tuple
)
import warnings

import torch
from torch import Tensor
import torch.distributions as D
from torch.distributions import Distribution
from torch.distributions.kl import _dispatch_kl
from torch.nn import Module
from torch.types import Device

from torchbayesian.bnn.utils import (
    get_posterior,
    get_prior,
    register_reparametrization
)
from torchbayesian.bnn.variational_posteriors import VariationalPosterior
from torchbayesian.types import _dtype


__all__ = ["BayesianModule"]


class BayesianModule(Module):
    """
    This class is a Torch 'nn.Module' container that reparametrizes in-place the 'nn.Parameter' parameters of any torch
    model or module with a variational posterior, and allows computation of the KL divergence.

    The wrapped modules are thus converted into a Bayesian neural network (BNN) via variational inference (VI) through
    Bayes by Backprop (BBB), as described in "Weight Uncertainty in Neural Networks" by Blundell et al.

    After construction, 'module' original parameters are no longer optimizable 'nn.Parameter'; instead, the trainable
    'nn.Parameter' weights live inside the injected reparametrizations.

    Parameters
    ----------
    module: Module
        The module or model to turn into a Bayesian neural network (BNN). Its parameters will be reparameterized
        in-place.
    variational_posterior : Optional[str | Tuple[str, Dict[str, Any]]]
        The variational posterior distribution for the parameters. Either the name (str) of the variational posterior or
        a tuple of its name and keyword arguments. Defaults to 'GaussianPosterior'.
    prior : Optional[str | Tuple[str, Dict[str, Any]]]
        The prior distribution for the parameters. Either the name (str) of the prior or a tuple of its name and
        keyword arguments. Defaults to 'GaussianPrior' with zero mean and unit standard deviation.
    dtype: Optional[_dtype]
        The dtype on which the KL divergence accumulator reference buffer is initialized. A buffer is initialized in
        order to track the device and dtype of the module's parameters through internal calls to '_apply' so that the KL
        accumulator's device and dtype fit that of the module's parameters. Optional. Defaults to torch default dtype.
        It is recommended to use 'BayesianModule(...).to(device, dtype)' instead of this argument !
    device: Device
        The device on which the KL divergence accumulator reference buffer is initialized. A buffer is initialized in
        order to track the device and dtype of the module's parameters through internal calls to _apply so that the KL
        accumulator's device and dtype fit that of the module's parameters. Optional. Defaults to torch default device.
        It is recommended to use 'BayesianModule(...).to(device, dtype)' instead of this argument !
    verbose : bool
        Whether to print a message for each parameter being reparameterized. Defaults to False.

    Attributes
    ----------
    module : nn.Module
        The reparameterized module or model.

    Notes
    -----
    By default, a Gaussian variational posterior distribution and a Gaussian prior distribution are used. This allows
    an analytical evaluation of the KL divergence and is a common standard in BBB VI, even though the original paper
    proposes a scale mixture of Gaussians as the prior.

    Custom variational posteriors and priors can be used by registering a factory function to 'PosteriorFactory' or
    'PriorFactory', as described in the docs of 'Factory' in 'torchbayesian.bnn.utils.factories'.

    Warning
    -------
    This should be applied before registering model parameters to an optimizer. Otherwise, the new variational
    parameters must be manually registered to the optimizer.
    """

    module: Module
    _prior: str | Tuple[str, Dict[str, Any]]
    _kl_meta: Tensor

    def __init__(
            self,
            module: Module,
            variational_posterior: Optional[str | Tuple[str, Dict[str, Any]]] = None,
            prior: Optional[str | Tuple[str, Dict[str, Any]]] = None,
            *,
            dtype: Optional[_dtype] = None,
            device: Device = None,
            verbose: bool = False
    ) -> None:
        """
        Turns a module or model into a Bayesian neural network (BNN).

        Reparametrizes in-place all the 'nn.Parameter' parameters of 'module' with variational posteriors, thus
        converting 'module' into a Bayesian neural network (BNN) via variational inference (VI) through Bayes by
        Backprop (BBB), as described in "Weight Uncertainty in Neural Networks" by Blundell et al.

        After construction, 'module' original parameters are no longer optimizable 'nn.Parameter'; instead, the
        trainable 'nn.Parameter' weights live inside the injected reparametrizations.

        Parameters
        ----------
        module: Module
            The module or model to turn into a Bayesian neural network (BNN). Its parameters will be reparameterized
            in-place.
        variational_posterior : Optional[str | Tuple[str, Dict[str, Any]]]
            The variational posterior distribution for the parameters. Either the name (str) of the variational
            posterior or a tuple of its name and keyword arguments. Defaults to 'GaussianPosterior'.
        prior : Optional[str | Tuple[str, Dict[str, Any]]]
            The prior distribution for the parameters. Either the name (str) of the prior or a tuple of its name and
            keyword arguments. Defaults to 'GaussianPrior' with zero mean and unit standard deviation.
        dtype: Optional[_dtype]
            The dtype on which the KL divergence accumulator reference buffer is initialized. A buffer is initialized in
            order to track the device and dtype of the module's parameters through internal calls to '_apply' so that
            the KL accumulator's device and dtype fit that of the module's parameters. Optional. Defaults to torch
            default dtype. It is recommended to use 'BayesianModule(...).to(device, dtype)' instead of this argument !
        device: Device
            The device on which the KL divergence accumulator reference buffer is initialized. A buffer is initialized
            in order to track the device and dtype of the module's parameters through internal calls to _apply so that
            the KL accumulator's device and dtype fit that of the module's parameters. Optional. Defaults to torch
            default device. It is recommended to use 'BayesianModule(...).to(device, dtype)' instead of this argument !
        verbose : bool
            Whether to print a message for each parameter being reparameterized. Defaults to False.

        Warning
        -------
        This class should be constructed before registering the model parameters to an optimizer. Otherwise, the new
        variational parameters must be manually registered to the optimizer.

        Notes
        -----
        By default, a Gaussian variational posterior distribution and a Gaussian prior distribution are used. This
        allows an analytical evaluation of the KL divergence and is a common standard in BBB VI, even though the
        original paper proposes a scale mixture of Gaussians as the prior.

        Custom variational posteriors and priors can be used by registering a factory function to 'PosteriorFactory' or
        'PriorFactory', as described in the docs of 'Factory' in 'torchbayesian.bnn.utils.factories.factory'.
        """
        super().__init__()

        if variational_posterior is None:
            variational_posterior = "NORMAL"                # Defaults to GaussianPosterior
        if prior is None:
            prior = ("NORMAL", {"mu": 0., "sigma": 1.})     # Defaults to GaussianPrior with mu = 0 and sigma = 1

        original_training_flag = module.training    # Register training flag of original module

        # Replace every parameter in the model by an instance of the variational posterior
        for name, param in list(module.named_parameters()):
            # Get name of parameter and name of the module owning it
            if "." in name:
                owner_module_name, param_name = name.rsplit('.', 1)
            else:
                owner_module_name, param_name = ("", name)

            # Register reparametrization to parameter
            if verbose:
                print(
                    f"Registering to parameter '{param_name}' of module '{module.get_submodule(owner_module_name)}'..."
                )
            register_reparametrization(
                module=module.get_submodule(owner_module_name),
                tensor_name=param_name,
                parametrization=get_posterior(
                    param=param,
                    posterior=variational_posterior
                ).to(device=param.device, dtype=param.dtype)
            )

        module.train(original_training_flag)  # Put BNN in same training mode as original module

        self.module = module
        self._prior = prior
        self.register_buffer("_kl_meta", torch.empty(0, device=device, dtype=dtype), persistent=False)

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass of the module.

        Parameters
        ----------
        input : Tensor
            The input tensor.

        Returns
        -------
        output : Tensor
            The output tensor.
        """
        return self.module(input)

    @staticmethod
    def _MC_approx_kl_divergence(
            posterior: Distribution,
            prior: Distribution,
            num_samples: int
    ) -> Tensor:
        """
        Computes the KL divergence KL(posterior || prior) of the 'BayesianModule' through a Monte Carlo approximation.

        Core idea is that KL(q || p) = Expectation[log q(w) - log p(w)] over w ~ q(w).

        Parameters
        ----------
        posterior : Distribution
            The variational posterior distribution.
        prior : Distribution
            The prior distribution.
        num_samples : int
            The number of samples for the MC approximation of the KL divergence.

        Returns
        -------
        approx_kl_div : Tensor
            The MC-approximate KL divergence KL(posterior || prior) of the 'BayesianModule'.
        """
        # Sample w ~ q(w)
        posterior_samples = posterior.rsample((num_samples, ))

        # MC approx of E_{w~q(w)}[log q(w) - log p(w)]
        approx_kl_div = (
            posterior.log_prob(posterior_samples) - prior.log_prob(posterior_samples)
        ).mean(0)

        return approx_kl_div

    def _compute_kl_divergence(
            self,
            variational_posterior: Distribution,
            prior: Distribution,
            approx_num_samples: Optional[int] = None
    ) -> Tensor:
        """
        Computes the elementwise KL divergence KL(posterior || prior) of a parameter from the 'BayesianModule'.

        If analytical solution is not defined, falls back to MC approximation of the KL divergence by sampling from the
        posterior and prior.

        Parameters
        ----------
        variational_posterior : Distribution
            The variational posterior distribution.
        prior : Distribution
            The prior distribution.
        approx_num_samples : Optional[int]
            The number of samples for the MC approximation of the KL divergence. Only useful if no analytical solution
            of the KL divergence between the posterior and prior distributions is implemented. Defaults to None.

        Returns
        -------
        kl_div : Tensor
            The KL divergence KL(posterior || prior) of the 'BayesianModule'.

        Raises
        ------
        ValueError
            If no analytical solution is implemented and 'approx_num_samples' is None.

        Warns
        -----
        UserWarning
            If MC approximation of the KL divergence is used but there is an analytical solution available.
        """
        if _dispatch_kl(type(variational_posterior), type(prior)) is not NotImplemented:
            # Analytical solution is implemented;
            # Warn user if he is still using MC approximation of the KL divergence
            if approx_num_samples is not None:
                warnings.warn(
                    f"You are using a Monte Carlo approximation of the KL div, but an analytical solution is "
                    f"implemented for distributions {type(variational_posterior)} and {type(prior)}. It is recommended "
                    f"that you set 'approx_num_samples' to None and use the default analytical solution instead.",
                    UserWarning
                )
        elif approx_num_samples is None:
            # Analytical solution is not implemented;
            # Check if number of samples is specified for MC approximation
            raise ValueError(
                f"No analytical solution implemented for distributions {type(variational_posterior)} and {type(prior)}."
                f" You can use a Monte-Carlo approximation instead by specifying the 'approx_num_samples' argument."
            )

        if approx_num_samples is None:
            kl_div = D.kl_divergence(variational_posterior, prior)
        else:
            kl_div = self._MC_approx_kl_divergence(variational_posterior, prior, approx_num_samples)

        return kl_div

    def kl_divergence(
            self,
            *,
            reduction: str = "sum",
            approx_num_samples: Optional[int] = None
    ) -> Tensor:
        """
        Gets the KL divergence KL(posterior || prior) of all parameters in the 'BayesianModule'.

        If analytical solution is not defined between the two distributions, MC approximation of the KL divergence can
        be used.

        Parameters
        ----------
        reduction : str
            The reduction to apply to the full elementwise parameters KL divergence. Either "sum" (sum of all elements
            making up all the parameters) or "mean" (mean of all elements making up all the parameters). In theory,
            true ELBO uses the sum of the elementwise KL divergences, but in practice this can scale badly with model
            size and mini-batching. Therefore, in practice, it is not uncommon to scale the KL divergence or to use a
            mean reduction of the KL divergence . Defaults to "sum".
        approx_num_samples : Optional[int]
            The number of samples for the MC approximation of the KL divergence. Only useful if no analytical solution
            of the KL divergence between the posterior and prior distributions is implemented. Defaults to None.

        Returns
        -------
        kl_div : Tensor
            The KL divergence KL(posterior || prior) of the 'BayesianModule'.

        Raises
        ------
        ValueError
            If reduction type is invalid.

        Warning
        -------
        KL divergence is computed using an accumulator in order to avoid the overhead with using a list of KL terms, but
        the accumulator must be on appropriate device which is why 'BayesianModule' tracks the buffer '_kl_meta'.
        As such, if 'BayesianModule' is not initialized on same device as the original module, and if no move to
        appropriate dtype/device is done afterward (e.g. using 'net.to(...)' or 'net.cuda()'), then the accumulator's
        dtype/device might not fit with KL divergence terms coming from the parameters. This is easily fixable
        by calling .to(...) to move all parameters and buffers of the 'BayesianModule' to the same dtype/device.

        TODO -- Currently, this feature may not work for distributions where factorization is not possible
                Element-wise 1d KL -> Sum (or mean) breaks down.
        """
        reduction = reduction.lower()

        if reduction not in {"mean", "sum"}:
            raise ValueError(f"Invalid reduction type: '{reduction}'. Expected 'mean' or 'sum'.")

        kl_div = torch.zeros((), dtype=self._kl_meta.dtype, device=self._kl_meta.device)    # Accumulator
        num_elements = 0 if reduction == "mean" else None       # Count elements for mean reduction
        for name, module in self.named_modules():
            # Compute KL divergence only for BNN modules
            if isinstance(module, VariationalPosterior):
                # Get posterior distribution
                posterior_dist = module.distribution

                # Instantiate prior and get its distribution
                prior_dist = get_prior(
                    shape=module.shape,
                    prior=self._prior,
                    dtype=module.dtype,
                    device=module.device,
                ).distribution

                # Compute KL divergence
                kl_div_elementwise = self._compute_kl_divergence(
                    variational_posterior=posterior_dist,
                    prior=prior_dist,
                    approx_num_samples=approx_num_samples
                )

                # Accumulation
                kl_div += kl_div_elementwise.sum()
                if reduction == "mean":
                    num_elements += kl_div_elementwise.numel()

        # Mean reduction
        if reduction == "mean":
            kl_div = kl_div / num_elements

        return kl_div
