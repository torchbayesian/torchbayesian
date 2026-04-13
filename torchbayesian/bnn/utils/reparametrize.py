"""
    @file:              reparametrize.py
    @Author:            Raphael Brodeur, PyTorch

    @Creation Date:     07/2025
    @Last modification: 02/2026

    @Description:       This file contains the 'register_reparametrization()' function which is used to replace a Torch
                        parameter or buffer by a variational posterior 'nn.Module' from which the parameter or buffer is
                        sampled. This is done via a property injection mechanism. Adapted from PyTorch's
                        'nn.utils.parametrize' in order to remove the original parameter from the registered parameters
                        and the state dict.
"""

from copy import deepcopy
import copyreg
from typing import (
    Dict,
    Optional,
    Tuple
)

import torch
from torch import Tensor
from torch.nn import Module, ModuleDict


__all__ = ["Reparametrization", "register_reparametrization"]


_cache_enabled = 0
_cache: Dict[Tuple[int, str], Optional[Tensor]] = {}


class Reparametrization(Module):
    """
    This class wraps the variational posterior module and handles safety checks for the replacement of the parameter or
    buffer by the variational posterior and its forward call.

    It is the type of 'module.reparametrizations[tensor_name]' when 'module[tensor_name]' has been reparametrized with
    'register_reparametrization()'.

    Parameters
    ----------
    parametrization : Module
        The parametrization function; the variational posterior in the context of Bayes by Backprop.
    original : Tensor
        The tensor that is being reparametrized.
    unsafe : bool
        Whether to bypass correctness checks. Optional. Defaults to False.

    Attributes
    ----------
    variational_posterior : Module
        The parametrization replacing 'original'.

    Notes
    -----
    This class is used internally by 'register_reparametrization()'. It shall not be instantiated by the user.
    """

    variational_posterior: Module

    def __init__(
            self,
            parametrization: Module,
            original: Tensor,
            *,
            unsafe: bool = False
    ) -> None:
        """
        Wraps a stochastic reparametrization and checks if the 'parametrization' function fits with original tensor.

        Parameters
        ----------
        parametrization : Module
            The stochastic reparametrization function; the variational posterior in the context of Bayes by Backprop.
        original : Tensor
            The tensor that is being reparametrized.
        unsafe : bool
            Whether to bypass correctness checks. Optional. Defaults to False.

        Raises
        ------
        TypeError
            If unsafe is set to False and the parametrization's forward call does not return a Tensor.
        TypeError
            If unsafe is set to False and the parametrization's forward call does not return a Tensor with the same
            dtype as the original tensor.
        ValueError
            If unsafe is set to False and the parametrization's forward call does not return a Tensor with the same
            shape as the original tensor.
        """
        super().__init__()

        # Correctness checks
        if not unsafe:
            new = parametrization()     # Get output tensor from reparametrization

            # Check if parametrization's forward call returns a Tensor
            if not isinstance(new, Tensor):
                raise TypeError(
                    f"A reparametrization must return a tensor. Got {type(new).__name__}."
                )

            # Check if dtypes match
            if new.dtype != original.dtype:
                raise TypeError(
                    "Reparametrization may not change the dtype of the tensor, unless the unsafe flag is enabled.\n"
                    f"Original tensor has dtype: {original.dtype}\n"
                    f"Reparametrization returns dtype: {new.dtype}"
                )

            # Check if shapes match
            if new.shape != original.shape:
                raise ValueError(
                    "Reparametrization may not change the shape of the tensor, unless the `unsafe` flag is enabled.\n"
                    f"Original tensor has shape: {original.shape}\n"
                    f"Reparametrization returns shape: {new.shape}"
                )

        self.variational_posterior = parametrization

    def forward(self) -> Tensor:
        """
        Wraps the parametrization's forward call. Checks for scripting.

        In the context of Bayes by Backprop, this returns a sample from the variational posterior.
        """
        if torch.jit.is_scripting():
            raise RuntimeError("Reparametrization is not working with scripting.")

        x = self.variational_posterior()

        return x

    def extra_repr(self) -> str:
        """
        Returns the variational posterior put in place.

        Returns
        -------
        extra_repr : str
            The str extra representation of the reparametrization.
        """
        return f"{self.variational_posterior}"

    def __repr__(self) -> str:
        """
        Representation of the reparametrization.

        Overwrites nn.Module.__repr__ in order to supress child modules representation.

        Returns
        -------
        repr : str
            The str representation of the reparametrization.
        """
        return f"{self.__class__.__name__}({self.extra_repr()})"


def register_reparametrization(
    module: Module,
    tensor_name: str,
    parametrization: Module,
    *,
    unsafe: bool = False
) -> Module:
    """
    Replaces a tensor (parameter or buffer) in a module by registering a stochastic reparametrization module in its
    place.

    Assume that 'tensor_name="weight"' for simplicity. After reparametrization, the original tensor 'module.weight' is
    removed and replaced by a Python property. Accessing 'module.weight' now calls a corresponding 'Reparametrization'
    module which returns a tensor returned by calling 'parametrization()' (typically, that is a tensor sampled from the
    variational posterior), rather than using the original tensor.

    If the original tensor requires a gradient, the backward pass differentiates through the reparametrization module
    and the optimizer updates the variational parameters of the reparametrization module instead of the original
    parameter. The parameters and buffers of 'parametrization' are registered to the model and state_dict, and the
    original tensor is removed from state_dict.

    Parameters
    ----------
    module : Module
        The module whose tensor is to be reparametrized.
    tensor_name : str
        The name of the parameter or buffer to reparametrize.
    parametrization : Module
        The module with which to replace the tensor. Typically, this is the variational posterior.
    unsafe : bool
        Whether to bypass correctness checks. Optional. Defaults to False.

    Returns
    -------
    module : Module
        The module, reparametrized in-place.

    Examples
    --------
        mod = nn.Linear(2, 4)
        # This module has parameters 'mod.bias' and 'mod.weight'

        # We replace the parameter 'mod.weight' by a Gaussian variational posterior with learnable parameters
        # corresponding to the mean and standard deviation :
        register_reparametrization(mod, "weight", bnn.GaussianPosterior(...))

        # The module 'mod' now has parameters 'mod.bias', 'mod.reparametrizations.weight.variational_posterior.mu'
        # and 'mod.reparametrizations.weight.variational_posterior.rho'
    """
    parametrization.train(module.training)  # Put parametrization in same training mode as module

    # Set the parametrization mechanism
    if tensor_name in module._buffers or tensor_name in module._parameters:
        # Fetch the original buffer or parameter
        original = getattr(module, tensor_name)

        # Wrap the parametrization with Reparametrization to check for errors
        reparametrization = Reparametrization(parametrization, original, unsafe=unsafe)

        # Delete the previous parameter or buffer that is now reparametrized
        delattr(module, tensor_name)

        # If this is the first reparametrization registered on the module,
        # we prepare the module to inject the property
        if not is_reparametrized(module):
            # Sets up the module to be reparametrized. Adds prefix Reparametrized to module's name
            _inject_new_class(module)

            # Inject a ModuleDict into the instance under module.reparametrizations
            module.reparametrizations = ModuleDict()

        # Add a property into the class
        _inject_property(module, tensor_name)

        # Add a Reparametrization
        assert isinstance(module.reparametrizations, ModuleDict)  # Make mypy happy
        module.reparametrizations[tensor_name] = reparametrization

    else:
        raise ValueError(
            f"Module '{module}' does not have a parameter, a buffer, or a "
            f"parametrized element with name '{tensor_name}'"
        )

    return module


def is_reparametrized(module: Module, tensor_name: Optional[str] = None) -> bool:
    """
    Whether the module has a reparametrization.

    If tensor_name is specified, checks for the specified parameter or
    buffer, otherwise checks for all parameters and buffers in the module.

    Parameters
    ----------
    module : Module
        The module to check for.
    tensor_name : Optional[str]
        The name of the parameter or buffer to check for within the module. Optional. Defaults to None.

    Returns
    -------
    is_reparametrized : bool
        Whether the module or the tensor is reparametrized.
    """
    reparametrizations = getattr(module, "reparametrizations", None)

    if reparametrizations is None or not isinstance(reparametrizations, ModuleDict):
        return False

    if tensor_name is None:
        # Check that there is at least one parametrized buffer or Parameter
        return len(reparametrizations) > 0

    else:
        return tensor_name in reparametrizations


def _inject_new_class(module: Module) -> None:
    """
    Sets up a module to be reparametrized.

    This works by substituting the class of the module by a class that extends it to be able to inject a property.
    Adds the prefix Reparametrized to the module's name.

    Parameters
    ----------
    module : Module
        The module to prepare for reparametrization.
    """
    cls = module.__class__

    def default_deepcopy(self, memo):
        """
        Emulates a standard deepcopy procedure when __deepcopy__ doesn't exist in the current class.
        """
        obj = memo.get(id(self), None)

        if obj is not None:
            return obj

        replica = self.__new__(self.__class__)
        memo[id(self)] = replica
        replica.__dict__ = deepcopy(self.__dict__, memo)

        # Also save all slots if they exist.
        slots_to_save = copyreg._slotnames(self.__class__)  # type: ignore[attr-defined]
        for slot in slots_to_save:
            if hasattr(self, slot):
                setattr(replica, slot, deepcopy(getattr(self, slot), memo))

        return replica

    def getstate(self):
        raise RuntimeError(
            "Serialization of parametrized modules is only "
            "supported through state_dict(). See:\n"
            "https://pytorch.org/tutorials/beginner/saving_loading_models.html"
            "#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training"
        )

    dct = {"__getstate__": getstate}

    # We don't allow serialization of parametrized modules but should still allow deep-copying.
    # Default 'deepcopy' function invokes __deepcopy__ method instead of __getstate__ when it exists.
    if not hasattr(cls, "__deepcopy__"):
        dct["__deepcopy__"] = default_deepcopy  # type: ignore[assignment]

    param_cls = type(
        f"Reparametrized{cls.__name__}",
        (cls,),
        dct,
    )

    module.__class__ = param_cls


def _inject_property(module: Module, tensor_name: str) -> None:
    """
    Injects a property into module[tensor_name].

    This function is the core of the reparametrization mechanism; this function replaces 'module[tensor_name]' (e.g.
    'module.weight') by a python property whose getter function computes/samples a tensor by calling the
    'Reparametrization' module (which itself calls the variational posterior).

    It assumes that the class in the module has already been modified from its original one using '_inject_new_class'
    and that the tensor under :attr:`tensor_name` has already been moved out.

    Same as in PyTorch's implementation but the getter function is modified to get reparametrization instead of
    parametrization, and the setter function is modified to prevent assignment to the original parameter tensor (which
    is replaced by a call to a 'Reparametrization' module). This might be changed in the future to allow assigning a new
    'Reparametrization'.

    Parameters
    ----------
    module : Module
        The module into which a property is injected.
    tensor_name : str
        The name of the property to create.
    """
    # We check if an attribute already exists under that name
    # This should never fire if 'register_reparametrization' is correctly implemented.
    # (We already 'delattr(module, tensor_name)' in 'register_reparametrization',
    # this ensures we are not overwriting some attribute)
    assert not hasattr(module, tensor_name)

    # Caching helper
    @torch.jit.unused
    def get_cached_reparametrization(reparametrization) -> Tensor:
        global _cache
        key = (id(module), tensor_name)
        tensor = _cache.get(key)
        if tensor is None:
            tensor = reparametrization()
            _cache[key] = tensor
        return tensor

    # Getter function
    # Returns a call to the 'Reparametrization' Module
    def get_reparametrized(self) -> Tensor:
        if torch.jit.is_scripting():
            raise RuntimeError("Reparametrization is not working with scripting.")

        reparametrization = self.reparametrizations[tensor_name]

        if _cache_enabled:
            if torch.jit.is_scripting():
                # Scripting
                raise RuntimeError(
                    "Caching is not implemented for scripting. "
                    "Either disable caching or avoid scripting."
                )
            elif torch._C._get_tracing_state() is not None:
                # Tracing
                raise RuntimeError(
                    "Cannot trace a model while caching reparametrizations."
                )
            else:
                return get_cached_reparametrization(reparametrization)

        else:
            # If caching is not active, this function just evaluates the reparametrization
            return reparametrization()

    # Setter function
    # For now, raises an error to prevent assignment to the original tensor
    def set_original(self, value: Tensor) -> None:
        # if torch.jit.is_scripting():
        #     raise RuntimeError("Reparametrization is not working with scripting.")

        raise RuntimeError(
            f"Cannot assign to '{tensor_name}' because it is reparametrized. "
            f"Access the reparametrization's attributes (e.g. parameters) via "
            f"'module.reparametrizations['{tensor_name}'].variational_posterior'."
        )

    # Inject the property with the specified getter and setter functions
    setattr(module.__class__, tensor_name, property(get_reparametrized, set_original))
