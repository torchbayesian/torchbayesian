"""
    @file:              prior.py
    @Author:            Raphael Brodeur

    @Creation Date:     08/2025
    @Last modification: 02/2026

    @Description:       This file contains the 'Prior' base class for all prior distributions used for Bayes by
                        Backprop (BBB) variational inference (VI).
"""

from abc import ABC, abstractmethod

from torch.distributions import Distribution


__all__ = ["Prior"]


class Prior(ABC):
    """
    This class serves as a base class for all priors used for Bayes by backprop (BBB) variational inference (VI).

    Base class implemented for consistency in the case of possible future implementation of learnable priors and making
    Prior a child of 'nn.Module' similarly to 'bnn.VariationalPosterior'.

    Warning
    -------
    Subclasses of 'Prior' used in 'bnn.BayesianModule' must work with 'get_prior()'; their constructor ('__init__')
    method must accept arguments 'shape', 'dtype' and 'device'.
    """

    @property
    @abstractmethod
    def distribution(self) -> Distribution:
        """
        A 'torch.distributions.Distribution' corresponding to the prior.

        This is used for KL computation aligned with PyTorch's framework.

        Returns
        -------
        distribution : Distribution
            A 'torch.distributions.Distribution' corresponding to the prior.
        """
        raise NotImplementedError

    def extra_repr(self) -> str:
        """
        Returns the extra representation of the prior.

        To print customized extra information, you should re-implement this method in your own priors. Both single-line
        and multi-line strings are acceptable.
        """
        return ""

    def __repr__(self) -> str:
        """
        Representation of the prior.

        Returns
        -------
        main_str : str
            The str representation of the prior.
        """
        main_str = self.__class__.__name__ + "("

        # extra representation
        extra_repr = self.extra_repr()

        # Get extra lines
        extra_lines = []
        if extra_repr:
            extra_lines = extra_repr.split("\n")

        # Add extra lines to main representation
        if extra_lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(extra_lines) + "\n"

        main_str += ")"

        return main_str
