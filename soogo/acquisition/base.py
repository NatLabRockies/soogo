"""Acquisition functions for surrogate optimization."""

# Copyright (c) 2025 Alliance for Energy Innovation, LLC

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

__authors__ = ["Weslley S. Pereira"]

import numpy as np
from abc import ABC
from typing import Optional

# Pymoo imports
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.core.mixed import MixedVariableGA, MixedVariableMating

# Local imports
from ..model import Surrogate
from ..integrations.pymoo import ListDuplicateElimination
from ..termination import TerminationCondition
from ..optimize.result import OptimizeResult
from ..utils import report_unused_kwargs


class Acquisition(ABC):
    """Base class for acquisition functions.

    This an abstract class. Subclasses must implement the method
    :meth:`optimize()`.

    Acquisition functions are strategies to propose new sample points to a
    surrogate. The acquisition functions here are modeled as objects with the
    goals of adding states to the learning process. Moreover, this design
    enables the definition of the :meth:`optimize()` method with a similar API
    when we compare different acquisition strategies.

    :param optimizer: Continuous optimizer to be used for the acquisition
        function. Default is Differential Evolution (DE) from pymoo.
    :param mi_optimizer: Mixed-integer optimizer to be used for the acquisition
        function. Default is Genetic Algorithm (MixedVariableGA) from pymoo.
    :param rtol: Minimum distance between a candidate point and the
        previously selected points relative to the domain size. Default is 1e-6.

    .. attribute:: optimizer

        Continuous optimizer to be used for the acquisition function. This is
        used in :meth:`optimize()`.

    .. attribute:: mi_optimizer

        Mixed-integer optimizer to be used for the acquisition function. This is
        used in :meth:`optimize()`.

    .. attribute:: rtol

        Minimum distance between a candidate point and the previously selected
        points.  This figures out as a constraint in the optimization problem
        solved in :meth:`optimize()`.
    """

    def __init__(
        self,
        optimizer=None,
        mi_optimizer=None,
        rtol: float = 1e-6,
        termination: Optional[TerminationCondition] = None,
    ) -> None:
        self.optimizer = DE() if optimizer is None else optimizer
        self.mi_optimizer = (
            MixedVariableGA(
                eliminate_duplicates=ListDuplicateElimination(),
                mating=MixedVariableMating(
                    eliminate_duplicates=ListDuplicateElimination()
                ),
            )
            if mi_optimizer is None
            else mi_optimizer
        )
        self.rtol = rtol
        self.termination = termination

    @classmethod
    def report_unused_optimize_kwargs(cls, kwargs) -> None:
        """Report any unused keyword arguments passed to the acquisition
        function.

        :param kwargs: Dictionary of keyword arguments.
        """
        report_unused_kwargs(f"{cls.__name__}.optimize", kwargs)

    def optimize(
        self,
        surrogateModel: Surrogate,
        bounds,
        n: int = 0,
        xbest=None,
        ybest=None,
        constr=None,
        exclusion_set: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Get n sample points that optimize the acquisition function.

        Optional parameters are only available in specific acquisition
        classes. Look documentation in the respective subclass.

        :param surrogateModel: Surrogate model.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Number of points requested.
        :param xbest: Best point(s) found so far.
        :param ybest: Best objective value(s) found so far.
        :param constr: Constraint function for candidate points.
        :param exclusion_set: Known points, if any, in addition to the ones
            used to train the surrogate.
        :param kwargs: Additional keyword arguments. Unused kwargs should be
            reported using :meth:`report_unused_optimize_kwargs`.

        :return: k-by-dim matrix with the selected points, where k <= n.
        """
        # Report non-used kwargs
        self.report_unused_optimize_kwargs(kwargs)

        return np.empty((n, len(bounds)))

    def tol(self, bounds) -> float:
        """Compute tolerance used to eliminate points that are too close to
        previously selected ones.

        The tolerance value is based on :attr:`rtol` and the diameter of the
        largest d-dimensional cube that can be inscribed whithin the bounds.

        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        """
        return (
            self.rtol
            * np.sqrt(len(bounds))
            * np.min([abs(b[1] - b[0]) for b in bounds])
        )

    def has_converged(self) -> bool:
        """Check if the acquisition function has converged.

        This method is used to check if the acquisition function has converged
        based on a termination criterion. The default implementation always
        returns False.
        """
        if self.termination is not None:
            return self.termination.is_met()
        else:
            return False

    def update(self, out: OptimizeResult, model: Surrogate) -> None:
        """Update the acquisition function knowledge about the optimization
        process.

        :param out: Current optimization result containing evaluation history.
        :param model: Updated surrogate model.
        """
        if self.termination is not None:
            self.termination.update(out, model)
