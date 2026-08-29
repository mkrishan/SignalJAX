# Copyright 2025 Manish Krishan Lal
# SPDX-License-Identifier: Apache-2.0
"""A modular, claim-aware calculus for learning and inference paradigms.

The taxonomy separates five independent choices: local relation, approximation family, geometry,
update rule, and readout.  Entries marked ``interface-only`` are comparison slots, not equivalence
claims.  Executable programs can nevertheless use the same trajectory and attenuation machinery.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from .operators import IterationTrace, PyTree, Readout, Transform, iterate_operator
from .signals import (
    SignalChannel,
    SignalGranularity,
    SignalTrace,
    readout_signal_trace,
)


class LocalRelation(str, Enum):
    """How one local architectural or probabilistic relation is represented."""

    HARD_DETERMINISTIC = "hard-deterministic"
    POSITIVE_KERNEL = "positive-kernel"
    PROBABILISTIC_FACTOR = "probabilistic-factor"
    ENERGY = "energy"
    LEARNED_INVERSE = "learned-inverse"
    LOCAL_OBJECTIVE = "local-objective"
    USER_DEFINED = "user-defined"


class ApproximationFamily(str, Enum):
    """Family in which local or global state is retained."""

    POINT = "point"
    EXACT_BELIEF = "exact-belief"
    PRODUCT_MEAN_FIELD = "product-mean-field"
    EXPONENTIAL_FAMILY = "exponential-family"
    MOMENT_MATCHED = "moment-matched"
    MAP = "map"
    LOCAL_REPRESENTATION = "local-representation"
    USER_DEFINED = "user-defined"


class Geometry(str, Enum):
    """Geometry used to enforce compatibility or form an update."""

    KL = "kl"
    EUCLIDEAN = "euclidean"
    WEIGHTED_EUCLIDEAN = "weighted-euclidean"
    BREGMAN = "bregman"
    ENERGY = "energy"
    USER_DEFINED = "user-defined"


class UpdateRule(str, Enum):
    """Local propagation or state-transition rule."""

    DIFFERENTIAL_READOUT = "differential-readout"
    SUM_PRODUCT = "sum-product"
    ITERATED_PROJECTION = "iterated-projection"
    VARIATIONAL_MESSAGE_PASSING = "variational-message-passing"
    EXPECTATION_PROPAGATION = "expectation-propagation"
    MAX_PRODUCT = "max-product"
    ENERGY_RELAXATION = "energy-relaxation"
    TARGET_TRANSPORT = "target-transport"
    FEEDBACK_ALIGNMENT = "feedback-alignment"
    FORWARD_LOCAL = "forward-local"
    USER_DEFINED = "user-defined"


class ReadoutSemantics(str, Enum):
    """Meaning assigned to the propagated quantity."""

    COTANGENT = "cotangent"
    MARGINAL = "marginal"
    POSTERIOR_MEAN = "posterior-mean"
    FINITE_RESIDUAL = "finite-residual"
    NATURAL_PARAMETER = "natural-parameter"
    MOMENT = "moment"
    MAP_STATE = "map-state"
    ENERGY_ERROR = "energy-error"
    LAYERWISE_TARGET = "layerwise-target"
    LOCAL_SCORE = "local-score"
    USER_DEFINED = "user-defined"


class ClaimLevel(str, Enum):
    """Strength of the relation asserted by this package."""

    ESTABLISHED = "established"
    ESTABLISHED_RESTRICTED = "established-restricted"
    COROLLARY = "corollary"
    DIAGNOSTIC = "diagnostic"
    INTERFACE_ONLY = "interface-only"
    CONJECTURAL = "conjectural"


@dataclass(frozen=True)
class ParadigmSpec:
    """One point in the modular message-passing design space."""

    identifier: str
    display_name: str
    local_relation: LocalRelation
    approximation: ApproximationFamily
    geometry: Geometry
    update_rule: UpdateRule
    readout: ReadoutSemantics
    claim_level: ClaimLevel
    scope_note: str
    references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.identifier or not self.display_name:
            raise ValueError("paradigm identifiers and display names must be non-empty")
        if not self.scope_note:
            raise ValueError(f"paradigm {self.identifier!r} needs an explicit scope note")


@dataclass(frozen=True)
class ParadigmRegistry:
    """Immutable registry supporting built-in and user-defined paradigm modules."""

    entries: tuple[ParadigmSpec, ...]

    def __post_init__(self) -> None:
        identifiers = tuple(entry.identifier for entry in self.entries)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("paradigm identifiers must be unique")

    def get(self, identifier: str) -> ParadigmSpec:
        """Return one specification by stable identifier."""

        for entry in self.entries:
            if entry.identifier == identifier:
                return entry
        raise KeyError(f"unknown paradigm {identifier!r}")

    def add(self, entry: ParadigmSpec) -> ParadigmRegistry:
        """Return a new registry containing one additional specification."""

        return ParadigmRegistry((*self.entries, entry))

    def by_claim_level(self, claim_level: ClaimLevel) -> tuple[ParadigmSpec, ...]:
        """Select all entries at one explicitly declared claim level."""

        return tuple(entry for entry in self.entries if entry.claim_level is claim_level)


@dataclass(frozen=True)
class ExecutableParadigm:
    """Bind a specification to a transition and a semantic readout."""

    specification: ParadigmSpec
    transition: Transform
    readout: Readout
    signal_channel: SignalChannel

    def __post_init__(self) -> None:
        if not callable(self.transition) or not callable(self.readout):
            raise TypeError("paradigm transitions and readouts must be callable")


@dataclass(frozen=True)
class ParadigmRun:
    """Finite state trajectory together with its paradigm-specific signal trace."""

    specification: ParadigmSpec
    iteration: IterationTrace
    signals: SignalTrace


def run_paradigm(
    program: ExecutableParadigm,
    initial_state: PyTree,
    *,
    steps: int,
    relaxation: float = 1.0,
    analytic_local_bounds: Sequence[float] | None = None,
) -> ParadigmRun:
    """Run any registered transition and analyze its declared readout channel."""

    iteration = iterate_operator(
        program.transition,
        initial_state,
        steps=steps,
        relaxation=relaxation,
    )
    labels = tuple(f"iteration-{index}" for index in range(len(iteration.states)))
    signals = readout_signal_trace(
        iteration.states,
        program.readout,
        name=program.specification.display_name,
        channel=program.signal_channel,
        granularity=SignalGranularity.ITERATION,
        labels=labels,
        analytic_local_bounds=analytic_local_bounds,
    )
    return ParadigmRun(program.specification, iteration, signals)


def standard_registry() -> ParadigmRegistry:
    """Return the built-in literature-aware design-space registry.

    The interface entries make methods comparable at the level of local objects, update schedules,
    and signal diagnostics.  They do not assert that predictive coding, target propagation, VMP, EP,
    feedback alignment, or forward-only learning is exactly the KL operator proved in the paper.
    """

    return ParadigmRegistry(
        (
            ParadigmSpec(
                "backpropagation",
                "Reverse-mode differential readout",
                LocalRelation.HARD_DETERMINISTIC,
                ApproximationFamily.POINT,
                Geometry.KL,
                UpdateRule.DIFFERENTIAL_READOUT,
                ReadoutSemantics.COTANGENT,
                ClaimLevel.ESTABLISHED,
                "The logarithmic differential identity holds at a fixed smooth forward point.",
                ("Parberry (1994)", "Baydin et al. (2018)", "Eaton (2020)"),
            ),
            ParadigmSpec(
                "tree-belief-propagation",
                "Exact tree sum-product",
                LocalRelation.PROBABILISTIC_FACTOR,
                ApproximationFamily.EXACT_BELIEF,
                Geometry.KL,
                UpdateRule.SUM_PRODUCT,
                ReadoutSemantics.MARGINAL,
                ClaimLevel.ESTABLISHED,
                "Exactness is restricted to finite connected trees with positive factors.",
                ("Walsh and Regalia (2010)", "Yedidia et al. (2005)"),
            ),
            ParadigmSpec(
                "spn-exact-marginals",
                "Complete/decomposable SPN marginals",
                LocalRelation.PROBABILISTIC_FACTOR,
                ApproximationFamily.EXACT_BELIEF,
                Geometry.KL,
                UpdateRule.SUM_PRODUCT,
                ReadoutSemantics.MARGINAL,
                ClaimLevel.ESTABLISHED,
                "Exactness requires positive evidence, completeness, and decomposability.",
                ("Darwiche (2003)", "Poon and Domingos (2011)"),
            ),
            ParadigmSpec(
                "finite-projection-learning",
                "Finite constraint projection dynamics",
                LocalRelation.HARD_DETERMINISTIC,
                ApproximationFamily.POINT,
                Geometry.EUCLIDEAN,
                UpdateRule.ITERATED_PROJECTION,
                ReadoutSemantics.FINITE_RESIDUAL,
                ClaimLevel.COROLLARY,
                "Cyclic projection is a direct finite-operator instance; convergence depends on "
                "the sets.",
                ("Pierra (1984)", "Bauschke and Combettes (2017)"),
            ),
            ParadigmSpec(
                "gaussian-softened-blr",
                "Conjugate Gaussian projection/BLR readout",
                LocalRelation.POSITIVE_KERNEL,
                ApproximationFamily.EXACT_BELIEF,
                Geometry.KL,
                UpdateRule.SUM_PRODUCT,
                ReadoutSemantics.POSTERIOR_MEAN,
                ClaimLevel.ESTABLISHED_RESTRICTED,
                "The Kalman/natural-gradient interpretation is restricted to the conjugate "
                "scalar tree.",
                ("Khan and Rue (2023)",),
            ),
            ParadigmSpec(
                "predictive-coding",
                "Predictive-coding local relaxation",
                LocalRelation.ENERGY,
                ApproximationFamily.POINT,
                Geometry.ENERGY,
                UpdateRule.ENERGY_RELAXATION,
                ReadoutSemantics.ENERGY_ERROR,
                ClaimLevel.INTERFACE_ONLY,
                "Agreement with backpropagation depends on architecture, equilibrium, and "
                "parameterization.",
                ("Whittington and Bogacz (2017)",),
            ),
            ParadigmSpec(
                "target-propagation",
                "Layerwise target transport",
                LocalRelation.LEARNED_INVERSE,
                ApproximationFamily.POINT,
                Geometry.USER_DEFINED,
                UpdateRule.TARGET_TRANSPORT,
                ReadoutSemantics.LAYERWISE_TARGET,
                ClaimLevel.INTERFACE_ONLY,
                "Finite targets are distinct from cotangents; users supply the inverse or "
                "target rule.",
                ("Lee et al. (2015)",),
            ),
            ParadigmSpec(
                "variational-message-passing",
                "Soft-factor variational message passing",
                LocalRelation.POSITIVE_KERNEL,
                ApproximationFamily.PRODUCT_MEAN_FIELD,
                Geometry.KL,
                UpdateRule.VARIATIONAL_MESSAGE_PASSING,
                ReadoutSemantics.NATURAL_PARAMETER,
                ClaimLevel.INTERFACE_ONLY,
                "Hard delta factors may be degenerate; this slot assumes a user-defined positive "
                "softening.",
                ("Yedidia et al. (2005)",),
            ),
            ParadigmSpec(
                "expectation-propagation",
                "Soft-factor expectation propagation",
                LocalRelation.POSITIVE_KERNEL,
                ApproximationFamily.MOMENT_MATCHED,
                Geometry.KL,
                UpdateRule.EXPECTATION_PROPAGATION,
                ReadoutSemantics.MOMENT,
                ClaimLevel.INTERFACE_ONLY,
                "Moment matching is represented as a distinct operator, not identified with "
                "exact BP.",
                ("Şenöz et al. (2021)",),
            ),
            ParadigmSpec(
                "max-product",
                "Max-product/MAP readout",
                LocalRelation.PROBABILISTIC_FACTOR,
                ApproximationFamily.MAP,
                Geometry.USER_DEFINED,
                UpdateRule.MAX_PRODUCT,
                ReadoutSemantics.MAP_STATE,
                ClaimLevel.INTERFACE_ONLY,
                "MAP readouts can be nonsmooth at ties and are not covered by the smooth "
                "differential theorem.",
            ),
            ParadigmSpec(
                "feedback-alignment",
                "Feedback-alignment signal",
                LocalRelation.LOCAL_OBJECTIVE,
                ApproximationFamily.POINT,
                Geometry.EUCLIDEAN,
                UpdateRule.FEEDBACK_ALIGNMENT,
                ReadoutSemantics.LOCAL_SCORE,
                ClaimLevel.INTERFACE_ONLY,
                "The feedback map is supplied explicitly and compared as its own signal channel.",
                ("Lillicrap et al. (2016)",),
            ),
            ParadigmSpec(
                "forward-only",
                "Forward-only local learning",
                LocalRelation.LOCAL_OBJECTIVE,
                ApproximationFamily.LOCAL_REPRESENTATION,
                Geometry.USER_DEFINED,
                UpdateRule.FORWARD_LOCAL,
                ReadoutSemantics.LOCAL_SCORE,
                ClaimLevel.INTERFACE_ONLY,
                "Local objectives and readouts are method-specific and supplied by the user.",
                (
                    "Ye et al. (2026), Beyond-Backpropagation Training",
                    "Huang, Ororbia, and Aminifar (2026), Backpropagation-Free Learning",
                ),
            ),
        )
    )
