"""Configuration and fallback tests for adaptive revision selection."""

import pytest

from src.revision import AdaptiveRevisionStrategy, RevisionStrategy


@pytest.mark.parametrize("strategy", list(RevisionStrategy))
@pytest.mark.parametrize("entailment_rate", [0.0, 0.5, 0.8, 1.0])
def test_single_enabled_strategy_is_always_selected(strategy, entailment_rate):
    revision = AdaptiveRevisionStrategy(strategies=[strategy.value])

    selected = revision._select_strategy({"entailment_rate": entailment_rate}, 0)

    assert selected == strategy


@pytest.mark.parametrize(
    ("entailment_rate", "expected"),
    [
        (0.0, RevisionStrategy.RE_RETRIEVAL),
        (0.5, RevisionStrategy.CONSTRAINED_GENERATION),
        (0.8, RevisionStrategy.CONSTRAINED_GENERATION),
    ],
)
def test_fallback_uses_enabled_strategy(entailment_rate, expected):
    revision = AdaptiveRevisionStrategy(
        strategies=["re_retrieval", "constrained_generation"]
    )

    assert revision._select_strategy({"entailment_rate": entailment_rate}, 0) == expected


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"strategy_selection_mode": "random"}, "strategy_selection_mode"),
        ({"strategies": []}, "At least one strategy"),
        ({"strategies": ["unknown"]}, "Invalid strategies"),
    ],
)
def test_invalid_dynamic_configuration_fails_at_startup(kwargs, message):
    with pytest.raises(ValueError, match=message):
        AdaptiveRevisionStrategy(**kwargs)
