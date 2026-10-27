import pandas as pd
import pytest

from assignment.utils import validate_targets


def targets(n=3, **overrides):
    df = pd.DataFrame(
        {
            "patient_token": [f"tok{i}" for i in range(1, n + 1)],
            "program_recommended": ["physical_therapy"] * n,
            "score": [1.0 / i for i in range(1, n + 1)],
        }
    )
    return df.assign(**overrides)


def test_accepts_valid_targets():
    validate_targets(targets())
    validate_targets(targets().assign(notes="extra columns are fine"))


@pytest.mark.parametrize(
    "bad, message",
    [
        (targets().drop(columns=["score"]), "missing columns"),
        (targets().iloc[:0], "empty"),
        (targets(patient_token=["tok1", None, "tok3"]), "not be null"),
        (targets(patient_token=["tok1", "tok1", "tok3"]), "unique"),
        (targets(score=["high", "mid", "low"]), "numeric"),
        (targets(score=[1.0, None, 0.5]), "not be null"),
        (targets(program_recommended=["pt", None, "pt"]), "program_recommended"),
    ],
)
def test_rejects_bad_targets(bad, message):
    with pytest.raises(ValueError, match=message):
        validate_targets(bad)
