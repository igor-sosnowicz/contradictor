"""Unit tests for golden_dataset schema models and cell parsers."""

import pytest
from pydantic import ValidationError

from src.data_models.data_models import NLIPrediction, NLIResult
from src.golden_dataset.datasets_handlers.parsers import (
    as_arg_type,
    as_bounds,
    as_nli,
    as_relations,
    gld_dataset_claims,
    gld_dataset_evidence,
    optional_str,
)
from src.golden_dataset.schema_models import (
    ArgumentRelation,
    Evidence,
    SchemaArgument,
    StanceType,
    TextBounds,
)

NAN = float("nan")


class TestTextBoundsIsLocated:
    """TextBounds.is_located accepts only real, non-negative ranges."""

    @pytest.mark.parametrize(
        ("start", "end", "expected"),
        [(0, 5, True), (3, 10, True), (-1, 5, False), (5, 5, False), (6, 2, False)],
    )
    def test_is_located(self, start: int, end: int, *, expected: bool) -> None:
        """Bounds point at text only when end > start >= 0."""
        assert TextBounds(start=start, end=end).is_located is expected


class TestSchemaArgumentValidator:
    """ensure_correct_type gates canonical rows on claim + support."""

    def test_canonical_without_claim_or_support_raises(self) -> None:
        """A canonical row with no claim and no support is rejected."""
        with pytest.raises(ValidationError):
            SchemaArgument(id="a:1", type="canonical", argument="text")

    def test_canonical_with_claim_and_evidence_passes(self) -> None:
        """Claim list (with is_main_claim) plus evidence is valid canonical."""
        row = SchemaArgument(
            id="a:1",
            type="canonical",
            argument="text",
            claim=[{"text": "claim", "is_main_claim": True}],
            evidence=[{"text": "evidence"}],
        )
        assert row.is_valid
        assert row.claim[0].is_main_claim is True

    def test_raw_with_empty_claim_list_passes(self) -> None:
        """Raw rows skip the support requirement entirely."""
        row = SchemaArgument(id="a:1", type="raw", argument="text")
        assert row.claim == []


class TestOptionalStr:
    """optional_str maps None/NaN/blank to None, else strips."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(None, None), (NAN, None), ("", None), ("  ", None), ("x", "x")],
    )
    def test_table(self, value: object, expected: str | None) -> None:
        """Each input maps to its coerced form."""
        assert optional_str(value) == expected


class TestAsArgType:
    """as_arg_type lowercases and strips before validating."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("raw", "raw"), ("canonical", "canonical"), ("RAW ", "raw")],
    )
    def test_valid(self, value: str, expected: str) -> None:
        """Case and surrounding space are normalised away."""
        assert as_arg_type(value) == expected

    def test_invalid_raises(self) -> None:
        """Unknown types raise ValueError."""
        with pytest.raises(ValueError, match="Invalid argument type"):
            as_arg_type("bogus")


class TestAsBounds:
    """as_bounds coerces dicts, keeps instances, nulls garbage."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, None),
            (NAN, None),
            (TextBounds(start=1, end=5), TextBounds(start=1, end=5)),
            ({"start": 1, "end": 5}, TextBounds(start=1, end=5)),
            ({"start": None, "end": 5}, None),
            ("garbage", None),
        ],
    )
    def test_table(self, value: object, expected: TextBounds | None) -> None:
        """Each input maps to its coerced TextBounds or None."""
        assert as_bounds(value) == expected


class TestGldDatasetClaims:
    """gld_dataset_claims handles Igor spans and dumped Claim dicts."""

    def test_single_span_dict_is_wrapped_with_bounds(self) -> None:
        """A bare span dict is wrapped, never iterated as keys."""
        claims = gld_dataset_claims(
            {"kind": "claim", "start": 0, "end": 4, "text": "main"}
        )
        assert len(claims) == 1
        assert claims[0].text == "main"
        assert claims[0].bounds == TextBounds(start=0, end=4)
        assert "kind" not in [claim.text for claim in claims]

    def test_list_of_span_dicts(self) -> None:
        """Each span dict becomes one Claim in order."""
        spans = [
            {"kind": "claim", "start": 0, "end": 4, "text": "a"},
            {"kind": "claim", "start": 5, "end": 9, "text": "b"},
        ]
        assert [claim.text for claim in gld_dataset_claims(spans)] == ["a", "b"]

    def test_dumped_claim_dict_carries_is_main_claim(self) -> None:
        """Parquet-round-tripped Claim dicts keep is_main_claim and bounds."""
        claims = gld_dataset_claims(
            {"text": "c", "is_main_claim": True, "bounds": {"start": 1, "end": 2}}
        )
        assert claims[0].is_main_claim is True
        assert claims[0].bounds == TextBounds(start=1, end=2)

    def test_plain_string_item_without_bounds(self) -> None:
        """A plain string item becomes an unlocated Claim."""
        claims = gld_dataset_claims(["bare text"])
        assert claims[0].text == "bare text"
        assert claims[0].bounds is None

    def test_blank_text_skipped(self) -> None:
        """Blank span texts are dropped."""
        spans = [{"kind": "claim", "start": 0, "end": 4, "text": "  "}]
        assert gld_dataset_claims(spans) == []

    def test_missing_start_and_end_yields_claim_without_bounds(self) -> None:
        """A text-only dict still yields a Claim, with bounds=None."""
        claims = gld_dataset_claims([{"text": "no span"}])
        assert claims[0].bounds is None


class TestGldDatasetEvidence:
    """gld_dataset_evidence coerces span dicts, skipping unusable rows."""

    def test_span_dict_becomes_evidence_with_bounds(self) -> None:
        """A full span dict becomes located Evidence."""
        spans = [{"kind": "evidence", "start": 3, "end": 7, "text": "e"}]
        assert gld_dataset_evidence(spans) == [
            Evidence(text="e", bounds=TextBounds(start=3, end=7))
        ]

    @pytest.mark.parametrize(
        "spans",
        [[{"kind": "evidence", "start": 3, "end": 7, "text": "  "}], [{"text": "e"}]],
    )
    def test_blank_or_unspanned_skipped(self, spans: list[dict]) -> None:
        """Blank texts and missing start/end rows are dropped."""
        assert gld_dataset_evidence(spans) == []


class TestAsRelations:
    """as_relations coerces dict cells and passes models through."""

    def test_dicts_become_models(self) -> None:
        """Target + stance dicts become ArgumentRelation models."""
        relations = as_relations([{"target_argument_id": "x:2", "stance": "rebuts"}])
        assert relations == [
            ArgumentRelation(target_argument_id="x:2", stance=StanceType.REBUTS)
        ]

    def test_missing_target_skipped(self) -> None:
        """Entries without a target are dropped."""
        assert as_relations([{"stance": "rebuts"}]) == []

    def test_nan_returns_empty(self) -> None:
        """A NaN cell means no relations."""
        assert as_relations(NAN) == []

    def test_instance_passthrough(self) -> None:
        """Existing ArgumentRelation instances are kept as-is."""
        relation = ArgumentRelation(target_argument_id="x:1", stance=StanceType.ATTACKS)
        assert as_relations([relation]) == [relation]


class TestAsNli:
    """as_nli builds NLIPrediction or None."""

    @pytest.mark.parametrize("value", [None, NAN])
    def test_absent_returns_none(self, value: object) -> None:
        """None and NaN cells mean no prediction."""
        assert as_nli(value) is None

    def test_dict_becomes_prediction(self) -> None:
        """A raw result/confidence dict is validated into NLIPrediction."""
        assert as_nli({"result": 1, "confidence": 0.9}) == NLIPrediction(
            result=NLIResult.ENTAILMENT, confidence=0.9
        )
