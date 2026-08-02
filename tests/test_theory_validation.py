"""Tests for the theory validation framework (theory_validation/).

Covers:
  - Registry integrity (unique ids, required fields populated)
  - Every code_ref actually resolves against the live codebase (this is
    the test that keeps the claims registry honest as code changes --
    a claim referencing renamed/removed code should fail CI)
  - CLI `report` and `check` commands run and exit correctly
  - EvidenceStatus / ExternalValidation ordinal ranking behaves as expected
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from theory_validation.claims import (
    CLAIMS,
    Claim,
    EvidenceStatus,
    ExternalValidation,
    get_claim,
    resolve_code_ref,
)


class TestRegistryIntegrity:
    """The claims registry itself must be well-formed."""

    def test_at_least_one_claim(self) -> None:
        assert len(CLAIMS) > 0

    def test_ids_are_unique(self) -> None:
        ids = [c.id for c in CLAIMS]
        assert len(ids) == len(set(ids)), f"Duplicate claim ids: {ids}"

    def test_ids_are_snake_case_slugs(self) -> None:
        for c in CLAIMS:
            assert c.id.islower()
            assert " " not in c.id

    @pytest.mark.parametrize("claim", CLAIMS, ids=[c.id for c in CLAIMS])
    def test_required_fields_populated(self, claim: Claim) -> None:
        assert claim.name.strip()
        assert claim.claim.strip()
        assert claim.mechanism.strip()
        assert len(claim.code_refs) > 0, f"{claim.id} has no code_refs"

    @pytest.mark.parametrize("claim", CLAIMS, ids=[c.id for c in CLAIMS])
    def test_evidence_status_is_valid_enum_member(self, claim: Claim) -> None:
        assert isinstance(claim.evidence_status, EvidenceStatus)

    @pytest.mark.parametrize("claim", CLAIMS, ids=[c.id for c in CLAIMS])
    def test_external_validation_is_valid_enum_member(self, claim: Claim) -> None:
        assert isinstance(claim.external_validation, ExternalValidation)

    @pytest.mark.parametrize("claim", CLAIMS, ids=[c.id for c in CLAIMS])
    def test_untested_claims_have_no_test_refs(self, claim: Claim) -> None:
        """A claim marked UNTESTED shouldn't simultaneously list test_refs
        (that would be a contradiction worth catching)."""
        if claim.evidence_status == EvidenceStatus.UNTESTED:
            assert claim.test_refs == [] or all(
                "routing only" in t or "not target" in t or "does not" in t for t in claim.test_refs
            ), (
                f"{claim.id} is UNTESTED but lists test_refs without an "
                "explanatory caveat -- either upgrade evidence_status or "
                "clarify why the listed test doesn't count."
            )

    def test_get_claim_found(self) -> None:
        c = get_claim(CLAIMS[0].id)
        assert c is not None
        assert c.id == CLAIMS[0].id

    def test_get_claim_not_found(self) -> None:
        assert get_claim("does_not_exist_xyz") is None


class TestCodeReferencesResolve:
    """Every code_ref must point at real, currently-existing code.

    This is the key regression test: if a claim references
    `cognition:CausalDiscovery` and that class is later renamed or removed,
    this test fails -- forcing the registry to be updated rather than
    silently going stale.
    """

    @pytest.mark.parametrize(
        "ref",
        sorted({ref for c in CLAIMS for ref in c.code_refs}),
    )
    def test_ref_resolves(self, ref: str) -> None:
        assert resolve_code_ref(ref), f"Stale code reference: {ref}"

    def test_resolve_code_ref_rejects_bogus_module(self) -> None:
        assert resolve_code_ref("this_module_does_not_exist_xyz") is False

    def test_resolve_code_ref_rejects_bogus_attr(self) -> None:
        assert resolve_code_ref("cognition:ThisClassDoesNotExistXYZ") is False

    def test_resolve_code_ref_accepts_whole_module(self) -> None:
        assert resolve_code_ref("cognition") is True


class TestEvidenceStatusOrdering:
    """The evidence ladder must be ordered weakest-to-strongest."""

    def test_ranks_are_monotonic(self) -> None:
        ordered = list(EvidenceStatus)
        ranks = [level.rank for level in ordered]
        assert ranks == sorted(ranks)

    def test_untested_is_weakest(self) -> None:
        assert EvidenceStatus.UNTESTED.rank < EvidenceStatus.EXTERNALLY_VALIDATED.rank

    def test_external_validation_ranks_monotonic(self) -> None:
        ordered = list(ExternalValidation)
        ranks = [level.rank for level in ordered]
        assert ranks == sorted(ranks)


class TestCLI:
    """The CLI should run cleanly against the current registry."""

    def test_report_table_exits_zero(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "theory_validation", "report"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Claim" in result.stdout
        assert str(len(CLAIMS)) in result.stdout

    def test_report_json_is_valid_and_complete(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "theory_validation", "report", "--format", "json"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert len(payload) == len(CLAIMS)
        assert {c["id"] for c in payload} == {c.id for c in CLAIMS}

    def test_check_exits_zero_when_all_refs_resolve(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "theory_validation", "check"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "OK" in result.stdout
