"""Malformed durable operator receipts fail through their closed error boundary."""

from pathlib import Path

import pytest

from capicove.batch.job_plan_receipt import decode_plan_receipt
from capicove.batch.job_terminal import decode_terminal_receipt


def test_unencodable_plan_receipt_is_rejected_as_invalid(tmp_path: Path) -> None:
    tmp_path.joinpath("inputs").mkdir()

    with pytest.raises(ValueError, match="invalid plan receipt"):
        decode_plan_receipt("\ud800", "plan:" + "a" * 64, tmp_path / "inputs")


def test_unencodable_terminal_receipt_is_rejected_as_invalid() -> None:
    with pytest.raises(ValueError, match="invalid terminal receipt"):
        decode_terminal_receipt("\ud800")
