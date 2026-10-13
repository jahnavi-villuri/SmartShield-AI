"""The one-command rebuild plan stays consistent with the code base."""

import importlib.util

import pytest

from scripts import rebuild_all


def test_every_step_points_at_a_real_module():
    for step in rebuild_all.STEPS:
        assert importlib.util.find_spec(step.command[0]) is not None, step.command


def test_training_always_follows_its_data_preparation():
    order = [" ".join(s.command) for s in rebuild_all.STEPS]
    assert order.index("scripts.prepare_sms") < order.index("src.training.train_sms")
    assert order.index("scripts.prepare_email") < order.index("src.training.train_email")
    assert order.index("scripts.prepare_toxicity") < order.index(
        "src.training.train_toxicity --augment"
    )
    assert order.index("scripts.split_phiusiil") < order.index(
        "src.training.train_phishing --with-phiusiil"
    )
    # measure and document only after everything is built
    assert order[-2:] == ["scripts.benchmark_models", "scripts.make_model_cards"]


def test_select_filters_by_module_and_rejects_unknown_ones():
    assert {s.module for s in rebuild_all.select("sms")} == {"sms"}
    with pytest.raises(SystemExit):
        rebuild_all.select("nope")


def test_missing_inputs_are_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(rebuild_all.settings, "DATA_DIR", tmp_path)
    missing = rebuild_all.missing_inputs(rebuild_all.select("sms"))
    assert any("raw/sms_uci" in line for line in missing)
    (tmp_path / "raw" / "sms_uci").mkdir(parents=True)
    (tmp_path / "raw" / "sms_uci" / "f.csv").write_text("x", encoding="utf-8")
    assert not any(
        "sms_uci" in line for line in rebuild_all.missing_inputs(rebuild_all.select("sms"))
    )
