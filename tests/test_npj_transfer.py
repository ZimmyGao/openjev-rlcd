"""Transfer tasks of the npj protocol (Q3b): ARC-Challenge for the MMLU-Pro models, SVAMP-Verify for the GSM8K-Verify
models, and train_direct --eval-only for the direct-answer weights."""
import json
import sys
from pathlib import Path

import pytest
import torch

from openjev_rlcd import train, train_direct
from openjev_rlcd.tasks import TASKS, arc, svamp_verify

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_baselines import _tiny  # noqa: E402


def test_arc_items(monkeypatch):
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    try:
        s = arc.load_items()
    except Exception as e:  # pragma: no cover - only without the cached dataset
        pytest.skip(f"ARC not cached: {e}")
    assert len(s["dev"]) == 299 and len(s["test"]) == 1172 and s["train"] == []
    for it in s["dev"] + s["test"]:
        assert 3 <= len(it["options"]) <= 5 and sum(it["q"]) == 1.0 and len(it["q"]) == len(it["options"])
    assert TASKS["arc"]["k"](s["test"][0]) == len(s["test"][0]["options"])
    assert "Options:\nA. " in TASKS["arc"]["message"](s["test"][0])


def test_svamp_verify_items():
    if not svamp_verify.DATA.exists():
        pytest.skip("data/svamp_verify.json not built yet")
    s = svamp_verify.load_items()
    assert len(s["dev"]) == 200 and len(s["test"]) > 700 and s["train"] == []
    assert len({it["uid"] for it in s["dev"] + s["test"]}) == len(s["dev"]) + len(s["test"])


def test_direct_eval_only_on_a_transfer_task(monkeypatch, tmp_path):
    tok, model = _tiny(monkeypatch)
    monkeypatch.setattr(train_direct, "DEVICE", "cpu")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    try:
        arc.load_items()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"ARC not cached: {e}")
    tok.save_pretrained(tmp_path / "m")
    model.save_pretrained(tmp_path / "m")
    ck = tmp_path / "w.pt"
    train.save_weights(model, ck)
    out = tmp_path / "ev.json"
    monkeypatch.setattr(sys, "argv", ["train_direct", "--task", "arc", "--model", str(tmp_path / "m"), "--labels",
                                      "fresh", "--eval-only", "--checkpoint", str(ck), "--dev-cap", "3", "--test-cap", "4",
                                      "--output", str(out)])
    train_direct.main()
    d = json.loads(out.read_text())
    assert len(d["test_logps"]) == 4 and len(d["dev_logps"]) == 3 and d["config"]["eval_only"] is True
    assert '"test": {' in out.read_text() and d["curve"] == []


def test_direct_eval_only_reproduces_the_training_runs_final_readouts(monkeypatch, tmp_path):
    """with weights that bf16 represents exactly, the eval-only pass of the saved checkpoint is the training run's own
    final forward pass (fp32 model under bf16 autocast), so the source temperature applies to it unchanged"""
    tok, model = _tiny(monkeypatch)
    monkeypatch.setattr(train_direct, "DEVICE", "cpu")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(p.bfloat16().float())
    tok.save_pretrained(tmp_path / "m")
    model.save_pretrained(tmp_path / "m")
    common = ["--task", "gsm8kv", "--model", str(tmp_path / "m"), "--labels", "fresh", "--dev-cap", "3",
              "--test-cap", "4"]
    monkeypatch.setattr(sys, "argv", ["train_direct"] + common + ["--steps", "0", "--save-checkpoint",
                                                                  str(tmp_path / "w.pt"), "--output", str(tmp_path / "a.json")])
    train_direct.main()
    monkeypatch.setattr(sys, "argv", ["train_direct"] + common + ["--eval-only", "--checkpoint", str(tmp_path / "w.pt"),
                                                                  "--output", str(tmp_path / "b.json")])
    train_direct.main()
    a, b = (json.loads((tmp_path / f).read_text()) for f in ("a.json", "b.json"))
    assert a["dev_logps"] == b["dev_logps"] and a["test_logps"] == b["test_logps"] and a["temperature"] == b["temperature"]


def test_svamp_build_runs_once(monkeypatch, tmp_path):
    data = tmp_path / "svamp_verify.json"
    data.write_text("{}")
    monkeypatch.setattr(svamp_verify, "DATA", data)
    with pytest.raises(SystemExit):  # refuses before loading any model or dataset
        svamp_verify.build()
    rows = [dict(uid=f"svamp-test-{i}-p{i % 7}", question=f"q{i}", gold="3", proposed=str(i % 7), correct=i % 7 == 3)
            for i in range(300)]
    data.write_text(json.dumps(dict(solver="x", seed=0, items=rows)))
    s = svamp_verify.load_items()
    assert len(s["dev"]) == 200 and len(s["test"]) == 100 and s["train"] == []
    for it in s["dev"] + s["test"]:
        assert it["uid"].endswith(f"-p{it['proposed']}") and it["q"] == ([1.0, 0.0] if it["proposed"] == "3" else [0.0, 1.0])


def test_svamp_summary_of_an_interrupted_build(monkeypatch, tmp_path):
    """the data file was renamed into place but the summary was not written: a rerun of the job writes the summary of
    the existing file instead of refusing forever (and never rebuilds); a later change of the data file is detected"""
    data, out = tmp_path / "svamp_verify.json", tmp_path / "svamp_build.json"
    rows = [dict(uid=f"svamp-test-{i}-p{i % 7}", question=f"q{i}", gold="3", proposed=str(i % 7), correct=i % 7 == 3)
            for i in range(300)]
    data.write_text(json.dumps(dict(solver="x", seed=0, items=rows)))
    monkeypatch.setattr(svamp_verify, "DATA", data)
    monkeypatch.setattr(svamp_verify, "build", lambda **kw: pytest.fail("must not rebuild"))
    assert not svamp_verify.built(out)
    svamp_verify.main(["--build", "--output", str(out)])
    summary = json.loads(out.read_text())["test"]
    assert summary["n"] == 300 and summary["n_dev"] == 200 and '"test": {' in out.read_text() and svamp_verify.built(out)
    data.write_text(data.read_text().replace('"q0"', '"q0 "'))  # another data file under the same summary
    assert not svamp_verify.built(out)


def test_transfer_and_chaosnli_split_hashes_match_the_loaders(monkeypatch):
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    ref = json.loads((Path(train.__file__).resolve().parents[1] / "data" / "npj_splits.json").read_text())
    for task in ("arc", "chaosnli"):
        try:
            sp = TASKS[task]["load"]()
        except Exception as e:  # pragma: no cover
            pytest.skip(f"{task} not cached: {e}")
        assert ref[task]
        train.check_splits(task, sp, [(n, int(k)) for n, d in ref[task].items() for k in d])


def test_transfer_plans(monkeypatch):
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import plan_npj
    monkeypatch.setattr(plan_npj, "svamp_built", lambda: False)
    with pytest.raises(SystemExit):  # the SVAMP-Verify jobs are planned only after the build
        plan_npj.transfer_jobs("gsm8kv")
    jobs = plan_npj.transfer_jobs("mmlupro")
    assert jobs[0].module == "tasks.svamp_verify" and all(j.name.startswith("arc_") for j in jobs[1:])
    assert len(jobs) == 1 + 2 * (1 + 3 * 5)
    assert not plan_npj.check(jobs)  # every job evaluates its source run's weights with the source's settings
    monkeypatch.setattr(plan_npj, "svamp_built", lambda: True)
    sv = plan_npj.transfer_jobs("gsm8kv")
    assert all(j.name.startswith("svampv_") and j.needs != plan_npj.SVAMP_BUILD + ".json" for j in sv)
    bad = [j for j in jobs if j.name == "arc_warm400_seed23"][0]
    bad.args = bad.args.replace("--m-eval 4", "--m-eval 2")
    assert plan_npj.check([bad])
    swapped = [j for j in jobs if j.name == "arc_ce_seed17"][0]  # a job named after one arm evaluating another's weights
    swapped.needs = swapped.needs.replace("_ce_", "_warm400_")
    swapped.args = swapped.args.replace("mp_ce_seed17", "mp_warm400_seed17")
    assert plan_npj.check([swapped])
    capped = [j for j in jobs if j.name == "arc_grpo_seed29"][0]
    capped.args = capped.args.replace("--test-cap 1172", "--test-cap 500")
    assert plan_npj.check([capped])


def test_chaos_plan_matches_the_earlier_chaosnli_runs(monkeypatch):
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import plan_npj
    if not (plan_npj.REFROOT / "results/teacher3_single_seed17.json").exists():
        pytest.skip("the earlier ChaosNLI run files are not here")
    jobs = plan_npj.chaos_jobs()
    assert len(jobs) == 6 and not plan_npj.check(jobs)
    jobs[0].args = jobs[0].args.replace("--max-new-tokens 128", "--max-new-tokens 256")
    assert plan_npj.check(jobs[:1])
