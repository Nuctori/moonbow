import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("polarity", ROOT / "tools/intent_eval/build_polarity_diagnostic.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_polarity_both_actions_and_directions():
    datasets = module.build_rows()
    for rows in datasets.values():
        assert len(rows) == 36
        assert len({r["case_id"] for r in rows}) == 36
        for group in {r["group"] for r in rows}:
            subset = [r for r in rows if r["group"] == group]
            labels = {(r["variant"], r["position"]): r["label"] for r in subset}
            assert labels["ban", "yes"] == "violation"
            assert labels["ban", "no"] == "compliant"
            assert labels["require", "yes"] == "compliant"
            assert labels["require", "no"] == "violation"
            assert labels["ban-paraphrase", "yes"] == labels["ban", "yes"]
            assert labels["ban-paraphrase", "no"] == labels["ban", "no"]


def test_polarity_scenarios_disjoint():
    datasets = module.build_rows()
    assert {r["group"] for r in datasets["train"]}.isdisjoint(
        {r["group"] for r in datasets["eval"]})
    assert {r["event_text"] for r in datasets["train"]}.isdisjoint(
        {r["event_text"] for r in datasets["eval"]})
