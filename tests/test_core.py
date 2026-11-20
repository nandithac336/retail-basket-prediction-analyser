import json
from pathlib import Path
from app import core

def test_milestone_1_generates_planted_itemsets(tmp_path):
    result = core.milestone_1(300, seed=5, output_dir=tmp_path)
    assert result["n_baskets"] == 300
    assert result["n_itemsets"] == 32
    assert Path(result["csv_path"]).exists()
    truth = json.loads(Path(result["truth_path"]).read_text())
    assert len(truth) == 32

def test_milestone_2_ngram_predicts_known_transition():
    model = core.NGramModel(max_order=3).fit([
        ["Bread", "Butter", "Milk"], ["Bread", "Butter", "Milk"],
        ["Bread", "Butter", "Eggs"],
    ])
    assert model.predict(["Bread", "Butter"], 2)[0] == "Milk"

def test_milestone_3_neural_model_trains_and_predicts(tmp_path):
    baskets = [["Bread", "Butter", "Milk"], ["Bread", "Butter", "Eggs"]] * 40
    result = core.milestone_3(baskets, epochs=2, output_dir=tmp_path)
    assert result["training_examples"] > 0
    assert len(result["loss_history"]) == 2
    assert (tmp_path / "basket_language_model.npz").exists()

def test_milestone_4_evaluates_both_models():
    baskets = ([["Bread", "Butter", "Milk"], ["Bread", "Butter", "Eggs"],
                ["Chips", "Soda", "Cookies"], ["Chips", "Soda", "Juice"]] * 50)
    ng = core.milestone_2(baskets)
    lm = core.milestone_3(baskets, epochs=2)
    metrics = core.milestone_4(ng, lm, baskets)
    assert metrics["n_test_predictions"] > 0
    assert 0 <= metrics["ngram"]["mrr"] <= 1
    assert 0 <= metrics["neural_lm"]["hits_at_3"] <= 1

def test_milestone_5_writes_report_and_chart(tmp_path):
    metrics = {
        "ngram": {"mrr": .5, "hits_at_1": .3, "hits_at_3": .7, "hits_at_5": .8, "mean_rank": 3.0},
        "neural_lm": {"mrr": .6, "hits_at_1": .4, "hits_at_3": .8, "hits_at_5": .9, "mean_rank": 2.5},
        "n_test_predictions": 100,
    }
    artifacts = core.milestone_5(metrics, [2.0, 1.5, 1.0], tmp_path)
    assert Path(artifacts["metrics_path"]).exists()
    assert Path(artifacts["chart_path"]).exists()
    assert Path(artifacts["report_path"]).exists()
