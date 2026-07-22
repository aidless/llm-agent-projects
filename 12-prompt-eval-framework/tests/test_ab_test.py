"""A/B 测试测试。"""

import pytest

from ab_test.statistics import paired_t_test, bootstrap_confidence_interval, compute_win_rate
from ab_test.runner import ABTestRunner, PromptVersion


# 生成一些简单的测试分数
def _generate_scores(mean: float, std: float, n: int, seed: int = 42):
    import random
    rng = random.Random(seed)
    return [max(0.0, min(1.0, rng.gauss(mean, std))) for _ in range(n)]


class TestStatistics:
    def test_paired_t_test_returns_required_fields(self):
        a = [0.8, 0.7, 0.9, 0.6, 0.85]
        b = [0.6, 0.5, 0.7, 0.4, 0.65]
        result = paired_t_test(a, b)
        assert "t_statistic" in result
        assert "p_value" in result
        assert "significant" in result
        assert "direction" in result

    def test_paired_t_test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="长度"):
            paired_t_test([0.5], [0.5, 0.6])

    def test_paired_t_test_min_samples_raises(self):
        with pytest.raises(ValueError, match="样本数"):
            paired_t_test([0.5], [0.4])

    def test_bootstrap_ci_returns_required_fields(self):
        scores = [0.7, 0.8, 0.6, 0.9, 0.75]
        result = bootstrap_confidence_interval(scores, n_bootstrap=1000)
        assert "mean" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]

    def test_win_rate_sums_to_one(self):
        a = [0.8, 0.6, 0.9, 0.5]
        b = [0.7, 0.7, 0.8, 0.6]
        result = compute_win_rate(a, b)
        total = result["win_rate_a"] + result["win_rate_b"] + result["tie_rate"]
        assert abs(total - 1.0) < 1e-4

    def test_win_rate_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="长度"):
            compute_win_rate([0.5], [0.5, 0.6])


class TestABTestRunner:
    def test_add_and_list_versions(self):
        runner = ABTestRunner()
        v = PromptVersion("v1", "Version 1", "Hello {input}")
        runner.add_version(v)
        versions = runner.list_versions()
        assert len(versions) == 1
        assert versions[0]["version_id"] == "v1"

    def test_remove_version(self):
        runner = ABTestRunner()
        v = PromptVersion("v1", "V1", "prompt")
        runner.add_version(v)
        runner.remove_version("v1")
        assert len(runner.list_versions()) == 0

    def test_remove_nonexistent_raises(self):
        runner = ABTestRunner()
        with pytest.raises(ValueError, match="不存在"):
            runner.remove_version("nonexistent")

    def test_run_ab_test(self):
        runner = ABTestRunner()
        runner.add_version(PromptVersion("v1", "V1", "prompt a"))
        runner.add_version(PromptVersion("v2", "V2", "prompt b"))

        preds_a = ["the cat is on the mat"] * 10
        preds_b = ["a cat sits on a mat"] * 10
        refs = ["the cat sits on the mat"] * 10

        result = runner.run_test("v1", "v2", preds_a, preds_b, refs, metric_names=["f1"])
        assert "statistical_test" in result
        assert "win_rate" in result
        assert "recommendation" in result

    def test_run_ab_test_nonexistent_version_raises(self):
        runner = ABTestRunner()
        with pytest.raises(ValueError, match="不存在"):
            runner.run_test("x", "y", ["a"], ["b"], ["c"])
