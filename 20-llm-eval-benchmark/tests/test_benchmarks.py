"""
tests/test_benchmarks.py - 评测基准测试

测试数据加载、提示构建、答案提取和答案检查。
"""

import pytest
from benchmarks import (
    get_benchmark,
    MMLUBenchmark,
    GSM8KBenchmark,
    HumanEvalBenchmark,
    MTBenchBenchmark,
    CustomBenchmark,
    BenchmarkQuestion,
)


class TestMMLU:
    """MMLU 评测测试."""

    def test_load_data(self):
        """测试 MMLU 数据加载."""
        bench = MMLUBenchmark()
        questions = bench.load_data()
        assert len(questions) == 50
        assert all(isinstance(q, BenchmarkQuestion) for q in questions)

    def test_total_questions(self):
        """测试题目总数."""
        bench = MMLUBenchmark()
        assert bench.total_questions == 50

    def test_build_prompt(self):
        """测试提示构建包含选项."""
        bench = MMLUBenchmark()
        q = bench.questions[0]
        prompt = bench.build_prompt(q)
        assert "A." in prompt
        assert "B." in prompt
        assert "C." in prompt
        assert "D." in prompt
        assert q.question in prompt

    def test_extract_answer_single_letter(self):
        """测试提取单个字母答案."""
        bench = MMLUBenchmark()
        q = bench.questions[0]
        assert bench.extract_answer("A", q) == "A"
        assert bench.extract_answer("B", q) == "B"

    def test_extract_answer_from_sentence(self):
        """测试从句子中提取答案."""
        bench = MMLUBenchmark()
        q = bench.questions[0]
        result = bench.extract_answer("The answer is C because...", q)
        assert result == "C"

    def test_check_answer_correct(self):
        """测试正确答案检查."""
        bench = MMLUBenchmark()
        q = bench.questions[0]
        correct_idx = q.metadata["answer"]
        labels = ["A", "B", "C", "D"]
        correct_label = labels[correct_idx]
        assert bench.check_answer(correct_label, q) is True

    def test_check_answer_incorrect(self):
        """测试错误答案检查."""
        bench = MMLUBenchmark()
        q = bench.questions[0]
        correct_idx = q.metadata["answer"]
        labels = ["A", "B", "C", "D"]
        wrong = [l for i, l in enumerate(labels) if i != correct_idx][0]
        assert bench.check_answer(wrong, q) is False

    def test_get_categories(self):
        """测试获取类别列表."""
        bench = MMLUBenchmark()
        categories = bench.get_categories()
        assert len(categories) > 0
        assert "mathematics" in categories

    def test_get_questions_by_category(self):
        """测试按类别筛选题目."""
        bench = MMLUBenchmark()
        math_qs = bench.get_questions_by_category("mathematics")
        assert len(math_qs) > 0
        assert all(q.metadata["category"] == "mathematics" for q in math_qs)


class TestGSM8K:
    """GSM8K 评测测试."""

    def test_load_data(self):
        """测试 GSM8K 数据加载."""
        bench = GSM8KBenchmark()
        questions = bench.load_data()
        assert len(questions) == 30

    def test_build_prompt(self):
        """测试 GSM8K 提示包含步骤要求."""
        bench = GSM8KBenchmark()
        q = bench.questions[0]
        prompt = bench.build_prompt(q)
        assert "step by step" in prompt.lower()
        assert q.question in prompt

    def test_extract_answer_hash_format(self):
        """测试从 #### 格式提取答案."""
        bench = GSM8KBenchmark()
        q = bench.questions[0]
        result = bench.extract_answer("The answer is 50.\n#### 50", q)
        assert result == 50.0

    def test_extract_answer_last_number(self):
        """测试从响应末尾提取数字."""
        bench = GSM8KBenchmark()
        q = bench.questions[0]
        result = bench.extract_answer("After calculation, the answer is 42.", q)
        assert result == 42.0

    def test_check_answer_correct(self):
        """测试正确数值答案."""
        bench = GSM8KBenchmark()
        q = bench.questions[0]
        assert bench.check_answer(q.metadata["answer"], q) is True

    def test_check_answer_incorrect(self):
        """测试错误数值答案."""
        bench = GSM8KBenchmark()
        q = bench.questions[0]
        assert bench.check_answer(999, q) is False

    def test_check_answer_tolerance(self):
        """测试数值容差."""
        bench = GSM8KBenchmark()
        q = bench.questions[0]
        expected = q.metadata["answer"]
        # 浮点数近似
        assert bench.check_answer(expected + 1e-10, q) is True

    def test_difficulty_distribution(self):
        """测试难度分布统计."""
        bench = GSM8KBenchmark()
        dist = bench.get_difficulty_distribution()
        assert "easy" in dist
        assert "medium" in dist


class TestHumanEval:
    """HumanEval 评测测试."""

    def test_load_data(self):
        """测试 HumanEval 数据加载."""
        bench = HumanEvalBenchmark()
        questions = bench.load_data()
        assert len(questions) == 10

    def test_build_prompt(self):
        """测试 HumanEval 提示."""
        bench = HumanEvalBenchmark()
        q = bench.questions[0]
        prompt = bench.build_prompt(q)
        assert "def " in prompt or "function" in prompt.lower()

    def test_extract_answer_code_block(self):
        """测试从 markdown 代码块提取代码."""
        bench = HumanEvalBenchmark()
        q = bench.questions[0]
        code = "def foo(x): return x"
        response = f"```python\n{code}\n```"
        extracted = bench.extract_answer(response, q)
        assert "def foo" in extracted

    def test_check_answer_correct_function(self):
        """测试正确函数定义检测."""
        bench = HumanEvalBenchmark()
        q = bench.questions[0]
        entry_point = q.metadata["entry_point"]
        code = f"def {entry_point}(numbers, threshold):\n    return True"
        assert bench.check_answer(code, q) is True

    def test_check_answer_wrong_function(self):
        """测试错误函数名检测."""
        bench = HumanEvalBenchmark()
        q = bench.questions[0]
        code = "def wrong_function(x): return 42"
        assert bench.check_answer(code, q) is False


class TestMTBench:
    """MT-Bench 评测测试."""

    def test_load_data(self):
        """测试 MT-Bench 数据加载."""
        bench = MTBenchBenchmark()
        questions = bench.load_data()
        assert len(questions) == 10

    def test_build_prompt_contains_conversation(self):
        """测试提示包含对话内容."""
        bench = MTBenchBenchmark()
        q = bench.questions[0]
        prompt = bench.build_prompt(q)
        assert "User:" in prompt or len(prompt) > 0

    def test_check_answer_non_empty(self):
        """测试非空响应通过检查."""
        bench = MTBenchBenchmark()
        q = bench.questions[0]
        assert bench.check_answer("Some response text", q) is True

    def test_check_answer_empty_fails(self):
        """测试空响应不通过检查."""
        bench = MTBenchBenchmark()
        q = bench.questions[0]
        assert bench.check_answer("", q) is False

    def test_get_categories(self):
        """测试获取对话类别."""
        bench = MTBenchBenchmark()
        categories = bench.get_categories()
        assert len(categories) > 0


class TestCustomBenchmark:
    """自定义基准测试."""

    def test_custom_exact_match(self, tmp_path):
        """测试精确匹配类型."""
        data = {
            "name": "test_bench",
            "questions": [
                {"id": 1, "question": "What is 2+2?", "answer": "4", "type": "exact_match"},
                {"id": 2, "question": "Capital of France?", "answer": "Paris", "type": "exact_match"},
            ],
        }
        data_file = tmp_path / "custom.json"
        import json
        data_file.write_text(json.dumps(data))

        bench = CustomBenchmark(str(data_file))
        questions = bench.load_data()
        assert len(questions) == 2
        assert bench.name == "test_bench"
        assert bench.check_answer("4", questions[0]) is True
        assert bench.check_answer("paris", questions[1]) is True  # case insensitive

    def test_custom_multiple_choice(self, tmp_path):
        """测试多选题类型."""
        data = {
            "name": "mc_bench",
            "questions": [
                {
                    "id": 1,
                    "question": "What color is the sky?",
                    "answer": 0,
                    "type": "multiple_choice",
                    "choices": ["Blue", "Red", "Green", "Yellow"],
                },
            ],
        }
        data_file = tmp_path / "mc.json"
        import json
        data_file.write_text(json.dumps(data))

        bench = CustomBenchmark(str(data_file))
        q = bench.load_data()[0]
        prompt = bench.build_prompt(q)
        assert "A. Blue" in prompt
        assert bench.check_answer("A", q) is True


class TestBenchmarkRegistry:
    """基准注册表测试."""

    def test_get_benchmark_by_name(self):
        """测试通过名称获取基准."""
        bench = get_benchmark("mmlu")
        assert isinstance(bench, MMLUBenchmark)

    def test_get_benchmark_unknown_raises(self):
        """测试获取未知基准抛出异常."""
        with pytest.raises(ValueError, match="Unknown benchmark"):
            get_benchmark("nonexistent")

    def test_all_benchmarks_loadable(self):
        """测试所有注册基准都可以加载."""
        for name in ["mmlu", "gsm8k", "humaneval", "mt_bench"]:
            bench = get_benchmark(name)
            assert bench.total_questions > 0