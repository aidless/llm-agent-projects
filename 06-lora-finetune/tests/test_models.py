"""
模型和配置模块测试
使用 mock 测试 LoRA 配置、模型加载和导出
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestLoRAConfig:
    """测试 LoRA 配置"""

    def test_default_config(self):
        """测试默认配置"""
        from models.lora_config import LoRAConfig

        config = LoRAConfig()
        assert config.r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.05
        assert config.use_qlora is False
        assert "q_proj" in config.target_modules
        assert "v_proj" in config.target_modules
        assert "k_proj" in config.target_modules
        assert "o_proj" in config.target_modules

    def test_qlora_config(self):
        """测试 QLoRA 配置"""
        from models.lora_config import LoRAConfig

        config = LoRAConfig(use_qlora=True, r=16)
        assert config.use_qlora is True
        assert config.r == 16

    def test_to_peft_config(self):
        """测试转换为 PEFT LoraConfig"""
        from models.lora_config import LoRAConfig

        config = LoRAConfig(r=8, lora_alpha=16)
        peft_config = config.to_peft_config()

        assert peft_config.r == 8
        assert peft_config.lora_alpha == 16
        assert peft_config.lora_dropout == 0.05
        assert peft_config.target_modules == {"q_proj", "v_proj", "k_proj", "o_proj"}

    def test_get_bnb_config_qlora(self):
        """测试 QLoRA BitsAndBytes 配置"""
        from models.lora_config import LoRAConfig

        config = LoRAConfig(use_qlora=True)
        bnb_config = config.get_bnb_config()
        assert bnb_config is not None

    def test_get_bnb_config_non_qlora(self):
        """测试非 QLoRA 时 BNB 配置为 None"""
        from models.lora_config import LoRAConfig

        config = LoRAConfig(use_qlora=False)
        bnb_config = config.get_bnb_config()
        assert bnb_config is None

    def test_to_dict(self):
        """测试转换为字典"""
        from models.lora_config import LoRAConfig

        config = LoRAConfig(r=4, lora_alpha=8)
        d = config.to_dict()
        assert d["r"] == 4
        assert d["lora_alpha"] == 8
        assert d["use_qlora"] is False

    def test_from_dict(self):
        """测试从字典创建配置"""
        from models.lora_config import LoRAConfig

        d = {"r": 32, "lora_alpha": 64, "lora_dropout": 0.1, "use_qlora": True}
        config = LoRAConfig.from_dict(d)
        assert config.r == 32
        assert config.lora_alpha == 64
        assert config.lora_dropout == 0.1
        assert config.use_qlora is True

    def test_from_dict_extra_keys_ignored(self):
        """测试从字典创建配置（忽略额外 key）"""
        from models.lora_config import LoRAConfig

        d = {"r": 8, "nonexistent_key": "value"}
        config = LoRAConfig.from_dict(d)
        assert config.r == 8

    def test_get_lora_config_function(self):
        """测试快捷创建函数"""
        from models.lora_config import get_lora_config

        config = get_lora_config(r=16, lora_alpha=32)
        assert config.r == 16
        assert config.lora_alpha == 32

    def test_preset_configs(self):
        """测试预定义配置"""
        from models.lora_config import PRESET_CONFIGS

        assert "small" in PRESET_CONFIGS
        assert "medium" in PRESET_CONFIGS
        assert "large" in PRESET_CONFIGS
        assert "qlora_default" in PRESET_CONFIGS

        assert PRESET_CONFIGS["small"].r == 4
        assert PRESET_CONFIGS["medium"].r == 8
        assert PRESET_CONFIGS["large"].r == 16
        assert PRESET_CONFIGS["qlora_default"].use_qlora is True


# 内存受限的环境标记：涉及 peft 导入的测试在内存受限环境中可能导致 OOM
_skip_heavy_imports = False
try:
    import os
    # 如果设置了此环境变量则跳过重导入测试
    if os.environ.get("SKIP_HEAVY_TESTS", "0") == "1":
        _skip_heavy_imports = True
except Exception:
    pass

requires_heavy_imports = pytest.mark.skipif(
    _skip_heavy_imports,
    reason="跳过：环境变量 SKIP_HEAVY_TESTS=1",
)


class TestModelLoader:
    """测试模型加载器（使用 mock）"""

    def test_loader_init(self):
        """测试加载器初始化"""
        from models.model_loader import ModelLoader

        loader = ModelLoader(
            model_name_or_path="test-model",
            use_gradient_checkpointing=False,
            torch_dtype="bfloat16",
        )
        assert loader.model_name_or_path == "test-model"
        assert loader.use_gradient_checkpointing is False

    @requires_heavy_imports
    @patch("transformers.AutoModelForCausalLM")
    def test_load_base_model(self, mock_auto_model):
        """测试加载基础模型"""
        from models.model_loader import ModelLoader

        # 创建 mock 模型
        mock_model = MagicMock()
        mock_model.num_parameters.return_value = 7000000000
        mock_model.gradient_checkpointing_enable = MagicMock()
        mock_model.enable_input_require_grads = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model

        loader = ModelLoader(
            model_name_or_path="test-model",
            use_gradient_checkpointing=True,
            torch_dtype="auto",
        )

        model = loader.load_base_model()
        assert model is mock_model
        mock_model.gradient_checkpointing_enable.assert_called_once()

    @requires_heavy_imports
    @patch("transformers.AutoModelForCausalLM")
    def test_load_with_lora(self, mock_auto_model):
        """测试加载模型并应用 LoRA"""
        from models.model_loader import ModelLoader
        from models.lora_config import LoRAConfig

        # mock 模型
        mock_model = MagicMock()
        mock_model.num_parameters.return_value = 7000000000
        mock_model.gradient_checkpointing_enable = MagicMock()
        mock_model.enable_input_require_grads = MagicMock()
        mock_model.print_trainable_parameters = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model

        lora_config = LoRAConfig(r=8, lora_alpha=16)
        loader = ModelLoader(
            model_name_or_path="test-model",
            use_gradient_checkpointing=True,
            torch_dtype="auto",
        )

        model = loader.load_base_model(lora_config=lora_config)
        assert model is mock_model

    @requires_heavy_imports
    @patch("transformers.AutoTokenizer")
    def test_load_tokenizer(self, mock_auto_tokenizer):
        """测试加载分词器"""
        from models.model_loader import ModelLoader

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        loader = ModelLoader(model_name_or_path="test-model")
        tokenizer = loader.load_tokenizer()

        assert tokenizer is mock_tokenizer
        assert tokenizer.pad_token == "</s>"


class TestModelExporter:
    """测试模型导出器（使用 mock）"""

    def test_merge_and_save_peft_model(self):
        """测试合并并保存 PEFT 模型"""
        from models.model_exporter import ModelExporter
        import tempfile

        mock_model = MagicMock()
        mock_merged = MagicMock()
        mock_model.merge_and_unload.return_value = mock_merged

        mock_tokenizer = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            output = ModelExporter.merge_and_save(
                model=mock_model,
                output_dir=tmpdir,
                save_tokenizer=True,
                tokenizer=mock_tokenizer,
            )

            mock_model.merge_and_unload.assert_called_once()
            mock_merged.save_pretrained.assert_called_once()
            mock_tokenizer.save_pretrained.assert_called_once()

    def test_merge_and_save_non_peft_model(self):
        """测试保存非 PEFT 模型"""
        from models.model_exporter import ModelExporter
        import tempfile

        mock_model = MagicMock()
        # 没有 merge_and_unload 方法
        del mock_model.merge_and_unload

        mock_tokenizer = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            output = ModelExporter.merge_and_save(
                model=mock_model,
                output_dir=tmpdir,
                save_tokenizer=True,
                tokenizer=mock_tokenizer,
            )

            mock_model.save_pretrained.assert_called_once()

    def test_save_lora_adapter(self):
        """测试保存 LoRA 适配器"""
        from models.model_exporter import ModelExporter
        import tempfile

        mock_model = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = str(Path(tmpdir) / "adapter")
            output = ModelExporter.save_lora_adapter(
                model=mock_model,
                output_dir=adapter_dir,
            )
            mock_model.save_pretrained.assert_called_once()

    def test_save_lora_adapter_non_peft(self):
        """测试保存非 PEFT 模型的适配器"""
        from models.model_exporter import ModelExporter
        import tempfile

        mock_model = MagicMock()
        # 没有 save_pretrained 方法
        del mock_model.save_pretrained

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="不是 PEFT 模型"):
                ModelExporter.save_lora_adapter(
                    model=mock_model,
                    output_dir=tmpdir,
                )
