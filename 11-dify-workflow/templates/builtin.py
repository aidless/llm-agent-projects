"""内置工作流模板 - 文本摘要 / 翻译 / 数据分析"""

from typing import Any, Dict, List


# ── 模板 1: 文本摘要 ──────────────────────────────────────

TEXT_SUMMARY_TEMPLATE: Dict[str, Any] = {
    "id": "builtin-text-summary",
    "name": "文本摘要",
    "description": "接收长文本，通过 LLM 生成摘要",
    "tags": ["nlp", "llm", "summary"],
    "nodes": {
        "start": {
            "type": "llm",
            "inputs": {
                "prompt": "请将以下文本总结为 3-5 个要点:\n{{__global__.text}}",
                "system_prompt": "你是一个专业的文本摘要助手，用简洁的语言提取关键信息。",
                "model": "gpt-3.5-turbo",
                "temperature": 0.3,
            },
        },
        "format": {
            "type": "code",
            "inputs": {
                "code": "raw = input.get('text', '')\nresult = raw.strip()",
            },
        },
    },
    "edges": [
        {"source": "start", "target": "format"},
    ],
    "global_inputs": {
        "text": {"type": "string", "description": "需要摘要的原始文本", "required": True},
    },
    "metadata": {
        "category": "nlp",
        "estimated_time": "5-30s",
    },
}

# ── 模板 2: 翻译工作流 ────────────────────────────────────

TRANSLATION_TEMPLATE: Dict[str, Any] = {
    "id": "builtin-translation",
    "name": "多语言翻译",
    "description": "将输入文本翻译为目标语言",
    "tags": ["nlp", "llm", "translation"],
    "nodes": {
        "detect": {
            "type": "llm",
            "inputs": {
                "prompt": "检测以下文本的语言，返回语言代码 (如 zh, en, ja, ko, fr):\n{{__global__.text}}",
                "system_prompt": "只返回语言代码，不要其他内容。",
                "temperature": 0.0,
            },
        },
        "translate": {
            "type": "llm",
            "inputs": {
                "prompt": "将以下文本翻译为{{__global__.target_language}}:\n{{__global__.text}}",
                "system_prompt": "你是专业翻译，保持原文的语义和风格。",
                "model": "gpt-3.5-turbo",
                "temperature": 0.2,
            },
        },
        "quality_check": {
            "type": "condition",
            "inputs": {
                "conditions": [
                    {
                        "left": "{{translate.output.text}}",
                        "op": "is_not_empty",
                        "right": "",
                        "branch": "valid",
                    },
                ],
                "default_branch": "empty_result",
            },
        },
    },
    "edges": [
        {"source": "detect", "target": "translate"},
        {"source": "translate", "target": "quality_check"},
    ],
    "global_inputs": {
        "text": {"type": "string", "description": "需要翻译的文本", "required": True},
        "target_language": {"type": "string", "description": "目标语言 (如 English, Japanese)", "default": "English"},
    },
    "metadata": {
        "category": "nlp",
        "estimated_time": "5-20s",
    },
}

# ── 模板 3: 数据分析 ──────────────────────────────────────

DATA_ANALYSIS_TEMPLATE: Dict[str, Any] = {
    "id": "builtin-data-analysis",
    "name": "数据分析报告",
    "description": "接收数据并生成分析报告",
    "tags": ["data", "code", "llm"],
    "nodes": {
        "preprocess": {
            "type": "code",
            "inputs": {
                "code": (
                    "data = input.get('data', [])\n"
                    "result = {\n"
                    "    'count': len(data),\n"
                    "    'keys': list(data[0].keys()) if data else [],\n"
                    "    'sample': data[:3] if data else [],\n"
                    "}\n"
                ),
            },
        },
        "analyze": {
            "type": "code",
            "inputs": {
                "code": (
                    "stats = input.get('input', {})\n"
                    "result = {\n"
                    "    'total_records': stats.get('count', 0),\n"
                    "    'fields': stats.get('keys', []),\n"
                    "    'summary': f'Total {stats.get(\"count\", 0)} records with {len(stats.get(\"keys\", []))} fields',\n"
                    "}\n"
                ),
            },
        },
        "report": {
            "type": "llm",
            "inputs": {
                "prompt": (
                    "基于以下数据分析结果生成简洁报告:\n"
                    "{{analyze.output}}"
                ),
                "system_prompt": "你是数据分析师，用结构化的方式呈现关键发现。",
                "temperature": 0.3,
            },
        },
    },
    "edges": [
        {"source": "preprocess", "target": "analyze"},
        {"source": "analyze", "target": "report"},
    ],
    "global_inputs": {
        "data": {
            "type": "array",
            "description": "待分析的 JSON 数据数组",
            "required": True,
        },
    },
    "metadata": {
        "category": "data",
        "estimated_time": "5-15s",
    },
}

# ── 模板注册表 ────────────────────────────────────────────

BUILTIN_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "text-summary": TEXT_SUMMARY_TEMPLATE,
    "translation": TRANSLATION_TEMPLATE,
    "data-analysis": DATA_ANALYSIS_TEMPLATE,
}


def get_template(template_id: str) -> Dict[str, Any]:
    """获取内置模板"""
    tpl = BUILTIN_TEMPLATES.get(template_id)
    if tpl is None:
        raise ValueError(f"模板 '{template_id}' 不存在。可用: {list(BUILTIN_TEMPLATES.keys())}")
    return tpl.copy()


def list_templates() -> List[Dict[str, Any]]:
    """列出所有内置模板的摘要信息"""
    return [
        {
            "id": tpl["id"],
            "name": tpl["name"],
            "description": tpl["description"],
            "tags": tpl.get("tags", []),
            "node_count": len(tpl.get("nodes", {})),
            "edge_count": len(tpl.get("edges", [])),
        }
        for tpl in BUILTIN_TEMPLATES.values()
    ]
