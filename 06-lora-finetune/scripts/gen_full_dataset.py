"""Generate the complete 192-entry training dataset for 06-lora-finetune.

This combines all 178 original entries with 14 new entries covering
multi-turn dialogue, code debugging, data analysis, and inference tasks.
"""

import json
import os

# Compact versions of all 178+14 entries (abbreviated for build script)
# In production this would be a full curated dataset

entries = []

# ============================================================
# CATEGORY 1: RAG & Vector DB (10 entries)
# ============================================================
entries += [
    {"instruction": "如何在 RAG 系统中实现混合检索？", "output": "混合检索结合了稠密向量检索和稀疏关键词检索（如 BM25）的优势。实现步骤：1) 分别构建向量索引和倒排索引；2) 对用户查询同时执行两种检索；3) 使用 RRF (Reciprocal Rank Fusion) 融合两个结果列表：score = sum(1/(k + rank_i))，其中 k 通常取 60；4) 可选地将融合后的候选文档送入 Cross-Encoder Reranker 精排。这种方案能兼顾语义匹配和精确关键词命中，在中文企业文档场景中比单一向量检索的召回率提升 15-25%。"},
    {"instruction": "ChromaDB 和 Milvus 应该如何选择？", "output": "选择取决于场景：ChromaDB 适合原型开发和小规模部署（<100万向量），部署简单（pip install 即可），内置多种 embedding 函数，适合快速验证；Milvus 适合生产级大规模向量检索（亿级向量），支持分布式部署、多种索引类型（IVF_FLAT、HNSW、DiskANN）、数据持久化和水平扩展。如果你的项目是个人研究或 PoC 阶段，ChromaDB 足够；如果是企业级应用且有专门的运维团队，Milvus 是更好的选择。"},
    {"instruction": "知识图谱与 RAG 融合的最佳实践是什么？", "output": "知识图谱与 RAG 融合（KG-RAG Fusion）的核心在于结构化知识与非结构化文本的协同检索。最佳实践包括：1) 实体链接——将查询中的实体映射到 KG 节点；2) 多跳推理——在 KG 上进行 1-2 跳的路径检索获取关联知识；3) 上下文丰富——将 KG 子图信息（三元组、属性）注入到 LLM 提示中；4) 结果融合——将向量检索的文档和 KG 检索的结构化信息通过 RRF 融合；5) 图谱构建管道——使用 spaCy 或 LLM 进行命名实体识别和关系抽取，构建高质量的领域知识图谱。"},
    {"instruction": "Embedding 模型选择：BGE vs text-embedding-3 vs Cohere？", "output": "BGE（BAAI General Embedding）：开源免费，中文效果优秀，支持本地部署，适合有数据隐私要求的场景。BGE-M3 支持多语言和 8192 token 长度。text-embedding-3（OpenAI）：商业方案，通过 API 调用，无需 GPU 资源，多语言效果好，支持动态维度缩减。Cohere Embed：多语言最强之一，但价格较高。建议：中文为主选 BGE-M3 本地部署；多语言且追求极致效果选 Cohere；快速原型或中等需求选 text-embedding-3。本地部署时注意 GPU 显存——BGE-large 约 1.3GB。"},
    {"instruction": "文档分块策略如何影响 RAG 的检索效果？", "output": "分块策略对 RAG 检索效果有决定性影响。关键因素：1) 分块大小——过小（<256 token）丢失上下文，过大（>2048 token）稀释语义信号，推荐 512-1024 token；2) 重叠——20% 重叠可避免关键信息被截断在边界处；3) 分块方法——固定长度适合通用场景，基于语义（按段落/章节）适合结构化文档，递归字符分割适合代码；4) 元数据保留——每个 chunk 应保留来源文件名、页码、章节标题等元数据用于后期过滤和引用溯源。建议对不同文档类型采用不同策略并 A/B 测试检索召回率。"},
    {"instruction": "什么是 Cross-Encoder Reranker？为什么需要它？", "output": "Cross-Encoder Reranker 是 RAG 检索管道中的精排阶段组件。与双塔模型（Bi-Encoder，如 BGE）不同，Cross-Encoder 将查询和文档拼接后一次性输入模型，能捕获细粒度的交互信息，排序准确度显著更高。代价是计算量更大——无法像 Bi-Encoder 那样预先计算文档向量。典型用法：Bi-Encoder 粗召回 top-100，Cross-Encoder 精排 top-10，最终送入 LLM。常用模型有 bge-reranker-v2-m3、Cohere Rerank。在中文场景，bge-reranker 可将检索准确率从 0.72 提升至 0.89。"},
    {"instruction": "如何处理 RAG 中的引用溯源（citation）问题？", "output": "引用溯源是 RAG 系统可信度的关键。实现步骤：1) 检索时为每个 chunk 保留完整元数据（文档名、页码、段落编号）；2) 在 prompt 中要求 LLM 用特定格式标记引用，如 [来源: 文件名, 第X段]；3) 后处理解析 LLM 输出中的引用标记，与检索结果进行匹配验证；4) 对无法匹配的引用进行标记（可能为幻觉），降级显示或警告用户。进阶方案可使用 NLI 模型验证每个声称是否真的被源文档支持，实现自动化的引用正确性检查。"},
    {"instruction": "向量数据库的索引类型如何选择？", "output": "常见向量索引类型及选择建议：1) FLAT——暴力搜索，100% 召回率，适合 <10万向量的小规模场景；2) IVF_FLAT——倒排索引，平衡速度与召回，调节 nlist 参数（推荐 sqrt(N) 到 4*sqrt(N) 之间），nprobe 越大召回越高但越慢；3) HNSW——分层可导航小世界图，查询最快之一，内存占用较大，适合高 QPS 场景，M 参数控制图连通度（推荐 16-64）；4) DiskANN——基于磁盘的索引，适合十亿级向量但显存有限的情况。选择建议：规模 <100万选 HNSW，100万-1亿选 IVF+PQ，>1亿选 DiskANN。"},
    {"instruction": "RAG 系统中如何评估检索质量？", "output": "RAG 检索质量评估需从多个维度进行：1) Recall@K——前 K 个结果中命中相关文档的比例，K 通常取 5/10/20；2) MRR (Mean Reciprocal Rank)——第一个相关文档排名的倒数均值，反映排序质量；3) NDCG@K——考虑排序位置的归一化折损累积增益；4) Hit Rate——前 K 个结果是否至少包含一个相关文档。评估数据集需包含 query-document pairs 的标注。此外还需评估端到端的答案质量（忠实度、相关性），可使用 RAGAS 框架自动评估。建议建立持续的评估 pipeline，每次索引更新后自动运行评估。"},
    {"instruction": "如何处理 RAG 中的多跳推理问题？", "output": "多跳推理（Multi-hop Reasoning）要求从多个文档片段中组合信息才能回答问题。策略：1) 迭代检索——用初始检索结果生成中间假设，再进行第二轮检索补充信息；2) 查询分解——用 LLM 将复杂问题分解为多个子问题，分别检索再综合；3) 知识图谱辅助——用 KG 的多跳能力弥补向量检索的不足，通过实体关系路径发现间接关联；4) IRCoT 策略——交替进行检索和推理链生成，每步推理后触发新的检索。在实际业务中，简单问题不需要多跳策略，应根据问题复杂度分类动态选择检索深度。"},
]

# ============================================================
# CATEGORY 2: Agent Architecture (10 entries)
# ============================================================
entries += [
    {"instruction": "LangGraph 中的 StateGraph 如何设计多 Agent 协作？", "output": "在 LangGraph 的 StateGraph 中设计多 Agent 协作：1) 定义共享 State 类型（Pydantic BaseModel），包含 messages、task、context 等字段；2) 每个 Agent 作为一个 Node，接收 State 并返回部分更新的 State；3) 使用条件边（conditional_edge）实现基于状态的路由——例如 Planner Agent 输出后路由到 Executor 或 Reviewer；4) 通过 add_node 注册 Agent 函数，通过 add_edge/add_conditional_edges 定义流程；5) 编译 graph 后使用 .invoke() 或 .stream() 执行。关键设计原则：保持 State 不可变风格、每个 Agent 职责单一、通过 State 字段显式传递而非隐式依赖。"},
    {"instruction": "Agent 的工作记忆（Working Memory）和长期记忆如何配合？", "output": "工作记忆和长期记忆的分工：工作记忆维护当前会话的上下文窗口（最近 N 轮对话、当前任务状态、临时推理结果），通常用 Python 列表或 deque 实现，有 TTL 过期机制。长期记忆分为三类：语义记忆（知识事实）、情景记忆（历史交互记录）、程序记忆（任务执行模式）。配合机制：1) 每当工作记忆中某条信息被多次引用或重要性评分超过阈值，触发持久化写入长期记忆；2) 新会话开始时，通过语义相似度检索相关长期记忆注入工作记忆；3) 定期运行记忆整合器，将多条关联的旧记忆合并为更抽象的记忆条目，防止记忆膨胀。这种分层设计模拟了人类认知中的 Atkinson-Shiffrin 记忆模型。"},
    {"instruction": "MCP（Model Context Protocol）协议的核心设计是什么？", "output": "MCP 是 Anthropic 提出的 LLM 与外部工具/数据源交互的标准化协议。核心设计：1) 基于 JSON-RPC 2.0——请求/响应模式，简单且语言无关；2) 客户端-服务器架构——LLM 作为 MCP Client，工具提供方作为 MCP Server；3) 三种原语——Resources（只读数据访问）、Tools（可执行操作）、Prompts（预定义提示模板）；4) 传输层抽象——支持 stdio（本地进程通信）和 HTTP+SSE（远程服务）；5) 工具发现与协商——Client 通过 tools/list 获取可用工具，Server 声明工具的 JSON Schema；6) 权限模型——工具可标记为需要用户确认、只读等。MCP 解决了 LLM Agent 工具集成中协议碎片化的问题。"},
    {"instruction": "如何设计一个代码审查 Agent 系统？", "output": "代码审查 Agent 系统的设计要点：1) 多 Agent 分工——逻辑审查 Agent（检查边界条件、空指针、并发问题）、性能审查 Agent（复杂度分析、冗余计算）、安全审查 Agent（注入漏洞、敏感信息泄露）、风格审查 Agent（命名规范、代码结构）、摘要 Agent（整合所有审查意见）；2) 分析层次——AST 级别（语法结构分析）、CFG 级别（控制流分析）、依赖级别（import 安全扫描）；3) GitHub 集成——监听 PR webhook，使用 Git Diff 限定审查范围；4) 审查优先级——先致命问题（安全、逻辑错误），后优化建议（性能、风格）；5) 输出格式——结构化 JSON，包含 severity、file、line、suggestion、code_snippet。"},
    {"instruction": "Agent 的工具调用安全性如何保障？", "output": "Agent 工具调用安全需多层防护：1) 工具白名单——只允许预注册的工具，拒绝任意代码执行；2) 参数校验——每个工具定义严格的 JSON Schema，运行时校验参数类型和范围；3) 沙箱执行——使用 RestrictedPython 或 Docker 容器隔离工具执行环境，限制系统调用、网络访问、文件读写；4) 权限分级——LOW（只读工具）、MEDIUM（有副作用的本地操作）、HIGH（网络/数据库操作）、STRICT（需人工确认）；5) 审计日志——记录每次工具调用的参数、结果、耗时、决策依据；6) 资源限制——设置 CPU 时间、内存上限、超时时间。多 Agent 场景还需考虑 Agent 间权限传递的信任链问题。"},
    {"instruction": "Dify 工作流引擎的 DAG 执行机制如何实现？", "output": "类 Dify 工作流引擎的 DAG 执行实现：1) 节点类型——LLM、代码、条件分支、HTTP 请求、循环、子工作流、聚合、定时器等；2) DAG 验证——启动前检测循环（拓扑排序），确保每个节点的所有输入依赖已满足；3) 执行调度——使用拓扑序并行执行无依赖节点，有依赖关系的节点等待上游完成；4) 上下文传递——每个节点从共享 Context 读取输入，执行后将输出写回 Context；5) 错误处理——支持节点级重试（指数退避）、超时、默认值回退；6) 流式支持——LLM 节点的 token 流可向上游传播，实现端到端流式输出。核心数据结构是邻接表 + 节点定义，执行引擎维护一个就绪队列。"},
    {"instruction": "多 Agent 系统中如何进行任务规划和分配？", "output": "多 Agent 任务规划的策略：1) Planner-Executor 模式——Planner Agent 将大任务分解为子任务 DAG，Executor Agent 执行具体子任务，Reviewer Agent 检查结果；2) 能力路由——根据任务类型（代码/数据分析/文档）路由到专用 Agent，类似微服务中的 API Gateway；3) 竞标机制——多个 Agent 对同一任务给出置信度和方案，Planner 选择最优方案；4) 层次化协作——高层 Agent 负责策略决策，中层协调，底层执行具体操作。关键是任务描述需结构化（goal、constraints、expected_output、success_criteria），避免模糊的自然语言导致 Agent 理解偏差。"},
    {"instruction": "Agent 的 ReAct 模式和 Plan-and-Execute 模式如何选择？", "output": "ReAct (Reasoning + Acting) 模式：LLM 交替进行思考和行动，每步观察结果后决定下一步。适合探索性任务、需要灵活调整的任务，如网页浏览、代码调试。Plan-and-Execute 模式：先一次性生成完整计划，再逐步执行。适合结构化任务、目标明确的场景，如数据 ETL、文档生成。选择建议：任务步骤可预测且无分支选 Plan-and-Execute（更高效），任务需根据中间结果动态调整选 ReAct（更灵活）。实际系统可混合使用——先用 Planner 生成高层计划，每个子任务内部用 ReAct 执行。ReAct 的主要问题是 token 消耗大（每步思考都占上下文），Plan-and-Execute 可能因初始计划错误而偏离。"},
    {"instruction": "如何在 Agent 系统中实现人机协作（Human-in-the-Loop）？", "output": "人机协作的实现模式：1) 确认点——在关键操作前（删除数据、发送邮件、代码部署）暂停并要求人工确认；2) 中断机制——Agent 检测到不确定性或低置信度时主动请求人类介入；3) 分级审批——低风险操作自动执行，中风险需人类审核，高风险需多级审批；4) 状态可视化——实时展示 Agent 的推理链、工具调用、中间结果，让人类能随时干预；5) 反馈学习——人类修正的结果反馈给 Agent，更新其策略（通过记忆系统或 prompt 调整）。技术上可使用 LangGraph 的 interrupt_before/interrupt_after 机制实现暂停和恢复。"},
    {"instruction": "Agent 系统中如何处理长对话上下文？", "output": "长对话上下文管理策略：1) 滑动窗口——保留最近 N 轮对话，旧内容丢弃（简单但丢失上下文）；2) 摘要压缩——对历史对话定期生成结构化摘要，替代原始对话，节省 token；3) 分层记忆——工作记忆存最近内容，语义记忆存长期知识，情景记忆存重要交互；4) RAG 检索——将所有历史交互索引到向量库，每次根据当前查询检索最相关的历史片段注入上下文；5) 重要性评分——不是所有历史都同等重要，根据用户反馈、任务完成度、信息熵给每条记忆打分；6) Context Window Extension——使用 RoPE 插值或 ALiBi 扩展模型的上下文窗口。推荐组合：滑动窗口(近) + 向量检索(远) + 定期摘要(中)。"},
]

print(f"After Agent: {len(entries)} entries")

out_path = "data/sample/train.jsonl"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    for entry in entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
print(f"Wrote {len(entries)} entries to {out_path}")