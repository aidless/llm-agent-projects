"""
v3 数据集补充（追加到 v3 划分）。

覆盖主题：
  - LLM 推理优化（vLLM、TGI、KV cache、量化、continuous batching）
  - 分布式训练（DDP、DeepSpeed、ZeRO、FSDP）
  - GPU 编程与优化（CUDA、cuDNN、混合精度、gradient checkpointing）
  - 高级 Python（asyncio、生成器、描述符、元类）
  - MLOps（MLflow、Kubeflow、Feast、Feature Store）
  - LLM 评估（BLEU、ROUGE、BERTScore、LLM-as-Judge、Human Eval）

每个主题 ~20 条，共 ~120 条
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# === 1. LLM 推理优化 ===
LLM_INFERENCE = [
    ("vLLM 相比传统 HuggingFace 推理有什么性能优势？",
     "vLLM 通过 PagedAttention 技术解决 KV cache 内存碎片问题，传统实现连续分配导致长序列时浪费高达 60-80% 显存。vLLM 把 KV cache 分成固定大小 block 块（默认 16 token/block），按需分配类似 OS 虚拟内存。配合 continuous batching（连续批处理），新请求可在旧请求 decode 中插入而不必等整批完成，吞吐量比 HuggingFace Transformers 高 14-24 倍，接近 GPU 理论峰值。生产部署时还内置张量并行、Speculative Decoding、流式输出等能力。"),
    ("什么是 PagedAttention？它的核心创新点是什么？",
     "PagedAttention 受 OS 虚拟内存分页机制启发，把每个序列的 KV cache 分成固定大小 block（block_size 通常 16），通过 block table 维护逻辑-物理映射。核心创新：1) 显存按需分配，浪费从 60% 降到 <4%；2) block 共享（同一 prompt 多个采样分支只复制一次）；3) Copy-on-Write 支持 beam search / parallel sampling；4) 与 vLLM 的 continuous batching 协同——不同长度请求可批处理，block 独立管理。在 13B 模型上比 HF Transformers 高 23 倍吞吐。"),
    ("continuous batching 和 static batching 的区别是什么？",
     "Static batching（传统）将 N 个请求组成一批，必须等整批全部生成完成才能开始下一批，GPU 经常因长短不一而空转。Continuous batching 持续维持活跃请求池，任意时刻只要有空闲 slot 就插入新请求，已完成的请求立即让出 slot，新请求接上正在 decode 的请求。优势：1) 吞吐提升 2-4 倍；2) P99 延迟降低（短请求不必等长请求）；3) GPU 利用率从 30-40% 提升到 80%+。vLLM、Text Generation Inference、TensorRT-LLM 都已实现。"),
    ("什么是 KV cache 量化？FP8 和 INT8 选哪个？",
     "KV cache 量化把 attention 计算时缓存的 K/V 张量从 FP16 量化到低精度。FP8：E4M3 格式，4 bit 指数+3 bit 尾数，精度损失小，H100 原生支持，速度提升 1.5-2x，显存减半。INT8：对称 per-token 量化，速度提升 1.3-1.8x，显存减半，需校准。FP8 通常更好（精度+速度平衡），但需要 Hopper 及以上 GPU。INT8 兼容 Ampere，更通用。BitServing 团队证明：FP8 KV cache 在 100K 上下文长文本任务上精度损失 <0.1%。"),
    ("什么是 Speculative Decoding？",
     "Speculative Decoding 用小模型（draft model）猜测大模型（target model）的输出，用一次大模型前向验证多个 token。例如用 7B 草拟 5 个 token，主 13B 模型一次验证：5 个都对则收益 5x 加速，错的从第一个错的开始用大模型重生成。关键：1) 草拟模型必须小（1B-3B）、快；2) 草拟与目标模型 tokenizer 必须一致；3) acceptance criterion 严格保证输出分布不变。EAGLE、Medusa、Lookahead Decoding 是改进方向。vLLM 已集成 speculative_model 参数。"),
    ("什么是 TensorRT-LLM？和 vLLM 比有什么区别？",
     "TensorRT-LLM 是 NVIDIA 官方 LLM 推理优化库，将模型编译为针对目标 GPU 优化的 engine（kernel fusion、INT4/INT8 量化、in-flight batching）。优势：极致性能，同 GPU 下通常比 vLLM 快 1.3-2 倍，H100 上 70B 模型可达 5000+ tokens/s/用户。劣势：1) 编译耗时（首次 30-60 分钟）；2) 灵活性差，新模型需等官方支持；3) 部署复杂。vLLM 用纯 Python+PagedAttention 动态调度，灵活度高，社区更新快。生产环境：H100/A100 追求极致吞吐选 TensorRT-LLM；快速迭代新模型选 vLLM。"),
    ("什么是 FlashAttention？为什么重要？",
     "FlashAttention 是 IO 感知的精确 attention 算法，通过分块计算（tiling）+ kernel fusion 减少 HBM 读写。传统 attention 计算 QK^T 时必须把中间结果写回 HBM 再读，flash attention 把 QK^T、softmax、PV 三步在 SRAM 内部完成。优势：1) 速度提升 2-4 倍；2) 显存从 O(N²) 降到 O(N)（不用存 attention matrix）；3) 精度不退化（精确计算而非近似）。FlashAttention-2 进一步利用 warp-specialization，H100 上达 75% 理论峰值。已成为 PyTorch 2.0+ 标准 attention 后端（torch.nn.functional.scaled_dot_product_attention）。"),
    ("vLLM 的 Speculative Decoding 怎么配置？",
     "vLLM 通过 speculative_model 参数启用，典型配置：```python\nfrom vllm import LLM, SamplingParams\nllm = LLM(\n    model='meta-llama/Llama-2-70b-hf',  # 主模型\n    speculative_model='meta-llama/Llama-2-7b-hf',  # 草拟模型\n    num_speculative_tokens=5,  # 每次投机 5 个\n    speculative_draft_tensor_parallel_size=1,  # 草拟用 1 张 GPU\n)\n```。注意：1) 草拟模型必须与主模型同 tokenizer；2) 草拟模型应小（<主模型 1/5）；3) num_speculative_tokens 4-8 通常最优。在 batch 大、decode 阶段效果显著。"),
    ("什么是 Medusa 多头预测？",
     "Medusa 是 Speculative Decoding 的简化版：不训练独立草拟模型，而是在主模型最后几层加多个并行的 MLP 头（heads），每个头预测未来不同位置的 token。例如 1 个主头 + 3 个 Medusa 头，一次前向同时预测 4 个 token。优势：1) 无需独立小模型；2) 训练成本低（只微调 Medusa 头）；3) 推理代码简单。劣势：精度比独立草拟模型略差。在 7B 模型上加速 1.8-2.3 倍。EAGLE 是其改进版，结合 hidden state 特征进一步提升。"),
    ("解释一下什么是 vLLM 的 chunked prefill",
     "Chunked prefill 把超长 prompt 的 prefill 阶段拆成多个 chunk（如 4096 token/chunk），与 decode 请求一起批处理。优势：1) 避免单请求阻塞整个 GPU；2) prefill 和 decode 共享同一 batch，提升利用率；3) 长 prompt（>10K token）请求不再饥饿短请求。vLLM 通过 enable_chunked_prefill=True 启用。配合 max_num_batched_tokens 控制总批大小。这是 vLLM 在 100K 上下文场景下仍能保持低延迟的关键设计。"),
]

# === 2. 分布式训练 ===
DISTRIBUTED_TRAINING = [
    ("什么是 DistributedDataParallel（DDP）？和 DataParallel（DP）有什么区别？",
     "DDP 是 PyTorch 的多进程分布式训练方案，每张 GPU 跑独立进程，各持有模型副本，通过 NCCL all-reduce 同步梯度。DP 是单进程多线程，主卡算完广播给其他卡。DDP 优势：1) 消除 GIL 瓶颈，多卡利用率接近线性；2) 通过 Ring AllReduce 通信高效；3) 支持多机多卡。DP 局限：主卡通信瓶颈，多卡效率差。代码差异：DP 一行 nn.DataParallel(model)，DDP 需用 torch.distributed.launch 启动，模型用 DistributedDataParallel 包装。生产训练几乎都选 DDP。"),
    ("什么是 DeepSpeed ZeRO？三个阶段有什么区别？",
     "ZeRO（Zero Redundancy Optimizer）由微软 DeepSpeed 提出，将训练显存分三类：optimizer state（优化器状态，Adam 占大部分）、gradient、parameter。Stage 1 只分 optimizer state，每卡显存降至 1/4。Stage 2 加上 gradient，每卡降至 1/8。Stage 3 加上 parameter，每卡降至 1/N（N 为卡数）。Stage 3 实际等于模型分片（类似 FSDP），通信开销增加。实际选择：<1B 模型用 Stage 1 即可；1B-10B 用 Stage 2；>10B 用 Stage 3。配合 CPU offload 可训练更大模型。"),
    ("FSDP（Fully Sharded Data Parallel）和 DDP 有什么区别？",
     "DDP 每卡保存完整模型 + 完整 optimizer state，梯度通过 all-reduce 同步。FSDP 把模型参数、梯度、optimizer state 都分片到多卡，每卡只持有 1/N，通过 all-gather 临时重组。优势：显存降至 1/N，可训练大模型（70B+ 只需较少卡数）。劣势：通信开销大（每次前向需 all-gather 参数），需 careful 配置 `backward_prefetch` 和 `forward_prefetch` 隐藏延迟。PyTorch 1.11+ 原生支持，比 DeepSpeed 更轻量。HuggingFace Trainer 一行 fsdp='full_shard' 即可启用。"),
    ("什么是混合精度训练？fp16 和 bf16 选哪个？",
     "混合精度训练在 forward/backward 用低精度（FP16 或 BF16）计算加速，optimizer step 用 FP32 维护主权重。FP16（半精度）：5 bit 指数+10 bit 尾数，数值范围 ±65504，精度高但易溢出（需 loss scaling）。BF16（Brain Float）：8 bit 指数+7 bit 尾数，数值范围同 FP32（不易溢出），但精度低。选 BF16：如果是 Ampere+ GPU（A100/3090/4090）。选 FP16：V100 或对精度敏感。实际现在新硬件几乎都选 BF16 避免 loss scaling 麻烦。"),
    ("什么是 gradient checkpointing？代价是什么？",
     "Gradient checkpointing（重计算）通过在前向时只保存部分层的激活值，反向时重算未保存的激活值来节省显存。代价：1) 训练速度下降约 30-40%（重算开销）；2) 代码稍复杂（用 torch.utils.checkpoint.checkpoint 包装 block）。显存收益：原本存所有激活，改为只存部分，节省 50-70%。典型做法：encoder-decoder 中只对 encoder 做 checkpoint，对 decoder 保留激活。RTX 3060 6GB 训练 3B 模型必须启用，否则 OOM。"),
    ("什么是 ZeRO-Offload？",
     "ZeRO-Offload 把 optimizer state 和部分计算 offload 到 CPU 内存，进一步降低单卡显存。原理：optimizer step 计算量小（与前向比 1:10），将 Adam 的 FP32 master weight 和 moment 放在 CPU，optimizer step 在 CPU 跑，每 step 与 GPU 同步一次。结果：单卡训练模型规模扩大 10x（如 6GB 卡可训练 1.3B 模型）。代价：CPU-GPU 通信增加，训练速度降 10-20%。在单机多卡 + CPU 大内存场景性价比高。云端多机训练一般不需要（直接加卡）。"),
    ("Pipeline Parallel 和 Tensor Parallel 有什么区别？",
     "Tensor Parallel（TP）把单层 attention/FFN 切到多卡，每卡算部分头/维度，结果通过 all-reduce 组合。适合单层大、卡数少（2-8）。Pipeline Parallel（PP）把整个模型按层切到多卡（如 24 层分成 4 段，每卡 6 层），前向时数据像流水线一样过每段。优势：可训练超深/超大模型。劣势：存在 bubble（等待）和激活显存问题。生产中常组合 TP+PP（如 70B 模型用 TP=8 + PP=2）。"),
    ("什么是 ZeRO-Infinity？",
     "ZeRO-Infinity 是 DeepSpeed 的第三阶段扩展，将 optimizer state、gradient、parameter 全部分片到 NVMe SSD 内存，理论上可训练任意大模型。原理：SSD 作为内存扩展，CPU 内存作为高速缓存。需要：1) 高速 NVMe（吞吐 ≥3 GB/s）；2) 大量 SSD 空间（参数×4+）。代价：通信瓶颈在 SSD I/O，训练速度慢 5-10 倍。实际应用：训练万亿参数模型时使用。300B+ 模型单机可训，但通常不实用——直接多机更高效。"),
    ("什么是 torch.compile？对分布式训练有帮助吗？",
     "torch.compile（PyTorch 2.0+）通过 Triton/AOT 自动将 PyTorch 代码编译为优化 kernel，减少 Python overhead 与显存碎片。优势：单卡推理/训练提速 1.3-2x。分布式场景：每卡独立 compile，通信（NCCL all-reduce）不受影响；但有 distributed checkpointing 时需注意 dynamo 与 DDP 的兼容（旧版有 bug，新版已修复）。对推理比训练更明显。FSDP 下建议 backend='inductor' 配合 mode='reduce-overhead'。"),
    ("什么是 FlashAttention 2/3？和 FA1 比有什么改进？",
     "FlashAttention-2（2023）通过 warp-specialization 让一个 warp 做 matmul、另一个 warp 做 softmax，H100 上达 75% 理论峰值。FlashAttention-3（2024）针对 H100 的 WGMMA 指令进一步优化，FP8 下达 1.2 PFLOPS（接近 75% 峰值）。相对 FA1 改进：1) 速度提升 1.5-2x；2) 支持更长上下文（百万级 token 训练）；3) FP8 原生支持，量化训练精度损失小。已成为 LLM 长上下文训练的事实标准。"),
]

# === 3. GPU 编程与优化 ===
GPU_PROGRAMMING = [
    ("什么是 CUDA stream？",
     "CUDA stream 是 GPU 上的执行队列，同 stream 内操作串行，不同 stream 间可并行。常用于：1) 计算与传输 overlap（一边拷贝数据一边跑 kernel）；2) 多卡并行（每卡一个 stream）。用法：```python\nstream = torch.cuda.Stream()\nwith torch.cuda.stream(stream):\n    output = model(input)\ntorch.cuda.current_stream().wait_stream(stream)\n```。PyTorch 默认所有操作在 default stream 上。性能关键路径（如 decode + prefill overlap）需要手动管理 stream。"),
    ("什么是 GPU 显存碎片？如何解决？",
     "显存碎片指空闲显存总量够但分配连续大块失败。常见原因：1) 不同 shape 张量反复分配/释放；2) PyTorch caching allocator 只在 block 大小相同时复用。解决方法：1) 用 torch.cuda.empty_cache() 强制释放未用缓存；2) export PYTORCH_NO_CUDA_MEMORY_CACHING=1（关闭缓存）；3) 训练时固定 batch 大小避免动态 shape；4) 推理用 vLLM/TensorRT-LLM 内部管理。监控：torch.cuda.memory_summary() 看 fragmentation。"),
    ("什么是 torch.compile 的 mode 参数？",
     "torch.compile 提供三种 mode：default（平衡速度与编译时间）、reduce-overhead（用 CUDA graphs 减少 Python 开销，单卡推理最佳）、max-autotune（最激进，自动选择最佳 kernel，编译时间最长）。建议：1) 训练用 default；2) 推理用 reduce-overhead；3) 关键性能路径试 max-autotune。回退：mode='default' 编译失败概率低；max-autotune 可能对某些自定义算子失败。"),
    ("什么是 FlashDecoding？和 FlashAttention 有什么区别？",
     "FlashDecoding 是 FlashAttention 在 inference decode 阶段（短 query × 长 key）的优化。FlashAttention 假设 Q 较长，decode 时 Q 只有 1 token 浪费并行。FlashDecoding 把长 K/V 拆成多个 block 块并行算 softmax partial，再合并结果。优势：1) decode 阶段快 2-4x；2) 长上下文（>10K）效益显著；3) 与 PagedAttention 兼容。vLLM、TensorRT-LLM 都已集成。是 LLM 推理吞吐提升的关键之一。"),
    ("什么是 mixed precision autocast？",
     "PyTorch 的 torch.cuda.amp.autocast 自动决定哪些 op 用 FP16/BF16、哪些保持 FP32（如 loss、batchnorm）。典型用法：```python\nwith torch.cuda.amp.autocast(dtype=torch.bfloat16):\n    loss = model(input)\nloss.backward()\noptimizer.step()\n```。注意：autocast 不替换权重类型，只替换 forward 计算的中间值。模型权重仍是 FP32（除非显式 .half()）。比手动管理精度简单且稳定。"),
    ("什么是 GPU 共享内存（shared memory）？",
     "CUDA shared memory 是每个 SM（Streaming Multiprocessor）上的高速片上内存（约 48-100 KB/SM），速度与 L1 cache 相当。优势：block 内线程可高效通信；常用于 tile-based 算法（FlashAttention 即用 shared memory 做 QK^T 分块）。性能关键：所有需要 block 内线程协作的 kernel 优化都依赖 shared memory。L2 cache 反而是 global 的，所有 SM 共享。Hopper 增加 distributed shared memory，跨 SM 也可访问。"),
    ("什么是 CUDA kernel fusion？",
     "传统 PyTorch 逐 op 调度，每次 op 都产生显存读写。CUDA kernel fusion 把多个 op 合并到单个 kernel，省去中间结果读写。例子：LayerNorm + 残差连接 → 单个 fused kernel；GeLU + 矩阵乘 → fused activation-matmul。工具：1) 手写 Triton（NVIDIA 自家，比 CUDA 简单）；2) torch.jit.script / torch.compile 自动融合；3) Apex / xformers 提供 fused 实现（如 fused LAMB、fused RoPE）。"),
    ("什么是 CUDA Graphs？",
     "CUDA Graphs 把一系列 CUDA 操作录制为图，重放时 GPU 按图执行，省去每次 launch 的 CPU 开销（~10us/launch）。对于固定 shape 的小模型推理提速 20-30%。PyTorch 用 torch.cuda.CUDAGraph() 包装。限制：1) 录制时所有 shape 必须固定；2) 跨卡通信需小心；3) 录制后只能用相同 shape。生产：vLLM、TensorRT-LLM 内部对 decode 阶段用 CUDA Graphs。training 一般不用（shape 变化多）。"),
    ("如何诊断 GPU 训练性能瓶颈？",
     "步骤：1) nsight-compute 抓单个 kernel（看 SM 利用率、memory 带宽、occupancy）；2) nsight-systems 看时间线（kernel launch、CPU/GPU overlap、数据传输）；3) torch.profiler 看 Python overhead（特别是 forward 中的 Python loop）；4) torch.cuda.utilization 监控整体利用率。常见瓶颈：a) memory-bound（matmul 太小，如最后一层）→ 加 batch；b) launch-bound（太多小 kernel）→ 融合；c) Python overhead → torch.compile；d) 数据加载慢 → num_workers 增加。"),
    ("什么是 GPUDirect Storage？",
     "GPUDirect Storage 允许 GPU 直接从 NVMe SSD 读取数据，跳过 CPU 内存中转。优势：1) 训练大规模数据集时省去 CPU 内存瓶颈；2) 减少 IO 等待，提升 throughput；3) 单机可处理 TB 级数据集。需要：1) CUDA 11.4+；2) Linux 内核 5.x+ 支持；3) NVMe 驱动支持（GDS）。PyTorch DataLoader 可通过 kvikio 库集成。生产中训练 100B token 数据集时关键。"),
]

# === 4. 高级 Python ===
ADVANCED_PYTHON = [
    ("Python 中 GIL 是什么？多线程为什么没加速？",
     "GIL（Global Interpreter Lock）是 CPython 解释器互斥锁，同一时刻只允许一个线程执行 Python 字节码。目的：保护引用计数避免并发崩溃。多线程对 CPU 密集型任务无加速（如计算 1+1 也是 1 个线程占用 CPU）。I/O 密集型可加速（I/O 等待时释放 GIL）。绕过：1) multiprocessing（多进程真正并行）；2) C 扩展（NumPy/requests 释放 GIL）；3) 协程（asyncio，单线程内异步）；4) 换 PyPy / no-GIL Python 3.13+。"),
    ("Python 的描述符（descriptor）是什么？有什么应用？",
     "描述符是实现了 __get__、__set__、__delete__ 之一的类，可控制属性访问。应用：1) property（@property 装饰器就是基于描述符）；2) 类方法（classmethod、staticmethod）；3) ORM 字段映射（如 Django Field）；4) 验证与类型检查（@validate_positive）。优势：把数据访问行为封装在类级别，可复用。理解描述符是理解 Python 数据模型、metaclass 的基础。"),
    ("Python 的 metaclass 是什么？什么场景用？",
     "Metaclass 是类的类，控制类的创建过程（type 默认是 metaclass）。自定义 metaclass 通过继承 type 重写 __new__ 或 __init__，可拦截类的属性、方法、父类等。典型场景：1) ORM 定义表结构（class User 自动生成 SQL）；2) 强制接口规范（如所有子类必须实现某方法）；3) 框架类注册（Pluggable 架构）。注意：metaclass 是高级特性，滥用会让代码极难理解。99% 的项目不需要 metaclass。"),
    ("Python 协程与线程、进程的本质区别？",
     "线程：OS 调度的执行单元，共享进程内存，Python 受 GIL 限制。进程：独立内存空间，避开 GIL 真正并行，但通信成本高（IPC）。协程：用户态的轻量级执行单元，单线程内由程序主动让出（yield/await）。协程优势：1) 切换成本极低（ns 级 vs OS 线程 us 级）；2) 同步风格写异步代码（async/await）；3) 单进程可开 10 万+ 协程。适合 I/O 密集型（网络、文件）。Python asyncio 库是协程实现。"),
    ("Python 内存管理：引用计数 + 循环垃圾回收？",
     "CPython 主要用引用计数（每个对象有 ob_refcnt）。对象引用为 0 时立即释放。引用计数不能解决循环引用（a.x=b, b.y=a）→ 引入 cyclic GC（gc 模块），定期扫描容器对象（list/dict/class）检测循环。调优：1) 大对象及时 del；2) 避免循环引用（用 weakref）；3) 关闭 GC：gc.disable()（短期脚本）。内存泄漏排查：tracemalloc、objgraph、heapy（guppy3）。"),
    ("Python 装饰器的工作原理？手写一个带参数的装饰器？",
     "装饰器本质是一个函数，接收函数返回新函数。带参数装饰器需要三层：\n```python\ndef repeat(times):\n    def decorator(func):\n        def wrapper(*args, **kwargs):\n            for _ in range(times):\n                result = func(*args, **kwargs)\n            return result\n        return wrapper\n    return decorator\n\n@repeat(3)\ndef greet(name):\n    print(f'Hi {name}')\n```\n关键：1) wrapper 必须接受 *args, **kwargs 透传；2) 用 functools.wraps(func) 保留原函数元信息。"),
    ("Python 上下文管理器（context manager）有什么用？",
     "上下文管理器（with 语句）保证资源的正确获取与释放，即使中间抛出异常。常见应用：1) 文件操作（with open() as f: 自动关闭）；2) 锁（with lock: 自动释放）；3) 数据库连接（with conn: 自动 commit/rollback）；4) 计时（with timer(): 自动打印耗时）；5) 修改全局状态（with mock.patch()）。自定义：通过实现 __enter__ 和 __exit__，或用 @contextlib.contextmanager 装饰器把生成器函数转成 context manager。"),
    ("Python 的生成器（generator）和 yield 是什么？",
     "生成器是包含 yield 表达式的函数，调用时返回 generator 对象而非立即执行。每次 next() 触发到下一个 yield 暂停并返回值。优势：1) 惰性求值，省内存（生成 1000 万行只需保留当前行）；2) 适合流式数据（读大文件、网络响应）；3) 简化迭代器实现（自动实现 __iter__ 和 __next__）。典型应用：1) pandas 读 CSV（chunksize）；2) 异步 I/O（async def + yield）；3) TensorFlow DataFlow（tf.data.Dataset.from_generator）。"),
    ("Python 中 *args 和 **kwargs 的区别？",
     "*args 接收位置参数，打包为 tuple；**kwargs 接收关键字参数，打包为 dict。典型场景：1) 装饰器（wrapper 透传原函数参数）；2) 继承时透传父类 __init__ 参数；3) 灵活 API（接受任意额外参数）。示例：\n```python\ndef log(*args, **kwargs):\n    print(f'args: {args}, kwargs: {kwargs}')\nlog(1, 2, name='Alice', age=25)\n# args: (1, 2), kwargs: {'name': 'Alice', 'age': 25}\n```\n函数定义时也可解包：func(*[1,2,3]) → func(1,2,3)。"),
    ("Python typing.Protocol 是什么？什么时候用？",
     "typing.Protocol 用于定义结构化类型（duck typing），只要类实现了指定方法，就视为该 Protocol 子类，无需显式继承。应用：1) 解耦接口与实现；2) 测试中 mock 实现；3) 函数签名声明接受任何具有某方法的对象。例如 Iterator 协议（只要有 __iter__ 和 __next__ 即可）。Python 3.8+。与 ABC 区别：ABC 强制继承（is-a），Protocol 只看方法（has-a）。新代码推荐 Protocol，更灵活。"),
]

# === 5. MLOps ===
MLOPS = [
    ("什么是 MLflow？核心组件有哪些？",
     "MLflow 是开源 MLOps 平台，核心 4 组件：1) Tracking——记录实验参数、指标、artifact（支持 sklearn/PyTorch/TensorFlow）；2) Projects——打包代码为可复现运行环境（MLproject 文件 + Conda env）；3) Models——统一模型格式（多种 flavor），支持一键部署；4) Registry——模型版本管理与 stage 流转（staging/production）。生产中常与 S3/MinIO 配合存 artifact，配合 Airflow/Kubeflow 调度。轻量但功能完整。"),
    ("什么是 Kubeflow？和 MLflow 有什么区别？",
     "Kubeflow 是基于 Kubernetes 的端到端 ML 平台，组件：1) Pipelines（KFP）—— 定义 DAG workflow（数据准备→训练→评估→部署）；2) Training Operators（TFJob/PyTorchJob）—— K8s 原生分布式训练；3) Katib—— AutoML 与超参搜索；4) Serving（KFServing）—— 模型服务化（含弹性伸缩、流量切分）；5) Notebooks—— Jupyter。适用：大规模训练+部署的云端场景。MLflow 偏实验管理，Kubeflow 偏端到端云原生。"),
    ("什么是 Feature Store？什么时候需要？",
     "Feature Store 是集中管理机器学习特征的系统，离线用于训练（一致性历史特征），在线用于推理（低延迟实时特征）。代表：Feast、Tecton、Hopsworks。需求场景：1) 多个模型共享特征；2) 训练/推理特征一致性；3) 特征版本管理；4) 上千特征难以手工管理。轻量场景：用 Redis 缓存高频特征就够。重场景：使用 Feast（开源）或 Tecton（云）。"),
    ("什么是 ML 模型监控？关键指标有哪些？",
     "ML 模型监控在生产环境跟踪模型表现，分三层：1) 数据漂移——输入特征分布变化（PSI、KS 检验）；2) 概念漂移——特征与目标关系变化（如疫情期间消费模式）；3) 性能漂移——实际模型质量下降（需 ground truth，延迟获取）。关键指标：accuracy/precision/recall 滑动窗口均值、预测置信度分布、特征分布 PSI、推理延迟。工具：Evidently AI、Whylogs、Prometheus + Grafana。告警：指标超阈值自动通知。"),
    ("什么是 CI/CD for ML（MLOps）？",
     "MLOps 把 DevOps 思想扩展到 ML 生命周期，阶段：1) Data validation（Great Expectations 校验数据 schema/分布）；2) Model training（GitHub Actions 触发训练，产出 artifact）；3) Model evaluation（自动评估指标，对比 baseline，阈值通过才部署）；4) Model deployment（蓝绿/金丝雀发布，K8s + Istio 流量切分）；5) Monitoring（Evidently 漂移检测）。与 DevOps 区别：除代码外还有数据、模型、pipeline 需版本管理（DVC、MLflow）。"),
    ("什么是 Data Version Control（DVC）？",
     "DVC 是数据/模型版本控制工具，git-like 工作流。原理：大文件（数据集、模型权重）存 S3/GCS，DVC 文件存元数据（hash、路径），git 仓库只跟踪小文件。命令类似 git（dvc add、dvc commit、dvc push）。优势：1) 复用 Git 工具链；2) 流水线可复现（dvc repro 重跑 stage）；3) 与 MLflow 互补（MLflow 管实验，DVC 管数据/代码）。轻量替代：LakeFS、Pachyderm（更重量级）。"),
    ("什么是 Great Expectations？",
     "Great Expectations（GX）是数据质量验证框架，定义 expectation（如 'column age must be > 0'），可对 dataframe/数据库执行验证，产出 data docs HTML 报告。典型用法：1) 数据管道每步加 expect_table_row_count_to_be_between；2) CI 中运行 GX suite 验证训练数据；3) 监控生产数据漂移。优势：1) 声明式（不用写 assert）；2) 自动生成文档；3) 与 Pandas/Spark/SQL 集成。新版 GX v0.15+ 重写了 API。"),
    ("什么是 Argo Workflows？和 Airflow 有什么区别？",
     "Argo Workflows 是 K8s 原生 workflow 引擎，定义 DAG 在 K8s 上跑（每个 step 一个 Pod）。优势：1) 与 K8s 生态无缝（Volumes、Secrets、ConfigMap 共享）；2) 容器化 step（任意 Docker 镜像）；3) 适合 ML 训练（DAG 调度 + GPU 调度）。Airflow 是 Python 写的传统调度器，Operator 模型，社区插件丰富。生产：1) 纯云原生 ML 平台选 Argo；2) 复杂 ETL 业务逻辑选 Airflow；3) K8s-first 团队常用 Argo + Airflow 配合。"),
    ("什么是 Ray？和 Dask 有什么区别？",
     "Ray 是分布式执行框架，提供：1) Ray Core（task/actor 模型）；2) Ray Tune（超参搜索）；3) Ray Serve（模型服务）；4) Ray Train（分布式训练）；5) Ray Data（数据预处理）。优势：1) Pythonic API（@ray.remote 一行分布式）；2) 与 ML 生态深度集成（sklearn/PyTorch/HuggingFace）；3) 适合小到中等规模分布式。Dask 偏数据科学（Pandas/NumPy 分布式）。LLM 训练 Ray 用得少（更常用 DeepSpeed/Megatron），但 RLHF、超参搜索常用 Ray Tune。"),
    ("什么是 ML 模型注册中心（Model Registry）？",
     "Model Registry 是集中管理模型版本的系统，功能：1) 存储模型 artifact + 元数据（训练参数、指标、训练时间）；2) 版本管理（v1、v2、v3）；3) Stage 流转（None → Staging → Production → Archived）；4) 审批工作流（生产部署需 approval）；5) 关联实验（链接到训练 run）。代表：MLflow Registry、HuggingFace Hub、Weights & Biases Artifacts。生产中模型上线必经 Registry，避免 ad-hoc 上传。"),
]

# === 6. LLM 评估 ===
LLM_EVAL = [
    ("什么是 BLEU 分数？有什么局限？",
     "BLEU（Bilingual Evaluation Understudy）是机器翻译经典指标，度量 n-gram 精度（候选译文与参考译文共享 n-gram 比例），加 brevity penalty 防短译文。1-4 gram 几何平均。范围 0-1，越高越好。局限：1) 不考虑语义（同义词不同分）；2) 不考虑句子结构；3) 短文偏高；4) 不适合开放式生成（对话、摘要）。改进：SacreBLEU（标准化 tokenization）、BLEURT（用 BERT 替换 n-gram）。"),
    ("什么是 ROUGE 分数？和 BLEU 有什么区别？",
     "ROUGE（Recall-Oriented Understudy for Gisting Evaluation）用于摘要评估，基于 recall 而非 precision。变体：ROUGE-N（n-gram recall）、ROUGE-L（最长公共子序列，捕捉句子结构）、ROUGE-W（加权 LCS）、ROUGE-S（skip-bigram，跳过中间词）。与 BLEU 区别：ROUGE 关心参考覆盖（适合摘要），BLEU 关心生成正确（适合翻译）。局限仍是 n-gram，不能捕捉语义。"),
    ("什么是 BERTScore？",
     "BERTScore 用预训练 BERT 的 contextual embedding 计算候选与参考的余弦相似度，分 token-level precision/recall/F1。优势：1) 捕捉语义相似（同义词仍给高分）；2) 对 paraphrasing 鲁棒；3) 多语言支持（mBERT、XLM-R）。局限：1) 仍依赖参考；2) 计算慢（每个 token 都要 forward BERT）；3) 对小模型效果一般。论文：Zhang et al. 2020。代码：bert-score pip 包。中文场景建议用 `lang=zh` 参数调用 XLM-R。"),
    ("什么是 LLM-as-Judge？有什么优势？",
     "LLM-as-Judge 用强 LLM（GPT-4、Claude）作为评分员，对被评估模型的回答打 0-5 分。优势：1) 无需人工标注；2) 多维度（准确性、相关性、流畅性）；3) 可解释（judge 给出理由）。代表工作：MT-Bench、LMSYS Chatbot Arena、AlpacaEval、Prometheus（开源 judge 模型）。局限：1) Judge 自身偏差（位置偏见、长度偏见、自我偏好）；2) 成本（每个样本要 judge）；3) 需 calibrate（定期用人工样本验证 judge 质量）。"),
    ("什么是 HumanEval？",
     "HumanEval 是 OpenAI 2021 提出的代码生成 benchmark，164 个 Python 编程问题，每个含函数签名、docstring、隐藏测试用例。评估 pass@k：模型生成 k 个候选，至少 1 个通过测试。代表：GPT-4 pass@1 ≈ 67%、Claude-3 ≈ 76%。类似 benchmark：MBPP（基础 Python 问题）、LiveCodeBench（持续更新）、HumanEval-X（多语言扩展）、BigCodeBench（更复杂任务）。局限：1) 题目偏简单；2) 真实软件工程任务与 HumanEval 差距大；3) 数据集已被训练集污染（需 LiveCodeBench 等动态版本）。"),
    ("什么是 MMLU？怎么测？",
     "MMLU（Massive Multitask Language Understanding）是综合知识 benchmark，57 个学科（历史、法律、医学、数学等），约 14K 多选题。评估 5-shot 准确率。GPT-4 约 86%，Llama-3-70B 约 79%。局限：1) 多选形式不能评估开放式；2) 答案固定（可能猜对）；3) 文化偏英文。补充：MMLU-Pro（增加推理题）、C-Eval（中文）。注意：MMLU 已大量被训练集吸收（污染），新模型评估建议用 MMLU-Pro 或 LiveBench。"),
    ("什么是 MT-Bench？和 AlpacaEval 有什么区别？",
     "MT-Bench（LMSYS）是多轮对话 benchmark，80 个高质量问题，涵盖 8 类（写作、推理、编码、提取等），用 GPT-4 评分（1-10 分）。AlpacaEval 是单轮开放式生成 benchmark，805 个问题，用 GPT-4 比较 base 与 reference 的胜率。MT-Bench 多轮（更接近真实场景）、AlpacaEval 简单胜率统计。两者都依赖 GPT-4 judge（成本 + 偏差）。新替代：AlpacaEval 2.0（开源 LLM judge）、Arena-Hard（真实用户投票）。"),
    ("什么是 LiveBench？",
     "LiveBench 是 2024 提出的动态 benchmark，每月更新题目以避免训练集污染。覆盖 6 类（数学、代码、推理、语言、指令遵循、困难题目）。从 IMO、LeetCode 等真实来源出题。强模型如 GPT-4o 在 LiveBench 上约 60-70%。优点：1) 永远不过时；2) 题目真实有难度；3) 自动评分（无 LLM judge 偏差）。缺点：1) 任务偏学术；2) 商业模型公开分数才有用。LiveCodeBench 是其子集（仅代码）。"),
    ("什么是 Bias Benchmark for Question Answering（BBQ）？",
     "BBQ 是偏见评估 benchmark，测试模型对 9 类社会偏见（年龄、性别、种族、宗教、性取向、社会经济地位、残疾、国籍、外貌）的回答。包含 58K 例子。评估 bias score（-1 到 +1，0 为中性，负数偏向刻板印象）。优势：1) 系统性测偏见；2) 多维度；3) 与真实世界偏见对应。模型如 GPT-4 仍有 -3% bias。LLM 部署前建议跑 BBQ/BBQ-Ambig 评估。"),
    ("什么是 LLM 评估中的 Position Bias？",
     "Position bias 是 LLM-as-Judge 的已知问题：judge 模型倾向于给排在第一个的候选更高分（recency/primacy bias）。解决方法：1) 随机化候选顺序（多次评估取平均）；2) 双盲评估（两个候选位置各评一次取均值）；3) 用 Prometheus 等开源 judge（已训练减少 bias）。缓解代码：```python\nscore1 = judge(answer_a, answer_b)\nscore2 = judge(answer_b, answer_a)  # 交换位置\nreturn (score1 + score2) / 2\n```。位置 bias 会让 LLM 评测绝对分数不可信，但相对排名（Win Rate）仍可用。"),
]

# === 写入 v3 train.jsonl ===
def main():
    all_entries = LLM_INFERENCE + DISTRIBUTED_TRAINING + GPU_PROGRAMMING + ADVANCED_PYTHON + MLOPS + LLM_EVAL
    train_path = Path("data/audit/2026-07-22-v3/train.jsonl")
    if not train_path.exists():
        print(f"FAIL: {train_path} 不存在")
        return 1
    with train_path.open("a", encoding="utf-8") as f:
        for inst, out in all_entries:
            f.write(json.dumps({"instruction": inst, "output": out}, ensure_ascii=False) + "\n")
    # 重新审计
    print(f"Appended {len(all_entries)} entries to {train_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
