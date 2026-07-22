"""DAG (有向无环图) 构建和验证"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

from .errors import (
    CyclicDependencyError,
    DAGValidationError,
    MissingConnectionError,
    OrphanNodeError,
    NodeNotFoundError,
)


class DAG:
    """有向无环图，用于工作流拓扑排序和验证"""

    def __init__(self):
        self._edges: Dict[str, Set[str]] = defaultdict(set)   # node -> successors
        self._reverse: Dict[str, Set[str]] = defaultdict(set) # node -> predecessors
        self._nodes: Set[str] = set()

    # ── 图操作 ──────────────────────────────────────────────

    def add_node(self, node_id: str) -> None:
        self._nodes.add(node_id)
        # 确保 edges/reverse 中也有 key
        self._edges.setdefault(node_id, set())
        self._reverse.setdefault(node_id, set())

    def add_edge(self, source: str, target: str) -> None:
        if source not in self._nodes:
            raise NodeNotFoundError(f"源节点 '{source}' 不存在")
        if target not in self._nodes:
            raise NodeNotFoundError(f"目标节点 '{target}' 不存在")
        self._edges[source].add(target)
        self._reverse[target].add(source)

    def remove_node(self, node_id: str) -> None:
        if node_id not in self._nodes:
            raise NodeNotFoundError(f"节点 '{node_id}' 不存在")
        # 删除所有入边
        for pred in self._reverse[node_id]:
            self._edges[pred].discard(node_id)
        # 删除所有出边
        for succ in self._edges[node_id]:
            self._reverse[succ].discard(node_id)
        self._nodes.discard(node_id)
        self._edges.pop(node_id, None)
        self._reverse.pop(node_id, None)

    # ── 查询 ────────────────────────────────────────────────

    @property
    def nodes(self) -> Set[str]:
        return set(self._nodes)

    def successors(self, node_id: str) -> Set[str]:
        return set(self._edges.get(node_id, set()))

    def predecessors(self, node_id: str) -> Set[str]:
        return set(self._reverse.get(node_id, set()))

    def roots(self) -> Set[str]:
        """返回入度为 0 的节点"""
        return {n for n in self._nodes if not self._reverse.get(n)}

    def leaves(self) -> Set[str]:
        """返回出度为 0 的节点"""
        return {n for n in self._nodes if not self._edges.get(n)}

    # ── 拓扑排序 ────────────────────────────────────────────

    def topological_sort(self) -> List[str]:
        """Kahn 算法拓扑排序"""
        in_degree: Dict[str, int] = {n: len(self._reverse.get(n, set())) for n in self._nodes}
        queue = deque([n for n, d in in_degree.items() if d == 0])
        result: List[str] = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for succ in self._edges.get(node, set()):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        if len(result) != len(self._nodes):
            raise CyclicDependencyError(
                "工作流存在循环依赖，无法执行"
            )
        return result

    # ── 并行层级 ────────────────────────────────────────────

    def parallel_layers(self) -> List[List[str]]:
        """按拓扑层级分组，同一层内的节点可以并行执行"""
        sorted_nodes = self.topological_sort()
        node_layer: Dict[str, int] = {}
        for node in sorted_nodes:
            preds = self.predecessors(node)
            if not preds:
                node_layer[node] = 0
            else:
                node_layer[node] = max(node_layer[p] for p in preds) + 1

        layers: Dict[int, List[str]] = defaultdict(list)
        for node, layer in node_layer.items():
            layers[layer].append(node)

        return [layers[i] for i in sorted(layers.keys())]

    # ── 验证 ────────────────────────────────────────────────

    def validate(self) -> List[DAGValidationError]:
        """执行全面验证，返回所有错误"""
        errors: List[DAGValidationError] = []

        # 1. 检测循环
        try:
            self.topological_sort()
        except CyclicDependencyError as e:
            errors.append(e)

        # 2. 检测孤立节点 (既无入边也无出边，且不是唯一的根)
        for node in self._nodes:
            has_in = bool(self._reverse.get(node))
            has_out = bool(self._edges.get(node))
            if not has_in and not has_out and len(self._nodes) > 1:
                errors.append(
                    OrphanNodeError(f"节点 '{node}' 是孤立节点（无入边也无出边）", node_id=node)
                )

        return errors

    # ── 工厂方法 ────────────────────────────────────────────

    @classmethod
    def from_edges(cls, nodes: List[str], edges: List[Tuple[str, str]]) -> "DAG":
        """从节点列表和边列表构建 DAG"""
        dag = cls()
        for n in nodes:
            dag.add_node(n)
        for src, tgt in edges:
            dag.add_edge(src, tgt)
        return dag
