"""DAG 模块测试"""

import pytest
from engine.dag import DAG
from engine.errors import CyclicDependencyError, OrphanNodeError, NodeNotFoundError


class TestDAG:
    """DAG 基本操作测试"""

    def test_add_and_query_nodes(self):
        dag = DAG()
        dag.add_node("a")
        dag.add_node("b")
        assert dag.nodes == {"a", "b"}

    def test_add_edge(self):
        dag = DAG()
        dag.add_node("a")
        dag.add_node("b")
        dag.add_edge("a", "b")
        assert dag.successors("a") == {"b"}
        assert dag.predecessors("b") == {"a"}

    def test_add_edge_missing_node_raises(self):
        dag = DAG()
        dag.add_node("a")
        with pytest.raises(NodeNotFoundError):
            dag.add_edge("a", "nonexistent")

    def test_remove_node(self):
        dag = DAG()
        dag.add_node("a")
        dag.add_node("b")
        dag.add_edge("a", "b")
        dag.remove_node("a")
        assert "a" not in dag.nodes
        assert dag.predecessors("b") == set()

    def test_remove_nonexistent_node_raises(self):
        dag = DAG()
        with pytest.raises(NodeNotFoundError):
            dag.remove_node("x")

    def test_roots_and_leaves(self):
        dag = DAG()
        dag.add_node("a")
        dag.add_node("b")
        dag.add_node("c")
        dag.add_edge("a", "b")
        dag.add_edge("b", "c")
        assert dag.roots() == {"a"}
        assert dag.leaves() == {"c"}

    def test_single_node_is_root_and_leaf(self):
        dag = DAG()
        dag.add_node("solo")
        assert dag.roots() == {"solo"}
        assert dag.leaves() == {"solo"}


class TestDAGTopologicalSort:
    """拓扑排序测试"""

    def test_simple_chain(self):
        dag = DAG.from_edges(["a", "b", "c"], [("a", "b"), ("b", "c")])
        order = dag.topological_sort()
        assert order.index("a") < order.index("b") < order.index("c")

    def test_parallel_branches(self):
        dag = DAG()
        dag.add_node("root")
        dag.add_node("branch1")
        dag.add_node("branch2")
        dag.add_node("merge")
        dag.add_edge("root", "branch1")
        dag.add_edge("root", "branch2")
        dag.add_edge("branch1", "merge")
        dag.add_edge("branch2", "merge")
        order = dag.topological_sort()
        assert order.index("root") < order.index("branch1")
        assert order.index("root") < order.index("branch2")
        assert order.index("branch1") < order.index("merge")
        assert order.index("branch2") < order.index("merge")

    def test_cycle_raises(self):
        dag = DAG()
        dag.add_node("a")
        dag.add_node("b")
        dag.add_node("c")
        dag.add_edge("a", "b")
        dag.add_edge("b", "c")
        dag.add_edge("c", "a")
        with pytest.raises(CyclicDependencyError):
            dag.topological_sort()

    def test_self_loop_raises(self):
        dag = DAG()
        dag.add_node("a")
        dag.add_edge("a", "a")
        with pytest.raises(CyclicDependencyError):
            dag.topological_sort()


class TestDAGValidation:
    """DAG 验证测试"""

    def test_valid_dag_no_errors(self):
        dag = DAG.from_edges(["a", "b"], [("a", "b")])
        errors = dag.validate()
        assert len(errors) == 0

    def test_orphan_node_detected(self):
        dag = DAG()
        dag.add_node("a")
        dag.add_node("b")
        dag.add_node("orphan")
        dag.add_edge("a", "b")
        errors = dag.validate()
        assert any(isinstance(e, OrphanNodeError) for e in errors)

    def test_cyclic_dependency_detected(self):
        dag = DAG()
        dag.add_node("a")
        dag.add_node("b")
        dag.add_edge("a", "b")
        dag.add_edge("b", "a")
        errors = dag.validate()
        assert any(isinstance(e, CyclicDependencyError) for e in errors)


class TestDAGParallelLayers:
    """并行层级测试"""

    def test_chain_is_single_layer_per_step(self):
        dag = DAG.from_edges(["a", "b", "c"], [("a", "b"), ("b", "c")])
        layers = dag.parallel_layers()
        assert len(layers) == 3
        assert layers == [["a"], ["b"], ["c"]]

    def test_parallel_nodes_same_layer(self):
        dag = DAG()
        dag.add_node("root")
        dag.add_node("p1")
        dag.add_node("p2")
        dag.add_node("merge")
        dag.add_edge("root", "p1")
        dag.add_edge("root", "p2")
        dag.add_edge("p1", "merge")
        dag.add_edge("p2", "merge")
        layers = dag.parallel_layers()
        assert len(layers) == 3
        assert "p1" in layers[1]
        assert "p2" in layers[1]
