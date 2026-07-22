"""工作流引擎自定义错误"""


class WorkflowError(Exception):
    """工作流基础错误"""
    def __init__(self, message: str, node_id: str = None):
        self.node_id = node_id
        super().__init__(message)


class DAGValidationError(WorkflowError):
    """DAG 验证错误"""
    pass


class CyclicDependencyError(DAGValidationError):
    """循环依赖错误"""
    pass


class OrphanNodeError(DAGValidationError):
    """孤立节点错误 (既无入边也无出边)"""
    pass


class MissingConnectionError(DAGValidationError):
    """缺失连接错误"""
    pass


class NodeExecutionError(WorkflowError):
    """节点执行错误"""
    pass


class VariableResolutionError(WorkflowError):
    """变量解析错误"""
    pass


class WorkflowExecutionError(WorkflowError):
    """工作流执行错误"""
    pass


class WorkflowNotFoundError(WorkflowError):
    """工作流未找到错误"""
    pass


class NodeNotFoundError(WorkflowError):
    """节点未找到错误"""
    pass


class TimeoutError(WorkflowError):
    """执行超时错误"""
    pass
