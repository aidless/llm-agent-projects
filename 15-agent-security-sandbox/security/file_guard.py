"""文件系统守卫 - 控制沙箱中的文件读写操作"""

import os
from typing import Set, Optional
from dataclasses import dataclass, field


@dataclass
class FileGuardConfig:
    """文件系统守卫配置"""
    allowed_read_dirs: Set[str] = field(default_factory=set)
    allowed_write_dirs: Set[str] = field(default_factory=set)
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    block_all_writes: bool = True
    block_all_reads: bool = False


class FileGuard:
    """文件系统访问守卫"""

    def __init__(self, config: Optional[FileGuardConfig] = None):
        self.config = config or FileGuardConfig()

    def check_read(self, filepath: str) -> bool:
        """检查是否允许读取文件"""
        if self.config.block_all_reads:
            return False
        if not self.config.allowed_read_dirs:
            return True  # 如果没有设置白名单，默认允许
        return self._is_in_allowed_dirs(filepath, self.config.allowed_read_dirs)

    def check_write(self, filepath: str) -> bool:
        """检查是否允许写入文件"""
        if self.config.block_all_writes:
            return False
        if not self.config.allowed_write_dirs:
            return False  # 写入默认不允许
        return self._is_in_allowed_dirs(filepath, self.config.allowed_write_dirs)

    def check_file_size(self, size: int) -> bool:
        """检查文件大小是否在限制内"""
        return 0 <= size <= self.config.max_file_size

    def _is_in_allowed_dirs(self, filepath: str, allowed_dirs: Set[str]) -> bool:
        """检查路径是否在允许的目录中"""
        abs_path = os.path.abspath(filepath)
        for allowed_dir in allowed_dirs:
            abs_dir = os.path.abspath(allowed_dir)
            if abs_path.startswith(abs_dir + os.sep) or abs_path == abs_dir:
                return True
        return False

    def sanitize_path(self, filepath: str) -> str:
        """清理路径，防止目录遍历"""
        abs_path = os.path.abspath(filepath)
        # 检查路径遍历尝试
        if ".." in filepath or filepath.startswith("/"):
            if not self.check_read(filepath):
                raise PermissionError(f"Path access denied: {filepath}")
        return abs_path