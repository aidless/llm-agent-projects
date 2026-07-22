"""Diff 解析器。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DiffHunk:
    """单个 Diff 块。"""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    content: str
    added_lines: list[str] = field(default_factory=list)
    removed_lines: list[str] = field(default_factory=list)


@dataclass
class DiffFile:
    """单个文件的 Diff 信息。"""

    old_path: str
    new_path: str
    hunks: list[DiffHunk] = field(default_factory=list)
    is_new: bool = False
    is_deleted: bool = False
    is_renamed: bool = False
    is_binary: bool = False


class DiffParser:
    """解析 unified diff 格式的差异内容。"""

    HUNK_HEADER = re.compile(
        r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@"
    )

    FILE_HEADER = re.compile(
        r"^diff --git a/(.+?) b/(.+)$"
    )

    OLD_FILE = re.compile(r"^---\s+(?:a/)?(.+)")
    NEW_FILE = re.compile(r"^\+\+\+\s+(?:b/)?(.+)")

    NEW_FILE_MODE = re.compile(r"^new file mode")
    DELETED_FILE_MODE = re.compile(r"^deleted file mode")
    RENAME_FROM = re.compile(r"^rename from (.+)")
    RENAME_TO = re.compile(r"^rename to (.+)")
    BINARY = re.compile(r"^Binary files")

    def parse(self, diff_content: str) -> list[DiffFile]:
        """解析 diff 内容。

        Args:
            diff_content: unified diff 格式的文本。

        Returns:
            DiffFile 列表。
        """
        files: list[DiffFile] = []
        current_file: Optional[DiffFile] = None
        current_hunk: Optional[DiffHunk] = None

        lines = diff_content.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]

            # 检查文件头
            file_match = self.FILE_HEADER.match(line)
            if file_match:
                current_file = DiffFile(
                    old_path=file_match.group(1),
                    new_path=file_match.group(2),
                )
                files.append(current_file)
                current_hunk = None
                i += 1
                continue

            if current_file is None:
                i += 1
                continue

            # 检查文件元信息
            if self.NEW_FILE_MODE.match(line):
                current_file.is_new = True
                i += 1
                continue

            if self.DELETED_FILE_MODE.match(line):
                current_file.is_deleted = True
                i += 1
                continue

            if self.BINARY.match(line):
                current_file.is_binary = True
                i += 1
                continue

            if self.RENAME_FROM.match(line):
                current_file.is_renamed = True
                i += 1
                continue

            # 检查 hunk 头
            hunk_match = self.HUNK_HEADER.match(line)
            if hunk_match:
                current_hunk = DiffHunk(
                    old_start=int(hunk_match.group(1)),
                    old_count=int(hunk_match.group(2)) if hunk_match.group(2) else 1,
                    new_start=int(hunk_match.group(3)),
                    new_count=int(hunk_match.group(4)) if hunk_match.group(4) else 1,
                    content=line,
                )
                if current_file:
                    current_file.hunks.append(current_hunk)
                i += 1
                continue

            # 检查 diff 行
            if current_hunk is not None:
                if line.startswith("+") and not line.startswith("+++"):
                    current_hunk.added_lines.append(line[1:])
                elif line.startswith("-") and not line.startswith("---"):
                    current_hunk.removed_lines.append(line[1:])
                elif not line.startswith("\\"):
                    # 普通行（空格或无前缀）
                    pass

            i += 1

        return files

    def extract_added_code(self, diff_content: str) -> str:
        """从 diff 中提取新增的代码。

        Args:
            diff_content: unified diff 格式的文本。

        Returns:
            仅包含新增行的代码字符串。
        """
        files = self.parse(diff_content)
        added_lines: list[str] = []

        for diff_file in files:
            if diff_file.is_binary:
                continue
            for hunk in diff_file.hunks:
                added_lines.extend(hunk.added_lines)

        return "\n".join(added_lines)

    def get_changed_files(self, diff_content: str) -> list[str]:
        """获取变更的文件列表。

        Args:
            diff_content: unified diff 格式的文本。

        Returns:
            变更的文件路径列表。
        """
        files = self.parse(diff_content)
        return [f.new_path for f in files]
