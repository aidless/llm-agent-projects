"""表格格式转换 - 支持多种输出格式。"""

import csv
import io
import json
import logging
from typing import Any, Dict, List, Optional

from app.models import TableData, TableOutputFormat, TableCell

logger = logging.getLogger(__name__)


class TableFormatter:
    """表格格式转换器。

    支持:
    - Markdown
    - HTML
    - JSON
    - CSV
    """

    def format_table(
        self,
        table: TableData,
        output_format: TableOutputFormat,
    ) -> str:
        """将表格数据转换为指定格式。

        Args:
            table: 表格数据。
            output_format: 目标格式。

        Returns:
            str: 格式化后的字符串。

        Raises:
            ValueError: 不支持的格式。
        """
        if output_format == TableOutputFormat.MARKDOWN:
            return self.to_markdown(table)
        elif output_format == TableOutputFormat.HTML:
            return self.to_html(table)
        elif output_format == TableOutputFormat.JSON:
            return self.to_json(table)
        elif output_format == TableOutputFormat.CSV:
            return self.to_csv(table)
        else:
            raise ValueError(f"不支持的格式: {output_format}")

    def to_markdown(self, table: TableData) -> str:
        """转换为 Markdown 格式。

        Args:
            table: 表格数据。

        Returns:
            str: Markdown 表格字符串。
        """
        if not table.cells:
            return ""

        # 构建二维矩阵
        matrix = self._build_matrix(table)
        if not matrix or not matrix[0]:
            return ""

        num_cols = len(matrix[0])
        lines = []

        # 表头
        header = matrix[0] if table.headers else matrix[0]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * num_cols) + " |")

        # 数据行
        for row in matrix[1:]:
            # 补齐列数
            while len(row) < num_cols:
                row.append("")
            lines.append("| " + " | ".join(row[:num_cols]) + " |")

        return "\n".join(lines)

    def to_html(self, table: TableData) -> str:
        """转换为 HTML 格式。

        Args:
            table: 表格数据。

        Returns:
            str: HTML 表格字符串。
        """
        matrix = self._build_matrix(table)
        if not matrix:
            return "<table></table>"

        num_cols = max(len(row) for row in matrix) if matrix else 0
        lines = ["<table>"]

        # 表头
        if table.headers:
            lines.append("  <thead>")
            lines.append("    <tr>")
            for h in table.headers:
                lines.append(f"      <th>{h}</th>")
            lines.append("    </tr>")
            lines.append("  </thead>")

        # 表体
        body_rows = matrix[1:] if table.headers else matrix
        if body_rows:
            lines.append("  <tbody>")
            for row in body_rows:
                while len(row) < num_cols:
                    row.append("")
                lines.append("    <tr>")
                for cell in row[:num_cols]:
                    lines.append(f"      <td>{cell}</td>")
                lines.append("    </tr>")
            lines.append("  </tbody>")

        lines.append("</table>")
        return "\n".join(lines)

    def to_json(self, table: TableData) -> str:
        """转换为 JSON 格式。

        Args:
            table: 表格数据。

        Returns:
            str: JSON 字符串。
        """
        matrix = self._build_matrix(table)
        if not matrix:
            return "[]"

        headers = table.headers if table.headers else matrix[0]
        rows_data = []

        for row in matrix[1:] if table.headers else matrix:
            row_dict = {}
            for i, h in enumerate(headers):
                row_dict[h] = row[i] if i < len(row) else ""
            rows_data.append(row_dict)

        return json.dumps(rows_data, ensure_ascii=False, indent=2)

    def to_csv(self, table: TableData) -> str:
        """转换为 CSV 格式。

        Args:
            table: 表格数据。

        Returns:
            str: CSV 字符串。
        """
        matrix = self._build_matrix(table)
        if not matrix:
            return ""

        output = io.StringIO()
        writer = csv.writer(output)
        for row in matrix:
            writer.writerow(row)

        return output.getvalue()

    @staticmethod
    def _build_matrix(table: TableData) -> List[List[str]]:
        """将表格数据构建为二维字符串矩阵。"""
        if not table.cells:
            return []

        num_rows = table.rows
        num_cols = table.cols

        if num_rows == 0 or num_cols == 0:
            return []

        # 初始化矩阵
        matrix = [["" for _ in range(num_cols)] for _ in range(num_rows)]

        for cell in table.cells:
            if 0 <= cell.row < num_rows and 0 <= cell.col < num_cols:
                matrix[cell.row][cell.col] = cell.text

        return matrix

    def validate_table(self, table: TableData) -> Dict[str, Any]:
        """验证表格数据完整性。

        Args:
            table: 表格数据。

        Returns:
            dict: 验证结果，包含 errors 和 warnings。
        """
        errors = []
        warnings = []

        if table.rows == 0:
            errors.append("表格行数为 0")
        if table.cols == 0:
            errors.append("表格列数为 0")
        if not table.cells and table.rows > 0:
            warnings.append("表格有行列定义但无单元格数据")

        # 检查表头
        if table.rows > 0 and not table.headers:
            warnings.append("未检测到表头")

        # 检查空单元格
        if table.cells:
            empty_count = sum(1 for c in table.cells if not c.text.strip())
            total = len(table.cells)
            if empty_count / total > 0.5:
                warnings.append(f"超过 50% 的单元格为空 ({empty_count}/{total})")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }