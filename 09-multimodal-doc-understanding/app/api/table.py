"""表格识别 API。"""

import io
import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.models import APIResponse, TableOutputFormat, TableRequest
from table.detector import TableDetector
from table.extractor import TableExtractor
from table.formatter import TableFormatter
from utils.image_utils import load_image

router = APIRouter(prefix="/table", tags=["表格识别"])


@router.post("/extract", response_model=APIResponse)
async def table_extract(
    file: UploadFile = File(...),
    output_format: str = Form(default="markdown"),
):
    """从上传图片中提取表格。"""
    content = await file.read()

    try:
        image = load_image(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法读取图片: {e}")

    # 检测并提取表格
    detector = TableDetector()
    extractor = TableExtractor(detector=detector)
    result = extractor.extract(image)

    if not result.tables:
        return APIResponse(
            success=True,
            message="未检测到表格",
            data={"tables": [], "count": 0},
        )

    # 格式化输出
    fmt = TableFormatter()
    format_enum = TableOutputFormat(output_format)
    formatted_tables = []

    for table in result.tables:
        try:
            formatted = fmt.format_table(table, format_enum)
            formatted_tables.append(formatted)
        except Exception as e:
            formatted_tables.append(f"[格式化失败: {e}]")

    # 验证表格
    validation_results = []
    for table in result.tables:
        validation_results.append(fmt.validate_table(table))

    return APIResponse(
        success=True,
        message=f"检测到 {len(result.tables)} 个表格",
        data={
            "count": len(result.tables),
            "format": output_format,
            "tables": formatted_tables,
            "validation": validation_results,
        },
    )