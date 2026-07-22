"""文档解析 API。"""

import io
import os
import tempfile
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.models import APIResponse, DocumentFormat, ParseRequest, ParsedDocument
from parsers.image_parser import ImageParser
from parsers.pdf_parser import PDFParser
from parsers.text_parser import TextParser

router = APIRouter(prefix="/parse", tags=["文档解析"])


def _get_file_extension(filename: str) -> str:
    return os.path.splitext(filename)[1].lower()


@router.post("/upload", response_model=APIResponse)
async def parse_upload(
    file: UploadFile = File(...),
    ocr_enabled: bool = Form(default=True),
    table_extraction: bool = Form(default=True),
    layout_analysis: bool = Form(default=False),
):
    """上传并解析文档。"""
    filename = file.filename or "unknown"
    suffix = _get_file_extension(filename)

    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = _parse_file(
            tmp_path, filename,
            ocr_enabled=ocr_enabled,
            table_extraction=table_extraction,
            layout_analysis=layout_analysis,
        )
        return APIResponse(
            success=True,
            message="解析成功",
            data=result.model_dump(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post("/request", response_model=APIResponse)
async def parse_request(req: ParseRequest):
    """通过请求参数解析文档。"""
    return APIResponse(
        success=True,
        message="请使用 /parse/upload 上传文件进行解析",
        data=req.model_dump(),
    )


def _parse_file(
    file_path: str,
    filename: str,
    ocr_enabled: bool = True,
    table_extraction: bool = True,
    layout_analysis: bool = False,
) -> ParsedDocument:
    """根据文件类型选择解析器。"""
    suffix = _get_file_extension(filename)

    if suffix == ".pdf":
        parser = PDFParser(enable_ocr_fallback=ocr_enabled)
        return parser.parse(file_path)
    elif suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}:
        img_parser = ImageParser()
        return img_parser.parse(file_path)
    elif suffix in {".txt", ".md"}:
        text_parser = TextParser()
        return text_parser.parse(file_path)
    elif suffix == ".docx":
        from parsers.docx_parser import DocxParser
        docx_parser = DocxParser()
        return docx_parser.parse(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {suffix}")