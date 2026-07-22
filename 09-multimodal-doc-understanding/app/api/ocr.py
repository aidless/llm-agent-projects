"""OCR API。"""

import io
import tempfile
import os
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models import APIResponse, OCRRequest, OCRResult, OCREngineType
from ocr.engine import OCREngine
from ocr.preprocessor import ImagePreprocessor
from utils.image_utils import load_image

router = APIRouter(prefix="/ocr", tags=["OCR"])


@router.post("/recognize", response_model=APIResponse)
async def ocr_recognize(
    file: UploadFile = File(...),
    languages: str = "chi_sim+eng",
    engine: str = "tesseract",
    preprocess: bool = True,
):
    """对上传图片进行 OCR 识别。"""
    lang_list = languages.split("+")
    preprocessor = ImagePreprocessor() if preprocess else None

    try:
        ocr_engine = OCREngine(
            engine_type=engine,
            preprocessor=preprocessor,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 读取图片
    content = await file.read()
    try:
        image = load_image(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法读取图片: {e}")

    try:
        result = ocr_engine.recognize(image, languages=lang_list)
        return APIResponse(
            success=True,
            message="OCR 识别完成",
            data=result.model_dump(),
        )
    except ImportError as e:
        # Tesseract 未安装
        return APIResponse(
            success=False,
            message=str(e),
            data=None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR 识别失败: {e}")