"""Pydantic 数据模型定义。"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ============ 枚举类型 ============

class DocumentFormat(str, Enum):
    PDF = "pdf"
    IMAGE = "image"
    DOCX = "docx"
    TEXT = "text"
    MARKDOWN = "markdown"


class OCREngineType(str, Enum):
    TESSERACT = "tesseract"
    API = "api"


class RegionType(str, Enum):
    TITLE = "title"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    IMAGE = "image"
    HEADER = "header"
    FOOTER = "footer"
    LIST = "list"
    CODE = "code"


class TableOutputFormat(str, Enum):
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    CSV = "csv"


# ============ 基础模型 ============

class BoundingBox(BaseModel):
    """边界框。"""
    x: int = Field(..., description="左上角 x 坐标")
    y: int = Field(..., description="左上角 y 坐标")
    width: int = Field(..., description="宽度")
    height: int = Field(..., description="高度")


class OCRWord(BaseModel):
    """OCR 识别出的单个词/字。"""
    text: str = Field(..., description="识别文本")
    confidence: float = Field(..., description="置信度 0-1")
    bbox: Optional[BoundingBox] = None


class OCRResult(BaseModel):
    """OCR 识别结果。"""
    text: str = Field(..., description="完整识别文本")
    words: List[OCRWord] = Field(default_factory=list, description="逐词结果")
    confidence: float = Field(default=0.0, description="平均置信度")
    engine: str = Field(default="unknown", description="使用的 OCR 引擎")


class DocumentRegion(BaseModel):
    """文档区域。"""
    region_type: RegionType = Field(..., description="区域类型")
    bbox: BoundingBox = Field(..., description="边界框")
    content: str = Field(default="", description="区域内容")
    confidence: float = Field(default=1.0, description="置信度")
    page: int = Field(default=1, description="页码")
    children: List["DocumentRegion"] = Field(default_factory=list, description="子区域")


class LayoutAnalysisResult(BaseModel):
    """版面分析结果。"""
    regions: List[DocumentRegion] = Field(default_factory=list, description="检测到的区域")
    page_count: int = Field(default=1, description="页数")
    reading_order: List[int] = Field(default_factory=list, description="阅读顺序索引")


class TableCell(BaseModel):
    """表格单元格。"""
    row: int = Field(..., description="行号")
    col: int = Field(..., description="列号")
    text: str = Field(default="", description="单元格文本")
    rowspan: int = Field(default=1, description="跨行数")
    colspan: int = Field(default=1, description="跨列数")


class TableData(BaseModel):
    """表格数据。"""
    rows: int = Field(..., description="行数")
    cols: int = Field(..., description="列数")
    cells: List[TableCell] = Field(default_factory=list, description="单元格列表")
    headers: List[str] = Field(default_factory=list, description="表头")


class TableExtractionResult(BaseModel):
    """表格提取结果。"""
    tables: List[TableData] = Field(default_factory=list, description="提取到的表格")
    page: int = Field(default=1, description="页码")


class ParsedPage(BaseModel):
    """解析后的单页。"""
    page_num: int = Field(..., description="页码")
    text: str = Field(default="", description="提取的文本")
    images: List[str] = Field(default_factory=list, description="提取的图片 (base64)")
    tables: List[TableData] = Field(default_factory=list, description="检测到的表格")
    ocr_result: Optional[OCRResult] = None
    layout: Optional[LayoutAnalysisResult] = None


class ParsedDocument(BaseModel):
    """解析后的完整文档。"""
    filename: str = Field(..., description="文件名")
    format: DocumentFormat = Field(..., description="文档格式")
    pages: List[ParsedPage] = Field(default_factory=list, description="页面列表")
    full_text: str = Field(default="", description="全文文本")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class VisionDescription(BaseModel):
    """图片描述结果。"""
    description: str = Field(..., description="图片描述")
    confidence: float = Field(default=1.0, description="置信度")
    labels: List[str] = Field(default_factory=list, description="标签列表")


class ChartUnderstanding(BaseModel):
    """图表理解结果。"""
    chart_type: str = Field(default="unknown", description="图表类型")
    description: str = Field(default="", description="图表描述")
    data_summary: Dict[str, Any] = Field(default_factory=dict, description="数据摘要")
    key_insights: List[str] = Field(default_factory=list, description="关键洞察")


class VQAResult(BaseModel):
    """视觉问答结果。"""
    question: str = Field(..., description="问题")
    answer: str = Field(..., description="答案")
    confidence: float = Field(default=1.0, description="置信度")
    sources: List[str] = Field(default_factory=list, description="引用来源")


class QAResult(BaseModel):
    """文档问答结果。"""
    question: str = Field(..., description="问题")
    answer: str = Field(..., description="答案")
    sources: List[str] = Field(default_factory=list, description="引用来源")
    score: float = Field(default=0.0, description="相关度评分")


class DocumentIndexResult(BaseModel):
    """文档索引结果。"""
    document_id: str = Field(..., description="文档 ID")
    filename: str = Field(..., description="文件名")
    chunks_count: int = Field(..., description="分块数量")
    status: str = Field(default="success", description="状态")


# ============ 请求/响应模型 ============

class ParseRequest(BaseModel):
    """文档解析请求。"""
    file_url: Optional[str] = Field(None, description="文件 URL (可选)")
    ocr_enabled: bool = Field(default=True, description="是否启用 OCR")
    table_extraction: bool = Field(default=True, description="是否提取表格")
    layout_analysis: bool = Field(default=False, description="是否进行版面分析")


class OCRRequest(BaseModel):
    """OCR 识别请求。"""
    languages: List[str] = Field(default=["chi_sim", "eng"], description="OCR 语言")
    engine: OCREngineType = Field(default=OCREngineType.TESSERACT, description="OCR 引擎")
    preprocess: bool = Field(default=True, description="是否预处理")


class TableRequest(BaseModel):
    """表格提取请求。"""
    output_format: TableOutputFormat = Field(
        default=TableOutputFormat.MARKDOWN, description="输出格式"
    )
    page: int = Field(default=0, description="页码, 0 表示所有页")


class QARequest(BaseModel):
    """问答请求。"""
    question: str = Field(..., description="问题")
    document_ids: Optional[List[str]] = Field(None, description="限定文档 ID")
    top_k: int = Field(default=3, description="返回 top-k 个结果")


class APIResponse(BaseModel):
    """统一 API 响应。"""
    success: bool = Field(default=True)
    message: str = Field(default="ok")
    data: Optional[Dict[str, Any]] = None