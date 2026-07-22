"""
快速启动脚本
提供非 Docker 方式的本地启动
"""
import subprocess
import sys
from pathlib import Path


def main():
    """主函数"""
    project_root = Path(__file__).parent
    print("=" * 60)
    print("  RAG 知识库问答系统 - 快速启动")
    print("=" * 60)

    # 1. 检查 .env 文件
    env_file = project_root / ".env"
    if not env_file.exists():
        print("[警告] 未找到 .env 文件，将使用默认配置")
        print("       请复制 .env.example 并修改配置")

    # 2. 创建数据目录
    dirs = ["data/uploads", "data/processed", "data/chroma_db"]
    for d in dirs:
        (project_root / d).mkdir(parents=True, exist_ok=True)
    print("[完成] 数据目录已创建")

    # 3. 启动服务
    print("[启动] 正在启动 FastAPI 服务...")
    print(f"       API 文档: http://localhost:8000/docs")
    print()

    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
    ], cwd=str(project_root))


if __name__ == "__main__":
    main()