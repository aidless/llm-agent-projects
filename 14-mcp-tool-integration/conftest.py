import pytest
import sys
import os

# 将项目目录加入 path
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

pytestmark = pytest.mark.asyncio