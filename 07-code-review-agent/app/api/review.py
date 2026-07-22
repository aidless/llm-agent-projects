"""代码审查 API 路由。"""

from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException

from app.models import (
    CodeSubmitRequest,
    GitHubPRRequest,
    ReviewResponse,
    ReviewStatus,
)
from app.store import review_store
from workflows.review_graph import create_review_graph, ReviewState

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/review/code", response_model=ReviewResponse)
async def review_code(request: CodeSubmitRequest) -> ReviewResponse:
    """提交代码进行审查。

    接收代码内容，通过 LangGraph 工作流并行执行多个 Agent 审查，
    汇总后返回结构化的审查报告。
    """
    start_time = time.time()
    logger.info(f"收到代码审查请求: filename={request.filename}, language={request.language}")

    try:
        graph = create_review_graph()

        state = ReviewState(
            code=request.code,
            language=request.language,
            filename=request.filename,
            context=request.context or "",
        )

        result = await graph.ainvoke(state)

        elapsed = time.time() - start_time

        # 构建响应
        total_findings = sum(len(r.get("findings", [])) for r in result.get("agent_results", []))

        response = ReviewResponse(
            status=ReviewStatus.COMPLETED,
            results=result.get("agent_results", []),
            total_findings=total_findings,
            report_markdown=result.get("report_markdown", ""),
            report_json=result.get("report_json", {}),
            language=request.language,
            filename=request.filename,
        )

        # 存储审查结果
        review_store[response.review_id] = response.model_dump()

        logger.info(f"审查完成: review_id={response.review_id}, findings={total_findings}, elapsed={elapsed:.2f}s")
        return response

    except SyntaxError as e:
        logger.error(f"代码语法错误: {e}")
        raise HTTPException(status_code=422, detail=f"代码语法错误: {str(e)}")
    except Exception as e:
        logger.error(f"审查失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"审查失败: {str(e)}")


@router.post("/review/github", response_model=ReviewResponse)
async def review_github_pr(request: GitHubPRRequest) -> ReviewResponse:
    """审查 GitHub PR。

    通过 GitHub API 获取 PR 的 diff 内容，执行代码审查，
    可选自动在 PR 上发表审查评论。
    """
    start_time = time.time()
    logger.info(
        f"收到 GitHub PR 审查请求: {request.repo_owner}/{request.repo_name}#{request.pr_number}"
    )

    try:
        from github.client import GitHubClient
        from github.reviewer import PRReviewer

        client = GitHubClient(token=request.github_token)
        reviewer = PRReviewer(client)

        # 获取 PR diff
        diff_content = await client.get_pr_diff(
            owner=request.repo_owner,
            repo=request.repo_name,
            pr_number=request.pr_number,
        )

        if not diff_content:
            raise HTTPException(status_code=404, detail="无法获取 PR diff 内容")

        # 使用第一个文件作为 filename
        files = await client.get_pr_files(
            owner=request.repo_owner,
            repo=request.repo_name,
            pr_number=request.pr_number,
        )
        filename = files[0]["filename"] if files else "changes.patch"

        # 执行审查
        graph = create_review_graph()
        state = ReviewState(
            code=diff_content,
            language="python",
            filename=filename,
            context=f"GitHub PR: {request.repo_owner}/{request.repo_name}#{request.pr_number}",
        )

        result = await graph.ainvoke(state)

        elapsed = time.time() - start_time
        total_findings = sum(len(r.get("findings", [])) for r in result.get("agent_results", []))

        response = ReviewResponse(
            status=ReviewStatus.COMPLETED,
            results=result.get("agent_results", []),
            total_findings=total_findings,
            report_markdown=result.get("report_markdown", ""),
            report_json=result.get("report_json", {}),
            language="python",
            filename=filename,
        )

        review_store[response.review_id] = response.model_dump()

        # 可选：自动发表评论
        if request.post_comments:
            await reviewer.post_review_comments(
                owner=request.repo_owner,
                repo=request.repo_name,
                pr_number=request.pr_number,
                findings=response.results,
            )

        logger.info(f"GitHub PR 审查完成: review_id={response.review_id}, elapsed={elapsed:.2f}s")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GitHub PR 审查失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"GitHub PR 审查失败: {str(e)}")
