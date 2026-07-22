"""PR Review 操作模块。"""

from __future__ import annotations

import logging
from typing import Any, Optional

from github.client import GitHubClient

logger = logging.getLogger(__name__)


class PRReviewer:
    """PR Review 操作器。

    负责将审查结果发布为 GitHub PR 的 review comments。
    """

    def __init__(self, client: GitHubClient) -> None:
        """初始化 PR Reviewer。

        Args:
            client: GitHub API 客户端。
        """
        self._client = client

    async def post_review_comments(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        findings: list[Any],
        commit_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """将审查发现发布为 PR review comments。

        Args:
            owner: 仓库所有者。
            repo: 仓库名称。
            pr_number: PR 编号。
            findings: 审查结果列表（ReviewResult 格式）。
            commit_id: commit SHA。

        Returns:
            发布的评论列表。
        """
        comments: list[dict[str, Any]] = []

        for result in findings:
            agent_name = getattr(result, "agent_name", "unknown")
            for finding in result.findings if hasattr(result, "findings") else []:
                body = self._format_finding_comment(agent_name, finding)
                comment = await self._client.create_review_comment(
                    owner=owner,
                    repo=repo,
                    pr_number=pr_number,
                    body=body,
                    path=finding.file,
                    line=finding.line,
                    commit_id=commit_id,
                )
                comments.append(comment)
                logger.info(
                    f"已发布评论: {finding.file}:{finding.line} - "
                    f"[{finding.severity.value}] {finding.message[:50]}"
                )

        return comments

    @staticmethod
    def _format_finding_comment(agent_name: str, finding: Any) -> str:
        """格式化单条审查发现为评论内容。

        Args:
            agent_name: Agent 名称。
            finding: Finding 对象。

        Returns:
            Markdown 格式的评论字符串。
        """
        severity_emoji = {
            "Critical": ":rotating_light:",
            "High": ":warning:",
            "Medium": ":eyes:",
            "Low": ":memo:",
            "Info": ":information_source:",
        }

        emoji = severity_emoji.get(finding.severity.value, ":speech_balloon:")

        parts = [
            f"**{emoji} [{agent_name}] {finding.severity.value}**",
            "",
            finding.message,
        ]

        if finding.suggestion:
            parts.append("")
            parts.append(f"**建议:** {finding.suggestion}")

        if finding.code_snippet:
            parts.append("")
            parts.append(f"```python\n{finding.code_snippet}\n```")

        if finding.rule_id:
            parts.append("")
            parts.append(f"规则: `{finding.rule_id}`")

        return "\n".join(parts)

    async def post_summary_comment(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        summary: str,
    ) -> dict[str, Any]:
        """发布审查摘要评论。

        Args:
            owner: 仓库所有者。
            repo: 仓库名称。
            pr_number: PR 编号。
            summary: 摘要内容。

        Returns:
            API 响应。
        """
        return await self._client.create_review_comment(
            owner=owner,
            repo=repo,
            pr_number=pr_number,
            body=summary,
            path="",
            line=1,
        )
