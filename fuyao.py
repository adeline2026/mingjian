# -*- coding: utf-8 -*-
"""扶摇（同花顺金融数据服务）取数客户端。

设计要点
--------
所有失败都作为 **结果对象** 返回，而不是抛异常，也不静默吞掉。
原因：本产品的「未知证据」必须知道"为什么拿不到"——是没披露（3002）、
是没权限（2003）、是端内专用（2004）、还是网络挂了。如果这里吞掉错误，
下游就只能瞎猜，而题目明确禁止"静默生成正常结论"。
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

BASE_URL = os.environ.get("FUYAO_BASE_URL", "https://fuyao.aicubes.cn").rstrip("/")
CN_TZ = timezone(timedelta(hours=8))

CRED_PATHS = [
    Path.home() / ".hithink-finance" / "credentials.env",
    Path.home() / ".hithink-finance" / "credentials.env.txt",
]

# 可重试的业务码：限流与服务端异常
RETRYABLE_CODES = {4001, 5001, 5002, 5003}

# 业务码 -> 人话（含文档未收录、实测得到的 2004）
CODE_MEANING = {
    1001: "缺少必填参数",
    1002: "参数格式无效",
    1003: "参数超出范围",
    1004: "参数冲突",
    2001: "未认证",
    2003: "无权限或 Key 无效",
    2004: "该数据为同花顺AI客户端专用（端内专用，公开接口不可达）",
    3001: "标的不存在",
    3002: "数据尚未准备（未披露或上游无值）",
    3004: "目标类型不支持该能力",
    4001: "限流",
    5001: "服务端异常",
    5002: "服务端异常",
    5003: "上游异常",
}


def load_credential(name: str) -> str | None:
    """读取一个凭据：环境变量优先，其次用户级凭据文件。找不到返回 None。

    凭据文件刻意放在**用户目录**（~/.hithink-finance/credentials.env），
    而不是项目目录里 —— 这样即使把整个项目打包 / 压缩 / 分享出去，
    也不可能把密钥带走。
    """
    value = os.environ.get(name)
    if value and value.strip():
        return value.strip()
    for path in CRED_PATHS:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or not line.startswith(name):
                continue
            rest = line[len(name):].lstrip()
            if rest.startswith("="):
                return rest[1:].strip().strip('"').strip("'")
    return None


def load_api_key() -> str | None:
    """扶摇 API Key。"""
    return load_credential("HITHINK_FINANCE_API_KEY")


@dataclass
class FuyaoResult:
    """一次取数的完整记录。成功与失败走同一条路，调用方永远拿到它。"""

    endpoint: str
    params: dict
    ok: bool
    code: int | None = None          # 业务码；网络层失败为 None
    message: str = ""
    data: Any = None
    request_id: str | None = None
    error: str | None = None         # 网络层错误描述
    retrieved_at: str = ""
    attempts: int = 1

    @property
    def reason(self) -> str:
        """给「未知证据」用的人话原因。"""
        if self.ok:
            return ""
        if self.error:
            return f"请求失败：{self.error}"
        return f"接口返回 code={self.code}：{self.message or CODE_MEANING.get(self.code, '未知错误')}"

    def to_dict(self) -> dict:
        return {
            "endpoint": self.endpoint,
            "params": self.params,
            "ok": self.ok,
            "code": self.code,
            "message": self.message,
            "request_id": self.request_id,
            "error": self.error,
            "retrieved_at": self.retrieved_at,
            "attempts": self.attempts,
        }


def get(path: str, params: dict | None = None, api_key: str | None = None,
        timeout: int = 20, max_retries: int = 3) -> FuyaoResult:
    """发一次 GET。业务码 != 0 不算抛异常，而是失败结果。"""
    params = params or {}
    key = api_key or load_api_key()
    retrieved = datetime.now(CN_TZ).isoformat(timespec="seconds")

    if not key:
        return FuyaoResult(path, params, False, error="NO_API_KEY",
                           message="未找到 API Key（环境变量 HITHINK_FINANCE_API_KEY 或用户级凭据文件）",
                           retrieved_at=retrieved)

    query = ("?" + urllib.parse.urlencode(params)) if params else ""
    url = BASE_URL + path + query
    last_error = ""

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={"X-api-key": key})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 8))
                continue
            return FuyaoResult(path, params, False, error=last_error,
                               retrieved_at=retrieved, attempts=attempt)
        except Exception as exc:  # noqa: BLE001 - 网络层一切异常都归为可重试
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 8))
                continue
            return FuyaoResult(path, params, False, error=last_error,
                               retrieved_at=retrieved, attempts=attempt)

        code = payload.get("code")
        if code == 0:
            return FuyaoResult(path, params, True, 0, payload.get("message", ""),
                               payload.get("data"), payload.get("request_id"),
                               None, retrieved, attempt)

        if code in RETRYABLE_CODES and attempt < max_retries:
            time.sleep(min(2 ** attempt, 8))
            continue

        return FuyaoResult(path, params, False, code, payload.get("message", ""),
                           payload.get("data"), payload.get("request_id"),
                           None, retrieved, attempt)

    return FuyaoResult(path, params, False, error=last_error,
                       retrieved_at=retrieved, attempts=max_retries)


# ---------------------------------------------------------------- 业务封装

def search_ticker(q: str, limit: int = 10) -> FuyaoResult:
    return get("/api/meta/tickers/search", {"q": q, "limit": limit})


def price_snapshot(thscodes: list[str]) -> FuyaoResult:
    return get("/api/a-share/prices/snapshot", {"thscodes": ",".join(thscodes)})


def price_historical(thscode: str, **params) -> FuyaoResult:
    return get("/api/a-share/prices/historical", {"thscode": thscode, **params})


def valuations_snapshot(thscodes: list[str]) -> FuyaoResult:
    return get("/api/a-share/valuations/snapshot", {"thscodes": ",".join(thscodes)})


def financials_indicators(thscode: str, report: str) -> FuyaoResult:
    return get("/api/a-share/financials/indicators", {"thscode": thscode, "report": report})


def news_events(query: str, size: int = 5) -> FuyaoResult:
    """已知端内专用，保留此函数是为了把失败如实暴露成「未知证据」。"""
    return get("/api/news/events/search", {"query": query, "size": size})


# ---------------------------------------------------------------- 工具

def report_periods(year: int, quarter: int, count: int = 12) -> list[str]:
    """从 (year, quarter) 往前生成 count 个报告期，倒序返回（最新在前）。

    报告期格式 yyyy-N：1 一季报 / 2 中报 / 3 三季报 / 4 年报。
    同口径趋势必须用**同类型**报告期比较，所以默认取足够长（12 期 ≈ 3 年），
    让每一种报告期类型都有 >= 3 个观测点。
    """
    seq: list[str] = []
    y, q = year, quarter
    while len(seq) < count and y >= 1990:
        seq.append(f"{y}-{q}")
        q -= 1
        if q == 0:
            q, y = 4, y - 1
    return seq


def last_completed_period(now: datetime) -> tuple[int, int]:
    """按当前日期推断"最近一个已结束的季度"对应的报告期。

    注意：已结束 ≠ 已披露。未披露的期次会返回业务码，正好被下游如实记为「未知」。
    """
    m = now.month
    if m <= 3:
        return now.year - 1, 4
    if m <= 6:
        return now.year, 1
    if m <= 9:
        return now.year, 2
    return now.year, 3


def flatten_indicators(payload: Any) -> dict[str, Any]:
    """把 financials/indicators 的 abilities[] 摊平成 {index_id: value}。"""
    out: dict[str, Any] = {}
    if not isinstance(payload, dict):
        return out
    for block in payload.get("abilities") or []:
        for item in block.get("indicators") or []:
            idx = item.get("index_id")
            if idx:
                out[idx] = item.get("value")
    return out


def to_float(value: Any) -> float | None:
    """原始值是字符串。空白、null、非数字一律返回 None，不补零。"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s or s.lower() in {"null", "none", "nan", "--", "-"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None
