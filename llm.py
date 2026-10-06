# -*- coding: utf-8 -*-
"""推断层：本产品里**唯一**允许调用大模型的地方，且被严格约束。

约束设计（这是「合理分工」的落地，也是防幻觉的闸门）
------------------------------------------------------
1. **输入只有事实**：未知项和矛盾项不进上下文，模型看不到未证实的东西。
2. **必须引用**：每条推断必须带至少一个事实编号；引用不存在或引用到非事实的，
   **整条丢弃**（降级为「未知」），而不是原样展示。
3. **输出结构化**：只接受 JSON，不接受自由文本。
4. **不启用也能跑**：没有配置 Key 时，本层如实产出「推断层未启用」的未知证据，
   产品其余部分完全不受影响。

兼容性：任何 OpenAI 兼容的 chat/completions 端点都能用，只改环境变量。
    DeepSeek   LLM_BASE_URL=https://api.deepseek.com/v1          LLM_MODEL=deepseek-chat
    智谱 GLM   LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4 LLM_MODEL=glm-4-flash
    通义千问    LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 LLM_MODEL=qwen-turbo
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import fuyao

SYSTEM_PROMPT = """你是一个受严格约束的金融文本解读器。你的输出会被程序校验，违反任何一条即被整条丢弃。

规则：
1. 只能使用「事实清单」中列出的信息。不得引入任何外部知识，不得推算未列出的数字。
2. 每条解读必须引用至少一个事实编号（形如 E-01）。
3. 禁止给出买卖建议、涨跌预测、目标价、收益承诺。
4. 禁止把事实清单中未标明方向的内容说成正向或负向。
5. 只输出 JSON，不要任何解释、不要 markdown 代码块。

输出格式：
{"inferences":[{"claim":"一句话解读，不超过60字","cites":["E-01","E-03"]}]}

如果没有可以负责地给出的解读，就返回 {"inferences":[]}。宁可少说，不要多说。"""


@dataclass
class LLMConfig:
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout: int = 40
    provider: str = ""

    @classmethod
    def from_env(cls) -> "LLMConfig":
        # 与扶摇 Key 一样：环境变量优先，其次用户级凭据文件。
        # 项目目录里不放任何真密钥。
        base = (os.environ.get("LLM_BASE_URL") or fuyao.load_credential("LLM_BASE_URL") or "").strip().rstrip("/")
        key = (os.environ.get("LLM_API_KEY") or fuyao.load_credential("LLM_API_KEY") or "").strip()
        model = (os.environ.get("LLM_MODEL") or fuyao.load_credential("LLM_MODEL") or "").strip()
        if key and not base:
            base, model = "https://api.deepseek.com/v1", model or "deepseek-chat"
        host = base.split("//")[-1].split("/")[0] if base else ""
        return cls(base_url=base, api_key=key, model=model or "deepseek-chat",
                   provider=host)

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)


@dataclass
class InferenceOutcome:
    inferences: list[dict] = field(default_factory=list)
    status: str = ""          # ok / disabled / error / empty
    detail: str = ""
    raw: str = ""
    dropped: list[dict] = field(default_factory=list)
    provider: str = ""
    model: str = ""


def _post_chat(cfg: LLMConfig, messages: list[dict],
               json_mode: bool = True) -> tuple[bool, Any, str]:
    url = f"{cfg.base_url}/chat/completions"
    payload: dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "temperature": 0,
    }
    if json_mode:
        # 并非所有 OpenAI 兼容厂商都支持这个参数；不支持时由调用方退回普通模式
        payload["response_format"] = {"type": "json_object"}

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {cfg.api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
            return True, json.loads(resp.read().decode("utf-8")), ""
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", "ignore")[:300]
        except Exception:
            detail = ""
        return False, None, f"HTTP {exc.code} {detail}"
    except Exception as exc:  # noqa: BLE001
        return False, None, f"{type(exc).__name__}: {exc}"


def extract_json(text: str) -> str:
    """模型常把 JSON 包在 ```json 代码块里，或在前后加一句话。

    这里统一剥壳：去掉代码块围栏，再截取第一个 { 到最后一个 }。
    """
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t)
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        t = t[i:j + 1]
    return t.strip()


def generate_inferences(facts: list[dict], cfg: LLMConfig | None = None) -> InferenceOutcome:
    """facts 必须是**事实类**证据（cog == 'fact'）。其余一律不传。"""
    cfg = cfg or LLMConfig.from_env()
    facts = [f for f in facts if f.get("cog") == "fact"]

    if not cfg.enabled:
        return InferenceOutcome(
            status="disabled",
            detail="未配置 LLM_BASE_URL / LLM_API_KEY，推断层未启用。"
                   "事实、未知、矛盾三类证据不受影响，仍由确定性代码产出。",
        )
    if not facts:
        return InferenceOutcome(status="empty", detail="没有事实可供解读，跳过推断层。",
                                provider=cfg.provider, model=cfg.model)

    lines = []
    for f in facts:
        lines.append(f"{f['id']}｜{f['dim']}｜{strip_tags(f['claim'])}")
    user = ("事实清单（这是你的全部信息边界）：\n" + "\n".join(lines) +
            "\n\n请给出解读。只输出 JSON。")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    ok, payload, err = _post_chat(cfg, messages, json_mode=True)
    fallback = ""
    if not ok:
        # 该端点可能不支持 response_format，退回普通模式再试一次
        ok2, payload2, err2 = _post_chat(cfg, messages, json_mode=False)
        if not ok2:
            return InferenceOutcome(status="error", detail=f"调用失败：{err2 or err}",
                                    provider=cfg.provider, model=cfg.model)
        payload = payload2
        fallback = "（该端点不支持 response_format，已退回普通模式解析）"

    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return InferenceOutcome(status="error", detail="响应结构不符合预期",
                                raw=json.dumps(payload, ensure_ascii=False)[:400],
                                provider=cfg.provider, model=cfg.model)

    outcome = InferenceOutcome(status="ok", raw=content, detail=fallback,
                               provider=cfg.provider, model=cfg.model)
    valid_ids = {f["id"] for f in facts}

    try:
        parsed = json.loads(extract_json(content))
        raw_items = parsed.get("inferences", [])
    except Exception as exc:  # noqa: BLE001
        outcome.status = "error"
        outcome.detail = f"模型未返回合法 JSON：{type(exc).__name__}" + fallback
        return outcome

    for it in raw_items if isinstance(raw_items, list) else []:
        claim = str(it.get("claim", "")).strip()
        cites = [str(c).strip() for c in (it.get("cites") or [])]
        bad = [c for c in cites if c not in valid_ids]
        if not claim:
            outcome.dropped.append({"claim": claim, "reason": "空解读"})
            continue
        if not cites:
            outcome.dropped.append({"claim": claim, "reason": "未引用任何事实"})
            continue
        if bad:
            outcome.dropped.append({"claim": claim, "reason": f"引用了不存在的事实 {bad}"})
            continue
        outcome.inferences.append({
            "dim": "推断解读",
            "cog": "infer",
            "lean": None,          # 倾向由规则给出，不由模型自评
            "claim": claim,
            "meta": f"由模型基于引用的 {len(cites)} 条事实生成：{cfg.provider} / {cfg.model}",
            "rule": "L-10　推断必须引用已存在的事实；引用不存在的事实，整条丢弃。"
                    "倾向不由模型自评，交由规则继承所引用事实的多数方向。",
            "source": {"llm_provider": cfg.provider, "llm_model": cfg.model},
            "series": [],
            "cites": cites,
            "note": "",
        })

    if not outcome.inferences and outcome.status == "ok":
        outcome.status = "empty"
        outcome.detail = ("模型没有给出可校验的解读（可能全部因未引用或引用无效被丢弃）。"
                          + fallback)
    return outcome


def strip_tags(html: str) -> str:
    """证据的 claim 里带 <b> 之类的展示标签，喂给模型前要剥掉。"""
    out, depth = [], 0
    for ch in html:
        if ch == "<":
            depth += 1
            continue
        if ch == ">":
            depth = max(0, depth - 1)
            continue
        if depth == 0:
            out.append(ch)
    return "".join(out).strip()


# ---------------------------------------------------------------- 自检
# 拿到模型 Key 后，先跑这个再上产品：
#     set LLM_BASE_URL / LLM_API_KEY / LLM_MODEL   然后
#     python llm.py
# 它会用一小撮真实形状的证据试一次调用，并把接受/丢弃的明细打出来。

DEMO_FACTS = [
    {"id": "E-01", "cog": "fact", "lean": "neg", "dim": "财务趋势",
     "claim": "中报营业收入同比增速连续 3 年回落 17.76% → 9.10% → 1.47%"},
    {"id": "E-02", "cog": "fact", "lean": "pos", "dim": "经营质量",
     "claim": "2025 年报加权平均ROE 32.53%"},
    {"id": "E-03", "cog": "unknown", "lean": None, "dim": "估值",
     "claim": "没有历史估值序列，无法判断分位"},   # 不应进入模型上下文
]

PRESETS = {
    # ⚠️ 实测：glm-4.7-flash 是推理模型，回复两个字要 57 秒，本场景不可用。
    #    glm-4-flash-250414 只要 0.4 秒。
    "glm": ("https://open.bigmodel.cn/api/paas/v4", "glm-4-flash-250414"),
    "deepseek": ("https://api.deepseek.com/v1", "deepseek-chat"),
    "qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-turbo"),
    "siliconflow": ("https://api.siliconflow.cn/v1", "Qwen/Qwen3-8B"),
}


def smoke_test() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    cfg = LLMConfig.from_env()
    print("推断层自检")
    print(f"  端点 : {cfg.base_url or '（未配置）'}")
    print(f"  模型 : {cfg.model or '（未配置）'}")
    print(f"  Key  : {'已配置（长度 %d，不回显）' % len(cfg.api_key) if cfg.api_key else '未配置'}")
    print()

    if not cfg.enabled:
        print("推断层未启用。设置以下环境变量后重试：")
        for name, (base, model) in PRESETS.items():
            print(f"  [{name}]")
            print(f"    LLM_BASE_URL={base}")
            print(f"    LLM_MODEL={model}")
        print("    LLM_API_KEY=<你的 key>")
        return 1

    outcome = generate_inferences(DEMO_FACTS, cfg)
    print(f"状态 : {outcome.status}")
    if outcome.detail:
        print(f"说明 : {outcome.detail}")
    print(f"接受 {len(outcome.inferences)} 条，丢弃 {len(outcome.dropped)} 条")
    for inf in outcome.inferences:
        print(f"  · {inf['claim']}")
        print(f"    引用 {inf['cites']}")
    for d in outcome.dropped:
        print(f"  × 丢弃：{d['claim'][:48]} —— {d['reason']}")
    if outcome.raw:
        print(f"\n模型原始输出（截断）：\n{outcome.raw[:500]}")

    ok = outcome.status in ("ok", "empty")
    print("\n结论：" + ("推断层可用。" if ok else "推断层不可用，请检查端点 / Key / 模型名。"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(smoke_test())
