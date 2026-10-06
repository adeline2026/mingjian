# -*- coding: utf-8 -*-
"""证据引擎：把扶摇的原始数据装配成带双坐标的证据对象。

证据模型（本产品的核心）
------------------------
每条证据有两个独立坐标，外加一个稳定 id：

    cog  认知：fact 事实 / infer 推断 / unknown 未知
    lean 倾向：pos 正面 / neg 负面 / neu 中性 / contra 矛盾 / None（未知项无倾向）

三条铁律
--------
1. 「事实 / 未知」由**代码**判定，不由模型决定。
2. 「正面 / 负面」由**规则**判定，规则编号随证据一起输出，可被追问「凭什么」。
3. 只有「推断」留给模型，且只能引用已存在的事实（见 llm.py）。

口径纪律
--------
同比增速只在**同类型报告期**之间比较（年报比年报、中报比中报）。
把年报、三季报、一季报混在一根序列里比，是错误做法，本引擎不做。
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any

import fuyao
import llm
from fuyao import FuyaoResult

CN_TZ = timezone(timedelta(hours=8))

REPORT_LABEL = {"1": "一季报", "2": "中报", "3": "三季报", "4": "年报"}

# 记录趋势用的关键指标
IDX_REVENUE_YOY = "calculate_operating_income_yoy_growth_ratio"
IDX_PROFIT_YOY = "calculate_parent_holder_net_profit_yoy_growth_ratio"
IDX_ROE = "index_weighted_avg_roe"
IDX_GROSS_MARGIN = "sale_gross_margin"
IDX_NET_MARGIN = "sale_net_interest_ratio"
IDX_DEBT_RATIO = "assets_debt_ratio"
IDX_CASH_CONTENT = "net_profit_cash_content"


class EvidenceBuilder:
    """按顺序发号，保证每条证据有稳定 id，便于「推断」引用。"""

    def __init__(self) -> None:
        self.items: list[dict] = []
        self._seq = 0

    def add(self, *, dim: str, cog: str, lean: str | None, claim: str,
            meta: str = "", rule: str = "", source: dict | None = None,
            series: list | None = None, cites: list | None = None,
            note: str = "") -> dict:
        self._seq += 1
        item = {
            "id": f"E-{self._seq:02d}",
            "dim": dim,
            "cog": cog,
            "lean": lean,
            "claim": claim,
            "meta": meta,
            "rule": rule,
            "source": source or {},
            "series": series or [],
            "cites": cites or [],
            "note": note,
        }
        self.items.append(item)
        return item


# ---------------------------------------------------------------- 规则库
# 规则编号固定，写进证据里，任何一条倾向都能被追问依据。

def rule_L00(why: str = "") -> str:
    base = "L-00　单期数值本身不含方向，标为中性；方向由同口径趋势或同业对比提供。"
    return base + ("　" + why if why else "")


def rule_L01(direction: str, values: list[float], label: str) -> str:
    arrow = "回落" if direction == "down" else "上升"
    verdict = "负面" if direction == "down" else "正面"
    return (f"L-01　同一指标在同口径报告期上连续单调{arrow} → {verdict}。"
            f"本期判定基于序列 {' → '.join(f'{v:.2f}%' for v in values)}。")


def rule_L02(profit_yoy: float, cash_content: float) -> str:
    return (f"L-02　同一报告期内归母净利同比 {profit_yoy:.2f}% 大于 0，"
            f"但净利现金含量 {cash_content:.2f}% 低于 100% → 矛盾。")


def rule_L03(spread: float, values: list[float]) -> str:
    return (f"L-03　同一指标连续 {len(values)} 期极差 {spread:.2f} 个百分点，小于 1 → 正面（稳定性）。")


def rule_L04(roe: float, rp_label: str) -> str:
    if roe >= 15:
        return f"L-04　{rp_label}加权平均ROE {roe:.2f}%，高于 15% → 正面。"
    if roe < 8:
        return f"L-04　{rp_label}加权平均ROE {roe:.2f}%，低于 8% → 负面。"
    return f"L-04　{rp_label}加权平均ROE {roe:.2f}%，位于 8%–15% → 中性。"


def rule_L05(net_margin: float, debt_ratio: float) -> str:
    return (f"L-05　净利率 {net_margin:.2f}% 高于 20% 且资产负债率 {debt_ratio:.2f}% 低于 40% "
            f"→ 正面。两项同时满足才判正面。")


def rule_L06(name: str, pe_ttm: float, pe_mrq: float) -> str:
    return (f"L-06　{name} 的 PE(TTM) {pe_ttm:.2f} 与 PE(MRQ) {pe_mrq:.2f} "
            f"比值 {pe_ttm / pe_mrq:.1f}，大于 3 → 矛盾。")


def rule_L07(label: str, yoy: float) -> str:
    if yoy < 0:
        return f"L-07　{label}单期同比 {yoy:.2f}% 为负，规模较上年同期收缩 → 负面。"
    return (f"L-00　{label}单期同比 {yoy:.2f}% 为正增长；但单看增速无法判断是否够好，"
            f"标为中性，方向交给同口径趋势判定。")


def rule_L08(debt: float) -> str:
    return (f"L-08　资产负债率 {debt:.2f}% 高于 100%，负债超过全部资产、"
            f"所有者权益为负 → 负面（资不抵债）。")


def rule_L09(drawdown: float) -> str:
    return (f"L-09　区间最大回撤 {drawdown:.2f}% 达到或超过 30% → 负面。"
            f"回撤只描述**已经发生的价格路径**，不预测后续走势。")


def _ms_to_date(ms: Any) -> str:
    v = fuyao.to_float(ms)
    if not v:
        return ""
    return datetime.fromtimestamp(v / 1000, CN_TZ).strftime("%Y-%m-%d")


def _range_stats(bars: list[dict]) -> dict:
    """从日线序列算区间统计。缺失值不参与，不补零。"""
    import statistics

    closes, highs, lows = [], [], []
    for bar in bars:
        c = fuyao.to_float(bar.get("close_price"))
        h = fuyao.to_float(bar.get("high_price"))
        lo = fuyao.to_float(bar.get("low_price"))
        if c is not None:
            closes.append(c)
        if h is not None:
            highs.append(h)
        if lo is not None:
            lows.append(lo)
    if len(closes) < 2:
        return {}

    peak, max_dd = closes[0], 0.0
    for c in closes:
        peak = max(peak, c)
        if peak > 0:
            max_dd = max(max_dd, (peak - c) / peak)

    rets = [(b / a - 1.0) for a, b in zip(closes, closes[1:]) if a > 0]
    vol = statistics.stdev(rets) * (252 ** 0.5) if len(rets) >= 2 else None

    return {
        "days": len(closes),
        "high": max(highs) if highs else None,
        "low": min(lows) if lows else None,
        "change_pct": (closes[-1] / closes[0] - 1.0) * 100 if closes[0] else None,
        "max_drawdown_pct": max_dd * 100,
        "annual_vol_pct": vol * 100 if vol is not None else None,
        "first_date": _ms_to_date(bars[0].get("date_ms")),
        "last_date": _ms_to_date(bars[-1].get("date_ms")),
    }


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return "—"


def _trend(values: list[float]) -> str | None:
    """连续单调则返回 'up' / 'down'，否则 None。"""
    if len(values) < 3:
        return None
    if all(b < a for a, b in zip(values, values[1:])):
        return "down"
    if all(b > a for a, b in zip(values, values[1:])):
        return "up"
    return None


def _rp_label(rp: str) -> str:
    if not rp:
        return ""
    y, _, r = rp.partition("-")
    return f"{y} {REPORT_LABEL.get(r, r)}"


def _src(res: FuyaoResult, field: str = "", as_of: str = "", unit: str = "",
         caliber: str = "", raw: Any = None) -> dict:
    return {
        "endpoint": res.endpoint,
        "params": res.params,
        "request_id": res.request_id,
        "retrieved_at": res.retrieved_at,
        "field": field,
        "as_of": as_of,
        "unit": unit,
        "caliber": caliber,
        "raw": raw,
    }


# ---------------------------------------------------------------- 问题路由
# 用户问题 -> 诊断维度。由**规则**判定，规则编号随结果一起输出，可被追问。
# 刻意不用模型来猜：路由错了，整个诊断的方向就错了，这事不该交给概率。

ANALYSIS_DIMS = ("财务趋势", "经营质量", "估值", "行情特征")

QUESTION_ROUTES = (
    ("Q-01", ("状态", "怎么样", "情况", "概览", "基本面", "整体"),
     ANALYSIS_DIMS, "公司整体状态概览"),
    ("Q-02", ("赚钱", "盈利", "利润", "挣钱", "roe", "赚钱能力"),
     ("经营质量", "财务趋势"), "盈利能力与盈利质量"),
    ("Q-03", ("贵", "便宜", "估值", "划算", "pe", "pb", "价格"),
     ("估值",), "估值水平与同业对比"),
    ("Q-04", ("风险", "警惕", "问题", "隐患", "异常", "矛盾", "雷"),
     ("财务趋势", "估值", "待验证问题"), "风险信号：矛盾与未知"),
    ("Q-05", ("趋势", "变化", "变好", "变差", "成长", "下滑", "增长", "放缓"),
     ("财务趋势",), "变化方向"),
    ("Q-06", ("行情", "股价", "走势", "交易", "涨跌", "成交"),
     ("行情特征",), "市场价格表现"),
)


def route_question(question: str) -> dict:
    """把用户问题映射到诊断维度。命中不了就展示全部，不做猜测。

    按**命中关键词的数量**取最优，而不是按顺序先命中先返回 ——
    否则「它赚钱能力怎么样」会先被 Q-01 的泛化关键词「怎么样」抢走，
    而它真正想问的是 Q-02（赚钱能力）。
    """
    q = (question or "").strip()
    if not q:
        return {"question": "", "rule": "Q-00",
                "reason": "未指定问题，展示全部维度",
                "dimensions": list(ANALYSIS_DIMS)}
    low = q.lower()
    best = None
    for rid, keys, dims, label in QUESTION_ROUTES:
        hit = [k for k in keys if k.lower() in low]
        if hit and (best is None or len(hit) > len(best[0])):
            best = (hit, rid, dims, label)
    if best:
        hit, rid, dims, label = best
        extra = [d for d in dims if d not in ANALYSIS_DIMS]
        reason = f"命中关键词「{'、'.join(hit)}」→ {label}"
        if extra:
            reason += "（并固定展示：" + "、".join(extra) + "）"
        return {"question": q, "rule": rid, "reason": reason, "dimensions": list(dims)}
    return {"question": q, "rule": "Q-99",
            "reason": "未命中任何已知问题模式，展示全部维度（不做猜测）",
            "dimensions": list(ANALYSIS_DIMS)}


# ---------------------------------------------------------------- 公司类型
# 类型会影响「哪些规则适用」。最典型的是资产负债率：
# 银行天然高杠杆（90%+），拿 40% 的阈值去衡量它，会把每一家银行都判错。

COMPANY_TYPES = ("金融/地产", "高毛利（消费/软件）", "高杠杆（非金融）", "通用")


def infer_company_type(ind: dict) -> dict:
    """按可观测的财务特征推断公司类型。规则透明，结果可被质疑。

    ⚠️ **高负债不等于金融业。** 这是实测踩到的坑：
      - *ST八钢（钢铁）资产负债率 106%、净利率 -10% —— 高负债是**陷入困境的结果**
      - 招商银行      资产负债率  90%、净利率  45% —— 高负债是**经营模式本身**
    区分二者的可观测特征就是**净利率**：赚利差的机构净利率不会低，
    资不抵债的企业净利率往往为负。只看负债率会把困境重资产企业误判成金融公司。
    """
    debt = fuyao.to_float(ind.get("assets_debt_ratio"))
    gross = fuyao.to_float(ind.get("sale_gross_margin"))
    margin = fuyao.to_float(ind.get("sale_net_interest_ratio"))

    if debt is not None and debt >= 70 and margin is not None and margin >= 10:
        return {
            "type": "金融/地产", "rule": "T-01",
            "reason": f"资产负债率 {debt:.2f}% 高于 70%，且净利率 {margin:.2f}% 高于 10%"
                      f" —— 高杠杆是它的经营模式本身，而非困境所致",
            "affected": ["L-05 的资产负债率条件对本类不适用（银行天然高杠杆）"],
        }
    if gross is not None and gross >= 50:
        return {
            "type": "高毛利（消费/软件）", "rule": "T-02",
            "reason": f"销售毛利率 {gross:.2f}% 高于 50% —— 典型的高毛利业务模式",
            "affected": ["L-03 毛利率稳定性规则对本类更有参考价值"],
        }
    if debt is not None and debt >= 70:
        margin_txt = f"净利率仅 {margin:.2f}%" if margin is not None else "净利率偏低"
        return {
            "type": "高杠杆（非金融）", "rule": "T-03",
            "reason": f"资产负债率 {debt:.2f}% 高于 70%，但{margin_txt}"
                      f" —— 高负债更可能是**风险信号**，而非经营模式",
            "affected": ["L-05 的资产负债率条件照常适用（不能按金融业豁免）"],
        }
    parts = []
    if debt is not None:
        parts.append(f"资产负债率 {debt:.2f}%")
    if gross is not None:
        parts.append(f"毛利率 {gross:.2f}%")
    return {
        "type": "通用", "rule": "T-00",
        "reason": "未命中高杠杆或高毛利的特征" + (f"（{ '、'.join(parts) }）" if parts else ""),
        "affected": [],
    }


# ---------------------------------------------------------------- 主流程

def diagnose(thscode: str, peers: list[str] | None = None,
             periods: list[str] | None = None,
             fetch_log: list | None = None,
             question: str = "", company_type: str = "") -> dict:
    peers = peers or []
    fetch_log = fetch_log if fetch_log is not None else []
    b = EvidenceBuilder()
    routing = route_question(question)

    def log(res: FuyaoResult) -> FuyaoResult:
        fetch_log.append(res.to_dict())
        return res

    now = datetime.now(CN_TZ)

    # ---------- 1. 标的消歧 ----------
    meta_res = log(fuyao.search_ticker(thscode, limit=1))
    name = thscode
    if meta_res.ok:
        items = (meta_res.data or {}).get("item") or []
        if items:
            name = items[0].get("name") or thscode

    # ---------- 2. 财务指标（多期，同口径） ----------
    if periods is None:
        y, q = fuyao.last_completed_period(now)
        periods = fuyao.report_periods(y, q, count=12)

    fin_by_period: dict[str, dict[str, Any]] = {}
    fin_res_by_period: dict[str, FuyaoResult] = {}
    missing_periods: list[tuple[str, FuyaoResult]] = []

    # 串行拉 12 个报告期要 20 秒以上，页面看起来像卡死。改成 3 路并发：
    # 单次请求约 1.5 秒，3 并发约 2 请求/秒，仍在接口的 5 次/秒限制之内。
    def _fetch_one(rp: str) -> tuple[str, FuyaoResult]:
        return rp, fuyao.financials_indicators(thscode, rp)

    with ThreadPoolExecutor(max_workers=3) as pool:
        fetched = list(pool.map(_fetch_one, periods))

    for rp, res in fetched:                      # map 保持入参顺序，日志可复现
        log(res)
        vals = fuyao.flatten_indicators(res.data) if res.ok else {}
        has_value = any(fuyao.to_float(v) is not None for v in vals.values())
        if res.ok and has_value:
            fin_by_period[rp] = vals
            fin_res_by_period[rp] = res
        else:
            missing_periods.append((rp, res))

    def _sort_key(rp: str) -> tuple[int, int]:
        y, _, r = rp.partition("-")
        return int(y), int(r)

    have = sorted(fin_by_period.keys(), key=_sort_key)

    def series_of(index_id: str) -> list[tuple[str, float]]:
        pts: list[tuple[str, float]] = []
        for rp in have:
            v = fuyao.to_float(fin_by_period[rp].get(index_id))
            if v is not None:
                pts.append((rp, v))
        return pts

    def same_type(pts: list[tuple[str, float]], tag: str) -> list[tuple[str, float]]:
        return [(rp, v) for rp, v in pts if rp.endswith("-" + tag)]

    def fin_src(rp: str, field: str, **kw) -> dict:
        """财务类证据的来源必须是**财报接口那一次调用**，不能张冠李戴到检索接口。"""
        return _src(fin_res_by_period.get(rp) or meta_res, field=field, as_of=rp, **kw)

    def series_src(pts: list[tuple[str, float]], field: str, **kw) -> dict:
        """多期证据的来源：逐个列出贡献过的报告期及其 request_id。

        单期证据只对应一次请求；多期趋势对应多次请求，必须逐条列出，
        否则「参数」写的是一期、而「原始值」是几期，点开会对不上。
        """
        parts = []
        for rp, v in pts:
            res = fin_res_by_period.get(rp)
            rid = (res.request_id or "")[:8] if res else "—"
            parts.append(f"{rp}={v:.2f}（req {rid}）")
        src = fin_src(pts[-1][0] if pts else "", field, **kw)
        src["contributions"] = "；".join(parts)
        return src

    latest_tag = have[-1].partition("-")[2] if have else "4"
    type_name = REPORT_LABEL.get(latest_tag, latest_tag)
    same_caliber = f"同为 {type_name}，报告期长度一致，可直接比较"

    # 公司类型：决定哪些规则适用。未显式指定时按财务特征自动推断。
    latest_ind = fin_by_period.get(have[-1], {}) if have else {}
    if company_type and company_type.lower() != "auto":
        ctype = {"type": company_type, "rule": "T-99",
                 "reason": f"使用者显式指定为「{company_type}」", "affected": []}
    else:
        ctype = infer_company_type(latest_ind)
    is_financial = ctype["type"] == "金融/地产"

    # ---- 2.1 单期同比（营收 / 净利） ----
    for idx, label in ((IDX_REVENUE_YOY, "营业收入"), (IDX_PROFIT_YOY, "归母净利润")):
        pts = series_of(idx)
        if not pts:
            continue
        rp, val = pts[-1]
        b.add(dim="财务趋势", cog="fact", lean=("neg" if val < 0 else "neu"),
              claim=f"{_rp_label(rp)}{label}同比 <b>{_fmt(val)}%</b>",
              meta=f"指标 index_id：{idx}",
              rule=rule_L07(label, val),
              source=fin_src(rp, idx, unit="%",
                             caliber=f"{_rp_label(rp)}累计口径", raw=val))

    # ---- 2.2 同口径多年趋势（L-01） ----
    for idx, label in ((IDX_REVENUE_YOY, "营业收入"), (IDX_PROFIT_YOY, "归母净利润")):
        pts = same_type(series_of(idx), latest_tag)
        if len(pts) < 3:
            continue
        vals = [v for _, v in pts]
        direction = _trend(vals)
        if direction is None:
            b.add(dim="财务趋势", cog="fact", lean="neu",
                  claim=f"{type_name}{label}同比近年未呈单调变化　"
                        + " → ".join(f"{_fmt(v)}%" for v in vals),
                  meta=f"同一 index_id，仅取 {type_name}（{len(vals)} 期）",
                  rule="L-01　未构成连续单调变化，不判定方向 → 中性。序列 "
                       + " → ".join(f"{_fmt(v)}%" for v in vals) + "。",
                  series=[[_rp_label(rp), f"{_fmt(v)}%"] for rp, v in pts],
                  source=series_src(pts, idx, unit="%", caliber=same_caliber,
                                    raw=[v for _, v in pts]))
            continue
        b.add(dim="财务趋势", cog="fact",
              lean=("neg" if direction == "down" else "pos"),
              claim=f"{type_name}{label}同比增速连续 {len(vals)} 年"
                    f"{'回落' if direction == 'down' else '上升'}　"
                    + " → ".join(f"{_fmt(v)}%" for v in vals),
              meta=f"同一 index_id，仅取 {type_name}（{len(vals)} 期）",
              rule=rule_L01(direction, vals, label),
              series=[[_rp_label(rp), f"{_fmt(v)}%"] for rp, v in pts],
              source=series_src(pts, idx, unit="%", caliber=same_caliber,
                                raw=[v for _, v in pts]))

    # ---- 2.3 毛利率稳定性（L-03） ----
    gp = same_type(series_of(IDX_GROSS_MARGIN), latest_tag) or series_of(IDX_GROSS_MARGIN)
    if len(gp) >= 3:
        vals = [v for _, v in gp][-3:]
        spread = max(vals) - min(vals)
        if spread < 1:
            b.add(dim="财务趋势", cog="fact", lean="pos",
                  claim=f"毛利率近三期稳定在 <b>{_fmt(min(vals))}–{_fmt(max(vals))}%</b>，"
                        f"最新为 {_fmt(vals[-1])}%",
                  meta="指标 index_id：sale_gross_margin",
                  rule=rule_L03(spread, vals),
                  series=[[_rp_label(rp), f"{_fmt(v)}%"] for rp, v in gp[-3:]],
                  source=series_src(gp[-3:], IDX_GROSS_MARGIN, unit="%",
                                    caliber=same_caliber, raw=[v for _, v in gp[-3:]]))

    # ---- 2.4 利润 vs 现金含量（L-02 矛盾检测） ----
    np_pts = series_of(IDX_PROFIT_YOY)
    cash_pts = series_of(IDX_CASH_CONTENT)
    if np_pts and cash_pts and np_pts[-1][0] == cash_pts[-1][0]:
        profit, cash = np_pts[-1][1], cash_pts[-1][1]
        if profit > 0 and cash < 100:
            b.add(dim="财务趋势", cog="fact", lean="contra",
                  claim=f"净利润同比为正（{_fmt(profit)}%），但净利现金含量仅 "
                        f"<b>{_fmt(cash)}%</b>，利润未充分转化为现金",
                  meta="两条均来自同一报告期，方向不一致",
                  rule=rule_L02(profit, cash),
                  source=fin_src(cash_pts[-1][0], IDX_CASH_CONTENT, unit="%",
                                 caliber=f"与利润同比同为 {_rp_label(cash_pts[-1][0])}",
                                 raw=cash),
                  note="两项指标均为真，但指向不同。现金含量低于 100% 是否属于趋势性问题，"
                       "需结合下一期数据判断。本产品不对此下判断。")

    # ---- 2.5 未取到的报告期 → 未知 ----
    if missing_periods:
        rp, res = missing_periods[0]
        others = len(missing_periods) - 1
        b.add(dim="财务趋势", cog="unknown", lean=None,
              claim=f"{_rp_label(rp)}数据未取到，该报告期不参与任何判定"
                    + (f"（另有 {others} 个报告期同样未取到）" if others else ""),
              meta=f"接口返回 code={res.code}" if res.code is not None else "请求失败",
              rule="认知为未知，倾向不适用。",
              source={"endpoint": res.endpoint, "params": res.params,
                      "code": res.code, "message": res.message,
                      "reason": res.reason,
                      "处理": "标记为未知；不补零、不估算、不参与结论"})

    # ---------- 3. 经营质量 ----------
    roe_pts = series_of(IDX_ROE)
    annual = [(rp, v) for rp, v in roe_pts if rp.endswith("-4")]
    if annual:
        rp, roe = annual[-1]
        lean = "pos" if roe >= 15 else ("neg" if roe < 8 else "neu")
        b.add(dim="经营质量", cog="fact", lean=lean,
              claim=f"{_rp_label(rp)}加权平均ROE <b>{_fmt(roe)}%</b>",
              meta="指标 index_id：index_weighted_avg_roe",
              rule=rule_L04(roe, _rp_label(rp)),
              source=fin_src(rp, IDX_ROE, unit="%",
                             caliber="年报为全年口径，与该阈值口径一致", raw=roe))
    elif roe_pts:
        rp, roe = roe_pts[-1]
        b.add(dim="经营质量", cog="fact", lean="neu",
              claim=f"{_rp_label(rp)}加权平均ROE <b>{_fmt(roe)}%</b>（累计口径）",
              meta="指标 index_id：index_weighted_avg_roe",
              rule="L-00　无年报数据，累计口径 ROE 与全年阈值不可比，标为中性。",
              source=fin_src(rp, IDX_ROE, unit="%",
                             caliber="累计口径，不可与全年阈值比较", raw=roe))

    nm_pts = series_of(IDX_NET_MARGIN)
    dr_pts = series_of(IDX_DEBT_RATIO)
    if nm_pts and dr_pts:
        nm_rp, nm = nm_pts[-1]
        dr_rp, dr = dr_pts[-1]
        same = nm_rp == dr_rp
        if is_financial:
            # 银行 / 地产天然高杠杆，40% 的阈值用在这里会把每一家都判错
            lean = "pos" if (same and nm > 20) else "neu"
            rule_txt = (f"L-05　公司类型为「{ctype['type']}」（{ctype['rule']}：{ctype['reason']}），"
                        f"资产负债率阈值不适用，仅按销售净利率 {nm:.2f}% 判定。")
        elif same:
            lean = "pos" if (nm > 20 and dr < 40) else "neu"
            rule_txt = rule_L05(nm, dr)
        else:
            lean = "neu"
            rule_txt = "L-05　两项指标的可用报告期不一致，无法在同一时点上同时满足判定条件 → 中性。"
        b.add(dim="经营质量", cog="fact", lean=lean,
              claim=f"{_rp_label(nm_rp)}销售净利率 <b>{_fmt(nm)}%</b>，"
                    f"资产负债率 <b>{_fmt(dr)}%</b>",
              meta="index_id：sale_net_interest_ratio / assets_debt_ratio",
              rule=rule_txt,
              source={
                  "net_margin": _src(fin_res_by_period.get(nm_rp) or meta_res,
                                     field=IDX_NET_MARGIN, as_of=nm_rp, unit="%", raw=nm),
                  "debt_ratio": _src(fin_res_by_period.get(dr_rp) or meta_res,
                                     field=IDX_DEBT_RATIO, as_of=dr_rp, unit="%", raw=dr),
                  "caliber": "同期，可直接同时判定" if same
                             else "非同期：两个指标取自不同报告期，无法在同一时点上同时判定",
              },
              note="" if same else
                   f"注意：净利率取 {_rp_label(nm_rp)}、资产负债率取 {_rp_label(dr_rp)}，两者非同期。")

    # 资不抵债：资产负债率 > 100%，这是一条独立且重要的负面信号
    if dr_pts and dr_pts[-1][1] > 100:
        drp, dv = dr_pts[-1]
        b.add(dim="经营质量", cog="fact", lean="neg",
              claim=f"{_rp_label(drp)}资产负债率 <b>{_fmt(dv)}%</b> 高于 100%，净资产为负（资不抵债）",
              meta="指标 index_id：assets_debt_ratio",
              rule=rule_L08(dv),
              source=fin_src(drp, IDX_DEBT_RATIO, unit="%",
                             caliber="资产负债率 = 负债 ÷ 资产；超过 100% 意味着所有者权益为负",
                             raw=dv),
              note="这不影响公司类型判定：高负债若来自困境而非经营模式，"
                   "分类会落到「高杠杆（非金融）」，相关规则不被豁免。")

    # ---------- 4. 估值 ----------
    val_res = log(fuyao.valuations_snapshot([thscode] + peers))
    val_map: dict[str, dict] = {}
    if val_res.ok:
        for it in (val_res.data or {}).get("item") or []:
            val_map[it.get("thscode", "")] = it
    val_ts = str((val_res.data or {}).get("timestamp", ""))

    self_val = val_map.get(thscode)
    if self_val:
        b.add(dim="估值", cog="fact", lean="neu",
              claim=(f"PE(TTM) <b>{_fmt(self_val.get('pe_ttm'))}</b>　"
                     f"PE(MRQ) {_fmt(self_val.get('pe_mrq'))}　"
                     f"PB(MRQ) {_fmt(self_val.get('pb_mrq'))}　"
                     f"PS(TTM) {_fmt(self_val.get('ps_ttm'))}"),
              meta="指标来自估值快照接口五个字段",
              rule=rule_L00("缺少历史分位与同业中枢，无法判定高低。"),
              source=_src(val_res, field="pe_ttm/pe_mrq/pb_mrq/ps_ttm/pcf_ttm",
                          as_of=val_ts, unit="倍", caliber="当期快照，无历史序列",
                          raw={k: self_val.get(k) for k in
                               ("pe_ttm", "pe_mrq", "pb_mrq", "ps_ttm", "pcf_ttm")}))

    peer_rows = [(val_map[c].get("name", c), _fmt(val_map[c].get("pe_ttm"))) for c in peers if c in val_map]
    if peer_rows:
        b.add(dim="估值", cog="fact", lean="neu",
              claim="同业 PE(TTM)：" + "　".join(f"{n} {v}" for n, v in peer_rows),
              meta="同一次批量请求取回，同一时点、同一口径",
              rule=rule_L00("横向对比本身不产生方向判断；列出比较不构成推荐。"),
              series=[[n, v] for n, v in peer_rows],
              source=_src(val_res, field="pe_ttm", as_of=val_ts, unit="倍",
                          caliber="同批请求，同一时点"))

    for code in [thscode] + peers:
        it = val_map.get(code)
        if not it:
            continue
        pe_ttm = fuyao.to_float(it.get("pe_ttm"))
        pe_mrq = fuyao.to_float(it.get("pe_mrq"))
        if pe_ttm and pe_mrq and pe_ttm > 0 and pe_mrq > 0 and pe_ttm / pe_mrq > 3:
            nm_label = it.get("name", code)
            b.add(dim="估值", cog="fact", lean="contra",
                  claim=f"{nm_label} 的 PE(TTM) {_fmt(pe_ttm)} 与 PE(MRQ) {_fmt(pe_mrq)} "
                        f"相差逾 {pe_ttm / pe_mrq:.0f} 倍，两口径严重打架",
                  meta="同一标的、同一时点、两个口径",
                  rule=rule_L06(nm_label, pe_ttm, pe_mrq),
                  source=_src(val_res, field="pe_ttm / pe_mrq", as_of=code, unit="倍",
                              caliber="同一次批量请求",
                              raw={"pe_ttm": pe_ttm, "pe_mrq": pe_mrq}),
                  note="这提示口径本身可能存在问题，而不是公司贵或便宜。"
                       "本产品不判断原因，列为待核实项。")

    b.add(dim="估值", cog="unknown", lean=None,
          claim="没有历史估值序列，无法判断当前 PE 处于历史什么分位",
          meta="接口明确不提供历史估值",
          rule="认知为未知，倾向不适用。",
          source={"来源": "估值接口文档原文：不提供历史估值、分页、指标选择或高低估结论",
                  "处理": "不推断分位，不与其他时点比较，列为能力边界"})

    # ---------- 5. 行情特征 ----------
    px_res = log(fuyao.price_snapshot([thscode]))
    px = None
    if px_res.ok:
        items = (px_res.data or {}).get("item") or []
        if items:
            px = items[0]
    quote: dict[str, Any] = {}
    if px:
        quote = {
            "last_price": fuyao.to_float(px.get("last_price")),
            "change": fuyao.to_float(px.get("price_change")),
            "change_pct": fuyao.to_float(px.get("price_change_ratio_pct")),
            "open": fuyao.to_float(px.get("open_price")),
            "high": fuyao.to_float(px.get("high_price")),
            "low": fuyao.to_float(px.get("low_price")),
            "prev": fuyao.to_float(px.get("prev_price")),
            "turnover": fuyao.to_float(px.get("turnover")),
            "timestamp": str((px_res.data or {}).get("timestamp", "")),
        }
    if px:
        chg = fuyao.to_float(px.get("price_change"))
        chg_pct = fuyao.to_float(px.get("price_change_ratio_pct"))
        turnover = fuyao.to_float(px.get("turnover"))
        b.add(dim="行情特征", cog="fact", lean="neu",
              claim=(f"最新价 <b>{_fmt(fuyao.to_float(px.get('last_price')))}</b>，"
                     f"{'涨' if (chg or 0) >= 0 else '跌'} {_fmt(abs(chg) if chg is not None else None)}"
                     f"（{_fmt(chg_pct)}%），成交额 {_fmt((turnover or 0) / 1e8)} 亿元"),
              meta="来自行情快照，含开高低收与前收",
              rule=rule_L00("单日价格变动不构成方向判断。"),
              source=_src(px_res,
                          field="last_price/price_change/price_change_ratio_pct/turnover",
                          as_of=str((px_res.data or {}).get("timestamp", "")),
                          unit="CNY", caliber="原始货币计价",
                          raw={k: px.get(k) for k in
                               ("last_price", "price_change", "price_change_ratio_pct",
                                "open_price", "high_price", "low_price", "prev_price", "turnover")}))

    # ---------- 5b. 行情区间（近一年日线） ----------
    now_ms = int(now.timestamp() * 1000)
    hist_res = log(fuyao.price_historical(
        thscode, interval="1d", start=now_ms - 365 * 24 * 3600 * 1000,
        end=now_ms, adjust="forward"))
    bars: list[dict] = []
    if hist_res.ok and isinstance(hist_res.data, dict):
        raw_bars = hist_res.data.get("item") or hist_res.data.get("items") or []
        bars = [x for x in raw_bars if fuyao.to_float(x.get("close_price")) is not None]

    st = _range_stats(bars) if len(bars) >= 20 else {}
    if st.get("change_pct") is not None:
        span = f"{st['first_date']} 至 {st['last_date']}"
        b.add(dim="行情特征", cog="fact", lean="neu",
              claim=(f"近 {st['days']} 个交易日区间涨跌幅 <b>{st['change_pct']:+.2f}%</b>，"
                     f"区间最高 {st['high']:.2f}、最低 {st['low']:.2f}"),
              meta=f"日线区间 {span}（前复权）",
              rule=rule_L00("一个区间的涨跌是事实。本产品不对**价格方向**下判断 —— "
                            "同一区间的风险信号由下方「最大回撤」承担，两者分工不同。"),
              source=_src(hist_res, field="close_price / high_price / low_price",
                          as_of=span, unit="CNY",
                          caliber="前复权日线；区间涨跌幅 = 末收盘 ÷ 首收盘 − 1",
                          raw={"bars": st["days"], "change_pct": round(st["change_pct"], 2),
                               "high": st["high"], "low": st["low"]}))

        dd = st.get("max_drawdown_pct")
        if dd is not None:
            lean = "neg" if dd >= 30 else "neu"
            b.add(dim="行情特征", cog="fact", lean=lean,
                  claim=f"区间最大回撤 <b>{dd:.2f}%</b>"
                        + ("，自区间高点回落幅度较大" if dd >= 30 else "，幅度在常规区间内"),
                  meta=f"日线区间 {span}",
                  rule=rule_L09(dd) if dd >= 30 else
                       f"L-00　区间最大回撤 {dd:.2f}%，未达到 30% 的判定阈值 → 中性。",
                  source=_src(hist_res, field="close_price", as_of=span, unit="%",
                              caliber="逐日跟踪区间峰值，取 (峰值 − 当日收盘) ÷ 峰值 的最大值",
                              raw=round(dd, 2)))

        vol = st.get("annual_vol_pct")
        if vol is not None:
            b.add(dim="行情特征", cog="fact", lean="neu",
                  claim=f"年化波动率 <b>{vol:.2f}%</b>",
                  meta=f"日线区间 {span}，共 {st['days']} 个交易日",
                  rule=rule_L00("波动率高只说明价格起伏大，本身不指向好或坏。"),
                  source=_src(hist_res, field="close_price", as_of=span, unit="%",
                              caliber="日收益率标准差 × √252", raw=round(vol, 2)))
    elif not hist_res.ok:
        b.add(dim="行情特征", cog="unknown", lean=None,
              claim="历史 K 线未取到，区间特征无法计算",
              meta=f"接口返回 code={hist_res.code}" if hist_res.code is not None else "请求失败",
              rule="认知为未知，倾向不适用。",
              source={"endpoint": hist_res.endpoint, "params": hist_res.params,
                      "reason": hist_res.reason, "处理": "标记为未知；不补零、不估算"})

    # ---------- 6. 待验证：事件与风险 ----------
    # 注意措辞：我们**只核实了扶摇**，不能把「扶摇拿不到」推断成「所有数据源都拿不到」。
    # 这正是本产品禁止的那种越界 —— 未验证的东西不能被当成结论。
    news_res = log(fuyao.news_events(name, size=1))
    b.add(dim="待验证问题", cog="unknown", lean=None,
          claim="重大事件与风险：扶摇公开接口为端内专用，本维度本轮未覆盖",
          meta=f"扶摇资讯接口实测返回 code={news_res.code}（端内专用）",
          rule="认知为未知，倾向不适用。",
          source={"endpoint": news_res.endpoint, "params": news_res.params,
                  "code": news_res.code, "message": news_res.message,
                  "reason": news_res.reason,
                  "已核实": "扶摇 /api/news/events/search 为同花顺 AI 客户端专用，公开 Key 无权调用",
                  "未核实": "iFinD MCP（mcp.51ifind.com）为需登录的 Web 应用，本项目无其账号，"
                            "未接入，其新闻与事件能力未经实测 —— 不得据此认为该维度在所有数据源上都不可得",
                  "处理": "不静默跳过，明确列为未知；也不把单一数据源的失败外推为全局结论"})

    # ---------- 7. 推断层：全流程唯一使用模型的地方 ----------
    # 只把**事实**喂进去；未知与矛盾不进上下文。引用校验不过的整条丢弃。
    fact_items = [it for it in b.items if it["cog"] == "fact"]
    outcome = llm.generate_inferences(fact_items, question=routing["question"])
    by_id = {it["id"]: it for it in b.items}

    for inf in outcome.inferences:
        leans = [by_id[c]["lean"] for c in inf["cites"] if c in by_id]
        pos, neg = leans.count("pos"), leans.count("neg")
        inf["lean"] = "neg" if neg > pos else ("pos" if pos > neg else "neu")
        # 追加，不覆盖：llm.py 已经写了 L-11 / L-12 的判定说明
        inf["rule"] = (inf.get("rule") or "") + (
            f"<br>L-20　倾向不由模型自评，由规则继承所引用事实的多数方向"
            f"（本次引用 {'、'.join(inf['cites'])}：正面 {pos} / 负面 {neg}）→ "
            + {"neg": "负面", "pos": "正面", "neu": "中性"}[inf["lean"]] + "。"
        )
        b.add(**inf)

    if not outcome.inferences:
        label = {"disabled": "推断层未启用",
                 "empty": "推断层未产出可校验的解读",
                 "error": "推断层调用失败"}.get(outcome.status, "推断层未产出解读")
        b.add(dim="推断解读", cog="unknown", lean=None,
              claim=f"{label}：{outcome.detail}",
              meta=(f"模型 {outcome.provider} / {outcome.model}"
                    if outcome.provider else "未配置模型"),
              rule="认知为未知，倾向不适用。",
              source={"status": outcome.status, "detail": outcome.detail,
                      "处理": "不编造解读；如实标注为未知。事实、未知、矛盾三类证据由代码产出，不受影响"})

    llm_meta = {
        "status": outcome.status,
        "detail": outcome.detail,
        "provider": outcome.provider,
        "model": outcome.model,
        "generated_at": now.isoformat(timespec="seconds"),
        "accepted": len(outcome.inferences),
        "dropped": outcome.dropped,
        "raw_output": outcome.raw[:2000],
    }

    # ---------- 汇总 ----------
    by_cog: dict[str, int] = {}
    by_lean: dict[str, int] = {}
    for it in b.items:
        by_cog[it["cog"]] = by_cog.get(it["cog"], 0) + 1
        k = it["lean"] or "unknown"
        by_lean[k] = by_lean.get(k, 0) + 1

    return {
        "subject": {
            "thscode": thscode,
            "name": name,
            "peers": peers,
            "periods_requested": periods,
            "periods_available": have,
            "same_caliber_type": type_name,
            "quote": quote,
        },
        "generated_at": now.isoformat(timespec="seconds"),
        "evidence": b.items,
        "summary": {"total": len(b.items), "by_cog": by_cog, "by_lean": by_lean},
        "routing": {**routing, "company_type": ctype},
        "llm": llm_meta,
        "fetch_log": fetch_log,
    }


def main(argv: list[str]) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if len(argv) < 2:
        print("用法: python diagnose.py <thscode> [--peers A,B] [--out file.json]")
        return 2

    thscode = argv[1]
    peers: list[str] = []
    out_path = ""
    i = 2
    while i < len(argv):
        if argv[i] == "--peers" and i + 1 < len(argv):
            peers = [s.strip() for s in argv[i + 1].split(",") if s.strip()]
            i += 2
        elif argv[i] == "--out" and i + 1 < len(argv):
            out_path = argv[i + 1]
            i += 2
        else:
            i += 1

    result = diagnose(thscode, peers=peers)

    if out_path:
        from pathlib import Path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"已写入 {out_path}")

    s = result["summary"]
    sub = result["subject"]
    print(f"\n{sub['name']} ({thscode})")
    print(f"证据 {s['total']} 条　认知 {s['by_cog']}　倾向 {s['by_lean']}")
    print(f"同口径基准：{sub['same_caliber_type']}　可用报告期 {len(sub['periods_available'])} 个")
    print(f"取数记录 {len(result['fetch_log'])} 次\n")
    for it in result["evidence"]:
        lean = it["lean"] or "—"
        print(f"[{it['cog']:>7} / {lean:>6}] {it['id']}  {it['claim'][:66]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
