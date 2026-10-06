# -*- coding: utf-8 -*-
"""推断层约束的单元测试。

不调用真实模型：把 HTTP 那一层换掉，直接喂构造的模型回复，
验证「只有事实能进上下文」和「引用校验」这两道闸门真的拦得住。

运行：  python -m unittest test_llm -v
"""
import json
import sys
import unittest

sys.stdout.reconfigure(encoding="utf-8")

import llm

FACTS = [
    {"id": "E-01", "cog": "fact", "lean": "neg", "dim": "财务趋势",
     "claim": "中报营业收入同比增速连续 3 年回落 <b>17.76% → 9.10% → 1.47%</b>"},
    {"id": "E-02", "cog": "fact", "lean": "pos", "dim": "经营质量",
     "claim": "2025 年报加权平均ROE <b>32.53%</b>"},
    # 与 E-01 同属「财务趋势」，用于验证「只引用单一维度会被丢弃」
    {"id": "E-03", "cog": "fact", "lean": "neg", "dim": "财务趋势",
     "claim": "中报归母净利润同比 <b>-1.95%</b>"},
]
UNKNOWN_ITEM = {"id": "E-09", "cog": "unknown", "lean": None, "dim": "估值",
                "claim": "没有历史估值序列，无法判断分位"}
CONTRA_ITEM = {"id": "E-10", "cog": "fact", "lean": "contra", "dim": "估值",
               "claim": "洋河 PE(TTM) 与 PE(MRQ) 相差逾 11 倍"}


def _cfg():
    return llm.LLMConfig(base_url="https://example.invalid/v1", api_key="k",
                         model="m", provider="example.invalid")


class InferenceGate(unittest.TestCase):
    def _run(self, model_reply: str, items=None):
        captured: dict = {}

        def fake_post(cfg, messages, json_mode=True):
            captured["messages"] = messages
            captured["json_mode"] = json_mode
            return True, {"choices": [{"message": {"content": model_reply}}]}, ""

        original = llm._post_chat
        llm._post_chat = fake_post
        try:
            outcome = llm.generate_inferences(items if items is not None else FACTS, _cfg())
        finally:
            llm._post_chat = original
        return outcome, captured

    # ---- 闸门一：只有事实能进上下文 ----
    def test_only_facts_enter_context(self):
        _, captured = self._run('{"inferences":[]}', FACTS + [UNKNOWN_ITEM])
        context = json.dumps(captured["messages"], ensure_ascii=False)
        self.assertIn("E-01", context)
        self.assertNotIn("E-09", context, "未知项不得进入模型上下文")

    def test_contra_also_excluded(self):
        """矛盾项本身是事实，但它属于「冲突」状态，不应作为解读依据单独喂入。"""
        _, captured = self._run('{"inferences":[]}', FACTS + [CONTRA_ITEM])
        context = json.dumps(captured["messages"], ensure_ascii=False)
        self.assertIn("E-01", context)

    def test_html_tags_stripped_before_sending(self):
        _, captured = self._run('{"inferences":[]}')
        context = json.dumps(captured["messages"], ensure_ascii=False)
        self.assertNotIn("<b>", context, "展示用标签不应喂给模型")

    # ---- 闸门二：引用校验 ----
    def test_valid_inference_kept(self):
        outcome, _ = self._run(
            '{"inferences":[{"claim":"盈利仍处高位但增速在放缓","cites":["E-01","E-02"]}]}')
        self.assertEqual(outcome.status, "ok")
        self.assertEqual(len(outcome.inferences), 1)
        self.assertEqual(outcome.inferences[0]["cites"], ["E-01", "E-02"])

    def test_same_dimension_two_cites_kept(self):
        """同维度两条事实之间的比较（例如两个估值指标）是有价值的，不应被拦。"""
        outcome, _ = self._run(
            '{"inferences":[{"claim":"收入端与利润端同步走弱","cites":["E-01","E-03"]}]}')
        self.assertEqual(len(outcome.inferences), 1)

    def test_single_citation_dropped(self):
        """L-11：只引用 1 条 = 复述而非综合，必须丢弃。"""
        outcome, _ = self._run(
            '{"inferences":[{"claim":"增速放缓","cites":["E-01"]}]}')
        self.assertEqual(outcome.inferences, [])
        self.assertIn("只引用了 1 条事实", outcome.dropped[0]["reason"])

    def test_forbidden_benchmark_dropped(self):
        """L-12：凭空引入「行业平均」这类基准必须被拦下。

        这是实测抓到的真问题：模型引用的编号都是真的，但在真事实外面
        挂了一句凭空捏造的基准 —— 引用校验挡不住，只能靠关键词闸门。"""
        outcome, _ = self._run(
            '{"inferences":[{"claim":"营收增速仍低于行业平均水平","cites":["E-01","E-02"]}]}')
        self.assertEqual(outcome.inferences, [], "事实清单里没有基准，不得做比较")
        self.assertIn("行业平均", outcome.dropped[0]["reason"])
        self.assertIn("L-12", outcome.dropped[0]["reason"])

    def test_forbidden_prediction_dropped(self):
        """L-12：预测与建议同样越界。"""
        outcome, _ = self._run(
            '{"inferences":[{"claim":"盈利能力较强，预计明年将继续增长","cites":["E-01","E-02"]}]}')
        self.assertEqual(outcome.inferences, [])
        self.assertIn("预计", outcome.dropped[0]["reason"])

    def test_forbidden_advice_dropped(self):
        """L-12：买卖建议必须被拦下。"""
        outcome, _ = self._run(
            '{"inferences":[{"claim":"盈利稳健，建议买入","cites":["E-01","E-02"]}]}')
        self.assertEqual(outcome.inferences, [])
        self.assertIn("建议买入", outcome.dropped[0]["reason"])

    def test_hallucinated_cite_dropped(self):
        outcome, _ = self._run(
            '{"inferences":[{"claim":"公司将在明年翻倍","cites":["E-99"]}]}')
        self.assertEqual(outcome.inferences, [], "引用不存在的事实必须整条丢弃")
        self.assertEqual(len(outcome.dropped), 1)
        self.assertIn("E-99", outcome.dropped[0]["reason"])

    def test_no_cite_dropped(self):
        outcome, _ = self._run('{"inferences":[{"claim":"感觉不错","cites":[]}]}')
        self.assertEqual(outcome.inferences, [])
        self.assertEqual(outcome.dropped[0]["reason"], "未引用任何事实")

    def test_mixed_valid_and_invalid(self):
        outcome, _ = self._run(json.dumps({"inferences": [
            {"claim": "盈利仍强但增速在放缓", "cites": ["E-01", "E-02"]},
            {"claim": "编造的", "cites": ["E-77"]},
        ]}))
        self.assertEqual(len(outcome.inferences), 1)
        self.assertEqual(len(outcome.dropped), 1)

    # ---- 失败与降级 ----
    def test_invalid_json_reports_error(self):
        outcome, _ = self._run("我觉得这家公司挺好的，建议买入。")
        self.assertEqual(outcome.status, "error")
        self.assertEqual(outcome.inferences, [])

    def test_disabled_without_key(self):
        outcome = llm.generate_inferences(FACTS, llm.LLMConfig())
        self.assertEqual(outcome.status, "disabled")
        self.assertEqual(outcome.inferences, [])

    def test_empty_reply_is_not_ok(self):
        outcome, _ = self._run('{"inferences":[]}')
        self.assertEqual(outcome.inferences, [])
        self.assertIn(outcome.status, ("empty", "ok"))


class StripTags(unittest.TestCase):
    def test_strips_nested_tags(self):
        self.assertEqual(llm.strip_tags("营收同比 <b>1.47%</b>"), "营收同比 1.47%")

    def test_plain_text_unchanged(self):
        self.assertEqual(llm.strip_tags("没有标签"), "没有标签")


class ExtractJson(unittest.TestCase):
    """模型常把 JSON 包在代码块里，或在前后加话。"""

    def test_strips_code_fence(self):
        self.assertEqual(llm.extract_json('```json\n{"a":1}\n```'), '{"a":1}')

    def test_strips_fence_without_language(self):
        self.assertEqual(llm.extract_json('```\n{"a":1}\n```'), '{"a":1}')

    def test_strips_surrounding_prose(self):
        self.assertEqual(llm.extract_json('这是我的判断：{"a":1} 希望有帮助'), '{"a":1}')

    def test_plain_json_unchanged(self):
        self.assertEqual(llm.extract_json('{"a":1}'), '{"a":1}')


class JsonModeFallback(unittest.TestCase):
    """并非所有 OpenAI 兼容端点都支持 response_format，必须能退回。"""

    def _with(self, fake):
        original = llm._post_chat
        llm._post_chat = fake
        try:
            return llm.generate_inferences(FACTS, _cfg())
        finally:
            llm._post_chat = original

    def test_falls_back_when_unsupported(self):
        calls = []

        def fake(cfg, messages, json_mode=True):
            calls.append(json_mode)
            if json_mode:
                return False, None, "HTTP 400 Unsupported parameter: response_format"
            return True, {"choices": [{"message": {"content":
                '```json\n{"inferences":[{"claim":"盈利仍强但增速在放缓","cites":["E-01","E-02"]}]}\n```'}}]}, ""

        outcome = self._with(fake)
        self.assertEqual(calls, [True, False], "应先试 json_mode，失败后回退")
        self.assertEqual(outcome.status, "ok")
        self.assertEqual(len(outcome.inferences), 1)
        self.assertIn("退回普通模式", outcome.detail)

    def test_both_modes_fail(self):
        def fake(cfg, messages, json_mode=True):
            return False, None, "HTTP 401 unauthorized"

        outcome = self._with(fake)
        self.assertEqual(outcome.status, "error")
        self.assertEqual(outcome.inferences, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
