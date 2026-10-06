# -*- coding: utf-8 -*-
"""规则引擎的单元测试：公司类型识别 + 问题路由。

都是纯函数，不需要网络、不需要 Key。

起因是实测中发现的一个真实误判：*ST八钢（钢铁，资产负债率 106%、净利率 −10%）
被自动识别成「金融/地产」，于是 L-05 的资产负债率条件被豁免 ——
**恰好放过了最该报警的那家公司**。这里把它固定成回归测试。

运行：  python -m unittest test_rules -v
"""
import sys
import unittest

sys.stdout.reconfigure(encoding="utf-8")

import diagnose


class RangeStats(unittest.TestCase):
    """行情区间统计。纯计算，不联网。"""

    @staticmethod
    def _bars(closes):
        return [{"close_price": str(c), "high_price": str(c * 1.01),
                 "low_price": str(c * 0.99), "date_ms": 1700000000000 + i * 86400000}
                for i, c in enumerate(closes)]

    def test_uptrend_change_and_drawdown(self):
        st = diagnose._range_stats(self._bars([100, 110, 120, 130, 140]))
        self.assertEqual(st["days"], 5)
        self.assertAlmostEqual(st["change_pct"], 40.0, places=4)
        self.assertAlmostEqual(st["max_drawdown_pct"], 0.0, places=4)

    def test_drawdown_detected(self):
        st = diagnose._range_stats(self._bars([100, 200, 100]))
        self.assertAlmostEqual(st["change_pct"], 0.0, places=4)
        self.assertAlmostEqual(st["max_drawdown_pct"], 50.0, places=4)

    def test_decline(self):
        st = diagnose._range_stats(self._bars([100, 80, 60]))
        self.assertAlmostEqual(st["change_pct"], -40.0, places=4)

    def test_missing_values_excluded_not_zero_filled(self):
        bars = self._bars([100, 110, 120])
        bars.insert(1, {"close_price": None, "high_price": None, "low_price": None,
                        "date_ms": 1700000000000})
        st = diagnose._range_stats(bars)
        self.assertEqual(st["days"], 3, "缺失的收盘价不应被当成 0 计入")

    def test_too_few_bars_returns_empty(self):
        self.assertEqual(diagnose._range_stats(self._bars([100])), {})
        self.assertEqual(diagnose._range_stats([]), {})


class CompanyType(unittest.TestCase):
    """取值全部来自实测：招商银行 / *ST八钢 / 贵州茅台 2025 年报口径。"""

    def test_bank_is_financial(self):
        """高负债 + 高净利率 = 经营模式本身就是高杠杆。"""
        ct = diagnose.infer_company_type({
            "assets_debt_ratio": "90.2001",
            "sale_net_interest_ratio": "44.7738",
        })
        self.assertEqual(ct["type"], "金融/地产")
        self.assertEqual(ct["rule"], "T-01")
        self.assertIn("不适用", "；".join(ct["affected"]))

    def test_distressed_manufacturer_is_not_financial(self):
        """回归：高负债 + 负净利率 = 困境，不是金融业，**不得豁免**。"""
        ct = diagnose.infer_company_type({
            "assets_debt_ratio": "106.1787",
            "sale_net_interest_ratio": "-10.0295",
            "sale_gross_margin": "1.0196",
        })
        self.assertEqual(ct["type"], "高杠杆（非金融）")
        self.assertEqual(ct["rule"], "T-03")
        self.assertIn("照常适用", "；".join(ct["affected"]),
                      "困境企业的高负债必须照常作为风险信号，不能被当成金融业豁免")

    def test_high_debt_with_missing_margin_is_not_financial(self):
        """净利率缺失时不能默认往金融业靠 —— 宁可归为高杠杆。"""
        ct = diagnose.infer_company_type({"assets_debt_ratio": "85"})
        self.assertEqual(ct["type"], "高杠杆（非金融）")

    def test_high_margin_consumer(self):
        ct = diagnose.infer_company_type({
            "assets_debt_ratio": "16.4154",
            "sale_gross_margin": "91.1796",
            "sale_net_interest_ratio": "50.5279",
        })
        self.assertEqual(ct["type"], "高毛利（消费/软件）")
        self.assertEqual(ct["rule"], "T-02")

    def test_general(self):
        ct = diagnose.infer_company_type({
            "assets_debt_ratio": "45", "sale_gross_margin": "25",
            "sale_net_interest_ratio": "8",
        })
        self.assertEqual(ct["type"], "通用")

    def test_empty_input_does_not_crash(self):
        """全部指标缺失时不得抛异常，退化为通用。"""
        self.assertEqual(diagnose.infer_company_type({})["type"], "通用")

    def test_null_values_do_not_crash(self):
        self.assertEqual(
            diagnose.infer_company_type({
                "assets_debt_ratio": None,
                "sale_gross_margin": "null",
                "sale_net_interest_ratio": "",
            })["type"], "通用")


class QuestionRouting(unittest.TestCase):
    def test_specific_query_beats_generic_suffix(self):
        """回归：Q-02 必须赢过 Q-01 的泛化关键词「怎么样」。"""
        self.assertEqual(diagnose.route_question("它赚钱能力怎么样")["rule"], "Q-02")

    def test_valuation_route(self):
        r = diagnose.route_question("现在贵不贵")
        self.assertEqual(r["rule"], "Q-03")
        self.assertEqual(r["dimensions"], ["估值"])

    def test_risk_route_includes_pending(self):
        r = diagnose.route_question("有没有需要警惕的事")
        self.assertEqual(r["rule"], "Q-04")
        self.assertIn("待验证问题", r["dimensions"])

    def test_overview_route(self):
        r = diagnose.route_question("这家公司现在什么状态")
        self.assertEqual(r["rule"], "Q-01")
        self.assertEqual(len(r["dimensions"]), 4)

    def test_unknown_question_shows_all_without_guessing(self):
        r = diagnose.route_question("今天天气不错")
        self.assertEqual(r["rule"], "Q-99")
        self.assertEqual(len(r["dimensions"]), 4)

    def test_empty_question_shows_all(self):
        r = diagnose.route_question("")
        self.assertEqual(r["rule"], "Q-00")
        self.assertEqual(len(r["dimensions"]), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
