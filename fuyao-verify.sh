#!/usr/bin/env bash
# 扶摇（同花顺金融数据服务）API 连通性验证
#
# 用法:  bash fuyao-verify.sh
# 说明:  从用户级凭据文件或环境变量读取 Key；脚本全程不回显 Key 本身。

set -u
BASE="https://fuyao.aicubes.cn"
CRED=""
for c in "$HOME/.hithink-finance/credentials.env" \
         "$HOME/.hithink-finance/credentials.env.txt" \
         "$HOME/.hithink-finance/credentials.txt"; do
  if [ -f "$c" ]; then CRED="$c"; break; fi
done

KEY="${HITHINK_FINANCE_API_KEY:-}"
if [ -z "$KEY" ] && [ -n "$CRED" ]; then
  KEY=$(sed -n 's/^[[:space:]]*HITHINK_FINANCE_API_KEY[[:space:]]*=[[:space:]]*//p' "$CRED" | head -1 | tr -d '\r ')
fi
if [ -z "$KEY" ]; then
  echo "未找到 API Key。"
  echo "请写入 $CRED （格式: HITHINK_FINANCE_API_KEY=你的key），或设置同名环境变量。"
  exit 1
fi
echo "已加载 API Key（长度 ${#KEY}，不回显）"
echo

pass=0
fail=0

probe () {
  local label="$1" url="$2" raw body http biz
  raw=$(curl -sS --max-time 30 -w $'\n%{http_code}' -H "X-api-key: $KEY" "$url" 2>&1)
  http="${raw##*$'\n'}"
  body="${raw%$'\n'*}"
  biz=$(printf '%s' "$body" | sed -n 's/.*"code"[[:space:]]*:[[:space:]]*\(-\{0,1\}[0-9]\{1,\}\).*/\1/p' | head -1)
  if [ "$biz" = "0" ]; then
    printf '[PASS] %-18s HTTP %s  code=0\n' "$label" "$http"
    pass=$((pass + 1))
  else
    printf '[FAIL] %-18s HTTP %s  code=%s\n' "$label" "$http" "${biz:-?}"
    fail=$((fail + 1))
  fi
  printf '       %s\n\n' "$(printf '%s' "$body" | head -c 240)"
}

probe "标的检索"        "$BASE/api/meta/tickers/search?q=600519.SH"
probe "行情快照"        "$BASE/api/a-share/prices/snapshot?thscodes=600519.SH"
probe "估值快照"        "$BASE/api/a-share/valuations/snapshot?thscodes=600519.SH"
probe "财务指标2025Q3"  "$BASE/api/a-share/financials/indicators?thscode=600519.SH&report=2025-3"
probe "资讯事件"        "$BASE/api/news/events/search?query=%E8%8C%85%E5%8F%B0&size=1"

echo "=================================="
echo "通过 $pass 项 / 失败 $fail 项"
