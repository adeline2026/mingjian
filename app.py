# -*- coding: utf-8 -*-
"""本地 Web 服务：把证据引擎包成一个可打开的网页。

零第三方依赖，只用标准库 —— 不用装任何东西就能跑：
    python app.py
然后打开 http://127.0.0.1:8765/

接口：
    GET /                      诊断页面
    GET /api/diagnose?code=... 诊断结果 JSON
"""
from __future__ import annotations

import json
import sys
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import diagnose

HERE = Path(__file__).resolve().parent
WEB_DIR = HERE / "web"
DEFAULT_PEERS = "000858.SZ,000568.SZ,002304.SZ"
HOST, PORT = "127.0.0.1", 8765


def normalize_code(raw: str) -> str:
    """接受 600519 / 600519.SH / 000858.sz 三种写法。只补后缀，不猜标的。"""
    code = raw.strip().upper()
    if not code:
        return ""
    if "." in code:
        return code
    if len(code) == 6 and code.isdigit():
        if code.startswith(("6", "9", "5")):
            return code + ".SH"
        if code.startswith(("0", "3", "2", "1")):
            return code + ".SZ"
        if code.startswith(("4", "8")):
            return code + ".BJ"
    return code


class Handler(BaseHTTPRequestHandler):
    server_version = "MingJian/0.1"

    def log_message(self, fmt, *args):  # 收敛日志噪音
        sys.stderr.write("[http] " + (fmt % args) + "\n")

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, obj) -> None:
        self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path in ("/", "/index.html"):
            index = WEB_DIR / "index.html"
            if not index.is_file():
                self._send(500, "找不到 web/index.html".encode("utf-8"),
                           "text/plain; charset=utf-8")
                return
            self._send(200, index.read_bytes(), "text/html; charset=utf-8")
            return

        if parsed.path == "/api/diagnose":
            qs = urllib.parse.parse_qs(parsed.query)
            code = normalize_code((qs.get("code") or [""])[0])
            peers_raw = (qs.get("peers") or [DEFAULT_PEERS])[0]
            peers = [normalize_code(p) for p in peers_raw.split(",") if p.strip()]

            if not code:
                self._json(400, {"error": "缺少参数 code", "示例": "/api/diagnose?code=600519.SH"})
                return
            try:
                result = diagnose.diagnose(code, peers=peers)
            except Exception:  # noqa: BLE001 - 任何异常都要变成结构化错误，不能白屏
                self._json(500, {"error": "诊断过程异常",
                                 "traceback": traceback.format_exc()[-1800:]})
                return
            self._json(200, result)
            return

        self._json(404, {"error": "not found", "path": parsed.path})


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if not diagnose.fuyao.load_api_key():
        print("警告：未找到扶摇 API Key。")
        print("请设置环境变量 HITHINK_FINANCE_API_KEY，或写入 ~/.hithink-finance/credentials.env")
        print("页面仍会启动，但所有取数都会如实失败并显示为「未知证据」。\n")

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"明鉴已启动： http://{HOST}:{PORT}/")
    print("按 Ctrl+C 停止。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
