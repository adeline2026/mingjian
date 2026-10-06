# -*- coding: utf-8 -*-
"""打包 / 推送前的密钥自检。

用法：  python check_secrets.py
退出码：0 = 干净，可以打包推送；1 = 发现问题，**不要打包、不要推送**

它做三件事：
  1. 用真实密钥的值去扫项目目录的每个文本文件（只比对，不回显密钥）
  2. 检查项目里有没有不该存在的凭据文件
  3. 若当前是 git 仓库，检查 git 已跟踪的文件里有没有敏感文件

为什么需要它：.gitignore 只挡得住 `git add .` 这一条路。
挡不住 `git add -f`、挡不住把文件夹压成 zip 上传、挡不住手动拖拽。
所以推送 / 打包前，用这个脚本做最后一道检查。
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import fuyao

HERE = Path(__file__).resolve().parent

SECRET_NAMES = ("HITHINK_FINANCE_API_KEY", "LLM_API_KEY")

# 文件名看着像凭据的
SUSPECT = re.compile(r"(^|/)(\.env($|\.)|credentials?($|\.)|id_rsa|.*\.(key|pem|p12))$", re.I)

# 只在这些配置类文件里做「长随机串」启发式扫描
CONFIG_SUFFIXES = {".env", ".ini", ".cfg", ".toml"}
CONFIG_NAMES = {"credentials.env", "credentials.env.txt"}
TOKENISH = re.compile(r"\b[A-Za-z0-9_\-]{32,}\b")

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".idea", ".vscode", "out"}
TEXT_SUFFIXES = {"", ".py", ".md", ".txt", ".json", ".html", ".css", ".js", ".yml", ".yaml",
                 ".toml", ".ini", ".cfg", ".env", ".example", ".sh", ".bat", ".gitignore"}


def collect_secret_values() -> dict[str, str]:
    """取真实密钥值，仅用于比对，绝不回显。"""
    found: dict[str, str] = {}
    for name in SECRET_NAMES:
        val = fuyao.load_credential(name)
        if val and len(val) >= 8:
            found[name] = val
    return found


def iter_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            yield Path(dirpath) / fn


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    secrets = collect_secret_values()
    problems: list[str] = []
    warnings: list[str] = []
    scanned = 0

    print("打包 / 推送前密钥自检")
    print(f"  项目目录 : {HERE}")
    print(f"  已知密钥 : {', '.join(sorted(secrets)) if secrets else '（读不到，跳过值比对）'}")
    print()

    for path in iter_files(HERE):
        rel = path.relative_to(HERE).as_posix()
        scanned += 1

        # ① 项目内不该有凭据文件
        is_example = rel.endswith(".example")
        if not is_example and SUSPECT.search("/" + rel):
            problems.append(f"项目内存在凭据类文件：{rel} —— 打包会把它带走")

        # ② 内容里有没有真实密钥
        if path.suffix.lower() not in TEXT_SUFFIXES and path.suffix != "":
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for name, val in secrets.items():
            if val in text:
                problems.append(f"{rel} 的**内容里**包含真实密钥 {name} 的值")

        # ③ 配置类文件里的长随机串（启发式，只提醒）
        if path.suffix.lower() in CONFIG_SUFFIXES or path.name in CONFIG_NAMES:
            m = TOKENISH.search(text)
            if m:
                warnings.append(f"{rel} 里有一个长随机串（{m.group(0)[:6]}…），请确认它不是密钥")

    # ④ git 已跟踪的文件
    if (HERE / ".git").exists():
        try:
            out = subprocess.run(["git", "-C", str(HERE), "ls-files"],
                                 capture_output=True, text=True, timeout=20).stdout
            for f in out.split():
                if not f.endswith(".example") and SUSPECT.search("/" + f):
                    problems.append(
                        f"git 已跟踪凭据文件：{f} —— 需 `git rm --cached {f}` 并清理历史")
            if not out.strip():
                warnings.append("git 仓库还没有跟踪任何文件")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"无法检查 git 跟踪文件：{exc}")
    else:
        warnings.append("当前不是 git 仓库。推送前记得确认 .gitignore 生效"
                        "（用 `git status` 看不到 .env 才算生效）")

    print(f"已扫描 {scanned} 个文件")
    print()
    if problems:
        print("发现问题（必须先解决）：")
        for p in problems:
            print(f"  ✗ {p}")
    if warnings:
        print("提醒：")
        for w in warnings:
            print(f"  ! {w}")
    if not problems:
        print("结论：项目目录内**没有发现任何真实密钥**，可以打包 / 推送。")
    else:
        print("\n结论：**不要打包、不要推送**，先按上面逐条处理。")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
