#!/usr/bin/env python3
"""扫描 mht_parts 目录下各 .mht 片段，统计 MIME 头部的 Content-Type。"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def read_mime_headers_prefix(data: bytes, max_scan: int = 65536) -> bytes:
    """取文件开头直到第一个空行（不含），视为该 part 的 MIME 头部。"""
    chunk = data[:max_scan]
    sep = b"\r\n\r\n"
    idx = chunk.find(sep)
    if idx != -1:
        return chunk[:idx]
    sep2 = b"\n\n"
    idx2 = chunk.find(sep2)
    if idx2 != -1:
        return chunk[:idx2]
    return chunk


def extract_content_type_from_headers(header_bytes: bytes) -> str | None:
    text = header_bytes.decode("utf-8", errors="replace")
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key.strip().lower() != "content-type":
            continue
        value = rest.strip()
        if not value:
            continue
        # 只保留主类型（去掉 charset 等参数）
        main = value.split(";")[0].strip().lower()
        return main
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="统计 mht_parts 中各文件的 Content-Type")
    parser.add_argument(
        "-d",
        "--dir",
        type=Path,
        default=Path(__file__).resolve().parent / "mht_parts",
        help="mht 片段目录（默认：脚本同目录下的 mht_parts）",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="列出缺少 Content-Type 的文件（最多 50 个）",
    )
    args = parser.parse_args()

    root: Path = args.dir
    if not root.is_dir():
        raise SystemExit(f"目录不存在: {root}")

    files = sorted(root.glob("part_*.mht"))
    if not files:
        raise SystemExit(f"未找到 part_*.mht: {root}")

    counts: Counter[str] = Counter()
    missing: list[str] = []

    for fp in files:
        data = fp.read_bytes()
        headers = read_mime_headers_prefix(data)
        ct = extract_content_type_from_headers(headers)
        if ct is None:
            missing.append(fp.name)
            counts["<无 Content-Type>"] += 1
        else:
            counts[ct] += 1

    print(f"目录: {root}")
    print(f"文件数: {len(files)}")
    print()
    print("Content-Type 统计（按出现次数降序）：")
    for ctype, n in counts.most_common():
        print(f"  {n:6d}  {ctype}")

    if args.verbose and missing:
        print()
        print(f"未解析到 Content-Type 的文件（共 {len(missing)} 个），示例：")
        for name in missing[:50]:
            print(f"  {name}")


if __name__ == "__main__":
    main()
