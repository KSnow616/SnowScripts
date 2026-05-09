#!/usr/bin/env python3
"""从 mht_parts 中的 MIME 片段提取图片，按 Content-Type 保存为对应扩展名。"""

from __future__ import annotations

import argparse
import base64
import binascii
import re
from pathlib import Path


MIME_TO_EXT: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/pjpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/x-ms-bmp": ".bmp",
}


def split_headers_body(data: bytes) -> tuple[bytes, bytes]:
    for sep in (b"\r\n\r\n", b"\n\n"):
        idx = data.find(sep)
        if idx != -1:
            return data[:idx], data[idx + len(sep) :]
    return data, b""


def parse_headers(header_bytes: bytes) -> dict[str, str]:
    text = header_bytes.decode("utf-8", errors="replace")
    out: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        out[key.strip().lower()] = rest.strip()
    return out


def extension_for_content_type(ct_main: str) -> str | None:
    ct = ct_main.strip().lower()
    if ct in MIME_TO_EXT:
        return MIME_TO_EXT[ct]
    if ct.startswith("image/"):
        sub = ct.split("/", 1)[1]
        sub = re.sub(r"[^a-z0-9]", "", sub)[:16]
        if sub:
            return f".{sub}"
    return None


def decode_body(body: bytes, cte: str) -> bytes | None:
    cte_l = cte.lower()
    if "base64" in cte_l:
        cleaned = b"".join(body.split())
        try:
            return base64.b64decode(cleaned, validate=False)
        except binascii.Error:
            return None
    if "quoted-printable" in cte_l:
        import quopri

        return quopri.decodestring(body.replace(b"\r\n", b"\n"))
    # 7bit / 8bit / binary：按原始字节写入（少数情况）
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description="从 mht_parts 提取图片到独立文件")
    parser.add_argument(
        "-d",
        "--dir",
        type=Path,
        default=Path(__file__).resolve().parent / "mht_parts",
        help="mht 片段目录",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "mht_images",
        help="输出目录",
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=10000,
        help="只处理 part_0000 … part_(n-1)（默认 1000）",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="减少输出")
    args = parser.parse_args()

    src: Path = args.dir
    out_root: Path = args.out
    if not src.is_dir():
        raise SystemExit(f"目录不存在: {src}")

    out_root.mkdir(parents=True, exist_ok=True)

    extracted = 0
    skipped_non_image = 0
    failed = 0

    for i in range(args.limit):
        part_path = src / f"part_{i:04d}.mht"
        if not part_path.is_file():
            if not args.quiet:
                print(f"跳过（文件不存在）: {part_path.name}")
            continue

        data = part_path.read_bytes()
        header_b, body_b = split_headers_body(data)
        headers = parse_headers(header_b)
        ct_raw = headers.get("content-type", "")
        ct_main = ct_raw.split(";")[0].strip().lower()

        if not ct_main.startswith("image/"):
            skipped_non_image += 1
            continue

        ext = extension_for_content_type(ct_main)
        if ext is None:
            failed += 1
            if not args.quiet:
                print(f"无法映射类型 {ct_main}: {part_path.name}")
            continue

        cte = headers.get("content-transfer-encoding", "")
        raw = decode_body(body_b, cte)
        if raw is None or len(raw) == 0:
            failed += 1
            if not args.quiet:
                print(f"解码失败: {part_path.name}")
            continue

        out_path = out_root / f"part_{i:04d}{ext}"
        out_path.write_bytes(raw)
        extracted += 1

    print(f"输出目录: {out_root}")
    print(f"已写入图片: {extracted}")
    print(f"非图片跳过: {skipped_non_image}")
    if failed:
        print(f"失败/未知: {failed}")


if __name__ == "__main__":
    main()
