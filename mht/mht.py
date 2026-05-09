import re
import os

mht_file = r"/Users/ksnow/Downloads/作者.mht"
output_dir = r"mht_parts"
os.makedirs(output_dir, exist_ok=True)

# 先从头部读取 boundary
with open(mht_file, "rb") as f:
    head = f.read(1024 * 1024)

m = re.search(br'boundary="([^"]+)"', head, re.IGNORECASE)
if not m:
    m = re.search(br"boundary=([^\r\n;]+)", head, re.IGNORECASE)

if not m:
    raise ValueError("没有找到 boundary")

boundary = m.group(1).strip(b'"')
sep = b"--" + boundary
end_sep = b"--" + boundary + b"--"

print("boundary =", boundary.decode(errors="replace"))

part_index = -1
out = None

with open(mht_file, "rb") as f:
    for line in f:
        stripped = line.rstrip(b"\r\n")

        if stripped == sep:
            if out:
                out.close()
            part_index += 1
            out_path = os.path.join(output_dir, f"part_{part_index:04d}.mht")
            out = open(out_path, "wb")
            continue

        elif stripped == end_sep:
            if out:
                out.close()
                out = None
            break

        else:
            if out:
                out.write(line)

if out:
    out.close()

print(f"完成，拆出 {part_index + 1} 个 part")
