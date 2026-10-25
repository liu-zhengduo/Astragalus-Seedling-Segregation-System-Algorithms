"""批量调用 V1 程序，每张图片使用独立结果目录。"""

import subprocess
import sys
from pathlib import Path


if len(sys.argv) != 3:
    raise SystemExit("用法：python batch_run.py 图片目录 结果目录")

source = Path(sys.argv[1])
target = Path(sys.argv[2])
extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
images = sorted(path for path in source.iterdir() if path.suffix.lower() in extensions)
script = Path(__file__).with_name("seedling_watershed.py")

for index, image in enumerate(images, 1):
    print(f"\n[{index}/{len(images)}] {image.name}")
    subprocess.run(
        [sys.executable, str(script), str(image), "-o", str(target / image.stem)],
        check=True,
    )

print(f"\n批量完成：{len(images)} 张，结果目录 {target}")
