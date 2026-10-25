"""批量运行高精度版本。"""
import subprocess
import sys
from pathlib import Path

if len(sys.argv) != 3:
    raise SystemExit("用法: python batch_run_v2.py 图片目录 结果目录")
source, target = Path(sys.argv[1]), Path(sys.argv[2])
extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
images = sorted(path for path in source.iterdir() if path.suffix.lower() in extensions)
script = Path(__file__).with_name("seedling_grasp_v2.py")
for index, image in enumerate(images, 1):
    print(f"\n[{index}/{len(images)}] {image.name}")
    subprocess.run([sys.executable, str(script), str(image), "-o",
                    str(target / image.stem)], check=True)
print(f"\n批量完成：{len(images)} 张")
