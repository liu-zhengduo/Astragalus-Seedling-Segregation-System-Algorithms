# -*- coding: utf-8 -*-
"""OpenCV 分水岭：输出深色长条苗的像素质心和夹爪姿态角。"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


VERSION = "1.1.0"

parser = argparse.ArgumentParser(description="V1 深色长条苗分水岭检测工具")
parser.add_argument("image", help="待检测图片")
parser.add_argument("-o", "--out", default="result", help="结果目录")
parser.add_argument("--show", action="store_true", help="逐步弹窗显示（PyCharm 可用）")
parser.add_argument("--version", action="version", version=f"Seedling Watershed V{VERSION}")
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
print(f"Seedling Watershed V{VERSION}")


def save(step, name, image):
    """保存阶段图；启用 --show 时同时弹窗。"""
    path = out / f"{step:02d}_{name}.png"
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"图片编码失败：{path}")
    encoded.tofile(str(path))
    print(f"步骤 {step}: {name} -> {path}")
    if args.show:
        cv2.imshow(name, image)
        cv2.waitKey(0)
        cv2.destroyWindow(name)


def angle_difference(a, b):
    """计算两个无方向直线角度的最小夹角。"""
    difference = abs(a - b) % 180
    return min(difference, 180 - difference)


# 1. 读取原图（兼容中文路径）。
image = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
if image is None:
    raise FileNotFoundError(args.image)
height, width = image.shape[:2]
scale = min(height, width)
save(1, "original", image)

# 2. 灰度化和 CLAHE，减弱工业现场照明不均。
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
enhanced = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
save(2, "enhanced", enhanced)

# 3. 闭运算估计亮背景，再相减得到深色长条响应。
kernel_size = max(15, int(scale * 0.075) | 1)
background = cv2.morphologyEx(
    enhanced,
    cv2.MORPH_CLOSE,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
)
contrast = cv2.GaussianBlur(cv2.subtract(background, enhanced), (3, 3), 0)
save(3, "dark_contrast", contrast)

# 4. Otsu 自动阈值、轻微开运算去散点，并清除托盘边框。
_, mask = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
small_kernel = max(2, round(scale / 250))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((small_kernel, small_kernel), np.uint8))
edge = max(3, round(scale * 0.025))
mask[:edge] = 0
mask[-edge:] = 0
mask[:, :edge] = 0
mask[:, -edge:] = 0
save(4, "binary_mask", mask)

# 5. 欧氏距离变换：值越大，越靠近长条内部中心。
distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
heat = cv2.applyColorMap(
    np.uint8(255 * distance / max(float(distance.max()), 1)), cv2.COLORMAP_TURBO
)
save(5, "distance_transform", heat)

# 6. 距离核心作为分水岭种子；无种子的有效连通域补一个局部峰值种子。
core = np.uint8(distance >= 0.31 * distance.max()) * 255
component_count, labels, stats, _ = cv2.connectedComponentsWithStats(core)
seeds = np.zeros_like(mask)
minimum_seed_area = max(3, round(height * width * 0.00002))
for component_id in range(1, component_count):
    if stats[component_id, cv2.CC_STAT_AREA] >= minimum_seed_area:
        seeds[labels == component_id] = 255

component_count, objects, stats, _ = cv2.connectedComponentsWithStats(mask)
minimum_area = max(20, round(height * width * 0.00012))
for component_id in range(1, component_count):
    region = objects == component_id
    if stats[component_id, cv2.CC_STAT_AREA] >= minimum_area and not np.any(seeds[region]):
        y, x = np.unravel_index(np.argmax(np.where(region, distance, -1)), distance.shape)
        cv2.circle(seeds, (int(x), int(y)), 1, 255, -1)

seed_count, seed_labels = cv2.connectedComponents(seeds)
seed_view = cv2.applyColorMap(
    np.uint8(seed_labels * 255 / max(seed_count - 1, 1)), cv2.COLORMAP_HSV
)
seed_view[mask == 0] = 0
save(6, "watershed_seeds", seed_view)

# 7. 未知区域置 0、背景置 1，执行 marker-controlled watershed。
markers = seed_labels.astype(np.int32) + 1
sure_background = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
markers[sure_background == 0] = 1
markers[(sure_background > 0) & (seeds == 0)] = 0
watershed = cv2.watershed(image.copy(), markers)
boundary = image.copy()
boundary[watershed == -1] = (255, 0, 0)
save(7, "watershed_boundary", boundary)

# 8. 筛选细长区域，计算质心与主轴角（相对图像 +x，顺时针，范围 [-90, 90)）。
raw_detections = []
minimum_length = scale * 0.035
for label_id in range(2, int(watershed.max()) + 1):
    region = np.uint8((watershed == label_id) & (mask > 0)) * 255
    # OpenCV 3 返回 3 项，OpenCV 4 返回 2 项；倒数第二项始终是轮廓列表。
    contour_result = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contour_result[-2]
    if not contours:
        continue
    contour = max(contours, key=cv2.contourArea)
    (_, _), (rect_width, rect_height), angle = cv2.minAreaRect(contour)
    long_side = max(rect_width, rect_height)
    short_side = max(1.0, min(rect_width, rect_height))
    ys, xs = np.where(region > 0)
    if len(xs) < minimum_area or long_side < minimum_length or long_side / short_side < 1.6:
        continue
    angle = ((angle + (0 if rect_width >= rect_height else 90) + 90) % 180) - 90
    raw_detections.append(
        {
            "x": float(xs.mean()),
            "y": float(ys.mean()),
            "angle": angle,
            "area": len(xs),
            "length": long_side,
        }
    )

# 合并同一条弯苗上相邻、近共线的小段。
detections = []
for detection in sorted(raw_detections, key=lambda item: item["area"], reverse=True):
    match = None
    for merged in detections:
        dx = detection["x"] - merged["x"]
        dy = detection["y"] - merged["y"]
        connecting_angle = math.degrees(math.atan2(dy, dx))
        close = math.hypot(dx, dy) < 0.14 * scale
        parallel = angle_difference(detection["angle"], merged["angle"]) < 35
        collinear = max(
            angle_difference(detection["angle"], connecting_angle),
            angle_difference(merged["angle"], connecting_angle),
        ) < 20
        if close and parallel and collinear:
            match = merged
            break
    if match:
        total_area = match["area"] + detection["area"]
        match["x"] = (match["x"] * match["area"] + detection["x"] * detection["area"]) / total_area
        match["y"] = (match["y"] * match["area"] + detection["y"] * detection["area"]) / total_area
        match["area"] = total_area
    else:
        detections.append(detection.copy())

detections.sort(key=lambda item: (item["y"], item["x"]))
result = image.copy()
for index, detection in enumerate(detections, 1):
    x = round(detection["x"])
    y = round(detection["y"])
    angle = detection["angle"]
    line_length = max(18, round(detection["length"] * 0.45))
    endpoint = (
        round(x + line_length * math.cos(math.radians(angle))),
        round(y + line_length * math.sin(math.radians(angle))),
    )
    cv2.circle(result, (x, y), 5, (0, 0, 255), -1)
    cv2.line(result, (x, y), endpoint, (255, 0, 255), 2)
    cv2.putText(result, str(index), (x + 6, y - 6), 0, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
    detection["id"] = index

save(8, "result", result)
csv_path = out / "poses.csv"
with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow(["id", "center_x_px", "center_y_px", "angle_deg", "area_px"])
    writer.writerows(
        (
            detection["id"],
            f'{detection["x"]:.2f}',
            f'{detection["y"]:.2f}',
            f'{detection["angle"]:.2f}',
            detection["area"],
        )
        for detection in detections
    )
print(f"完成：识别 {len(detections)} 个候选，位姿已保存到 {csv_path}")
