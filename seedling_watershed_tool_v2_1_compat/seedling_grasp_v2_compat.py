# -*- coding: utf-8 -*-
"""高精度版：分水岭 + 厚度核心 + 骨架/Hough，输出抓取点、角度和置信度。"""
import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("image", help="待检测图片")
parser.add_argument("-o", "--out", default="result_v2", help="结果目录")
parser.add_argument("--show", action="store_true", help="逐步弹窗显示")
parser.add_argument("--core-radius", type=float, default=0.004,
                    help="最小半径/短边，越大越偏向粗壮主体，默认 0.004")
parser.add_argument("--min-confidence", type=float, default=0.0,
                    help="仅输出不低于该置信度的结果，默认全部输出")
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)


def save(step, name, image):
    path = out / f"{step:02d}_{name}.png"
    cv2.imencode(".png", image)[1].tofile(str(path))
    print(f"步骤{step}: {name} -> {path}")
    if args.show:
        cv2.imshow(name, image)
        cv2.waitKey(0)
        cv2.destroyWindow(name)


def angle_diff(a, b):
    value = abs(a - b) % 180
    return min(value, 180 - value)


def skeletonize(binary):
    """OpenCV 形态学骨架，避免增加 scikit-image 依赖。"""
    work = binary.copy()
    skeleton = np.zeros_like(work)
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(work):
        opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, cross)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(work, opened))
        work = cv2.erode(work, cross, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return skeleton


def prune_skeleton(skeleton, min_length):
    """反复删除连接到交叉点的短毛刺，避免根须噪声制造假分支。"""
    sk = np.uint8(skeleton > 0)
    for _ in range(8):
        degree = cv2.filter2D(sk, -1, np.ones((3, 3), np.uint8),
                              borderType=cv2.BORDER_CONSTANT) - sk
        endpoints = list(zip(*np.where((sk > 0) & (degree == 1))))
        remove = set()
        for start in endpoints:
            path = [start]
            previous = None
            current = start
            while len(path) <= min_length:
                cy, cx = current
                neighbors = []
                for ny in range(max(0, cy - 1), min(sk.shape[0], cy + 2)):
                    for nx in range(max(0, cx - 1), min(sk.shape[1], cx + 2)):
                        point = (ny, nx)
                        if sk[point] and point != current and point != previous:
                            neighbors.append(point)
                if len(neighbors) != 1:
                    break
                previous, current = current, neighbors[0]
                path.append(current)
                if degree[current] != 2:
                    break
            if len(path) <= min_length and degree[current] >= 3:
                remove.update(path[:-1])
        if not remove:
            break
        for point in remove:
            sk[point] = 0
    return sk * 255


def weighted_pose(xs, ys, weights):
    """距离平方加权 PCA：抓取点更靠近粗壮部位，主轴给出姿态角。"""
    total = float(weights.sum())
    cx = float((xs * weights).sum() / total)
    cy = float((ys * weights).sum() / total)
    points = np.column_stack((xs - cx, ys - cy))
    covariance = (points * weights[:, None]).T @ points / total
    vector = np.linalg.eigh(covariance)[1][:, -1]
    angle = math.degrees(math.atan2(vector[1], vector[0]))
    return cx, cy, ((angle + 90) % 180) - 90


def project_to_pixels(cx, cy, xs, ys):
    """把数学质心投影回粗芯像素，保证抓取点落在苗体上。"""
    index = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
    return float(xs[index]), float(ys[index])


# 1. 读取原图，兼容中文路径
image = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
if image is None:
    raise FileNotFoundError(args.image)
h, w = image.shape[:2]
scale = min(h, w)
save(1, "original", image)

# 2. 灰度与局部对比度增强
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
enhanced = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
save(2, "enhanced", enhanced)

# 3. 用大尺度闭运算估计亮背景，突出深色苗体
kernel_size = max(15, int(scale * 0.075) | 1)
background = cv2.morphologyEx(
    enhanced, cv2.MORPH_CLOSE,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))
contrast = cv2.GaussianBlur(cv2.subtract(background, enhanced), (3, 3), 0)
save(3, "dark_contrast", contrast)

# 4. Otsu 二值化并清除相机黑框
_, mask = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
small = max(2, round(scale / 250))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((small, small), np.uint8))
edge = max(3, round(scale * 0.025))
mask[:edge] = mask[-edge:] = 0
mask[:, :edge] = mask[:, -edge:] = 0
save(4, "binary_mask", mask)

# 5. 欧氏距离变换：细根距离小，粗壮可抓取主体距离大
distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
heat = cv2.applyColorMap(
    np.uint8(255 * distance / max(float(distance.max()), 1)), cv2.COLORMAP_TURBO)
save(5, "distance_transform", heat)

# 6. 取厚度核心；同一粗芯若有交叉，再用骨架和 Hough 拆分方向
core = np.uint8(distance >= scale * args.core_radius) * 255
core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
component_count, core_labels, stats, _ = cv2.connectedComponentsWithStats(core)
raw_candidates = []
skeleton_all = np.zeros_like(mask)

for label_id in range(1, component_count):
    if stats[label_id, cv2.CC_STAT_AREA] < scale * scale * 0.000015:
        continue
    x, y, bw, bh, _ = stats[label_id]
    component = np.uint8(core_labels[y:y + bh, x:x + bw] == label_id) * 255
    skeleton = prune_skeleton(skeletonize(component), max(3, round(scale * 0.02)))
    skeleton_all[y:y + bh, x:x + bw] = cv2.bitwise_or(
        skeleton_all[y:y + bh, x:x + bw], skeleton)
    one = np.uint8(skeleton > 0)
    degree = cv2.filter2D(one, -1, np.ones((3, 3), np.uint8),
                          borderType=cv2.BORDER_CONSTANT) - one
    junction = np.uint8((one > 0) & (degree >= 3)) * 255
    branch_count = cv2.connectedComponents(
        cv2.dilate(junction, np.ones((3, 3), np.uint8)))[0] - 1
    endpoint_count = cv2.connectedComponents(np.uint8((one > 0) & (degree == 1)))[0] - 1
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (_, _), (rw, rh), _ = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    long_side, short_side = max(rw, rh), max(1.0, min(rw, rh))
    global_y, global_x = np.where(core_labels == label_id)

    # 分支少或整体高度细长时，作为一条连续弯苗处理
    if branch_count <= 2 or long_side / short_side > 3.0:
        weights = np.maximum(distance[global_y, global_x], 1) ** 2
        cx, cy, angle = weighted_pose(global_x, global_y, weights)
        cx, cy = project_to_pixels(cx, cy, global_x, global_y)
        if long_side >= scale * 0.035:
            raw_candidates.append(dict(x=cx, y=cy, angle=angle, length=long_side,
                                       method="PCA", core_id=label_id))
        continue

    # X/L/Y 形粗芯使用概率 Hough 提取不同穿越方向
    lines = cv2.HoughLinesP(
        skeleton, 1, np.pi / 180,
        threshold=max(8, round(scale * 0.012)),
        minLineLength=max(12, round(scale * 0.045)),
        maxLineGap=max(4, round(scale * 0.02)))
    chosen = []
    if lines is not None:
        segments = []
        # OpenCV 不同版本可能返回 (N, 1, 4)、(N, 4) 或单条 (4,)。
        # 统一展平为 N×4，避免 lines[:, 0] 在部分版本中变成标量序列。
        for x1, y1, x2, y2 in np.asarray(lines, dtype=np.int32).reshape(-1, 4):
            length = math.hypot(x2 - x1, y2 - y1)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            segments.append(dict(length=length, mx=(x1 + x2) / 2 + x,
                                 my=(y1 + y2) / 2 + y, angle=angle,
                                 x1=x1 + x, y1=y1 + y, x2=x2 + x, y2=y2 + y))
        for segment in sorted(segments, key=lambda item: item["length"], reverse=True):
            duplicate = False
            for old in chosen:
                ux = math.cos(math.radians(old["angle"]))
                uy = math.sin(math.radians(old["angle"]))
                perpendicular = abs((segment["mx"] - old["mx"]) * uy -
                                    (segment["my"] - old["my"]) * ux)
                if angle_diff(segment["angle"], old["angle"]) < 22 and \
                        perpendicular < scale * 0.04:
                    duplicate = True
                    break
            if not duplicate:
                chosen.append(segment)

    max_lines = max(2, min(5, round(endpoint_count / 2)))
    for segment in chosen[:max_lines]:
        vx = segment["x2"] - segment["x1"]
        vy = segment["y2"] - segment["y1"]
        denominator = vx * vx + vy * vy
        projection = ((global_x - segment["x1"]) * vx +
                      (global_y - segment["y1"]) * vy) / max(denominator, 1)
        perpendicular = np.abs((global_x - segment["x1"]) * vy -
                               (global_y - segment["y1"]) * vx) / max(math.sqrt(denominator), 1)
        keep = (projection >= -0.15) & (projection <= 1.15) & \
               (perpendicular <= scale * 0.025)
        if keep.sum() < 3:
            continue
        xs, ys = global_x[keep], global_y[keep]
        weights = np.maximum(distance[ys, xs], 1) ** 2
        cx, cy, _ = weighted_pose(xs, ys, weights)
        cx, cy = project_to_pixels(cx, cy, xs, ys)
        angle = ((segment["angle"] + 90) % 180) - 90
        raw_candidates.append(dict(x=cx, y=cy, angle=angle,
                                   length=segment["length"], method="HOUGH",
                                   core_id=label_id))

core_view = cv2.cvtColor(core, cv2.COLOR_GRAY2BGR)
core_view[skeleton_all > 0] = (0, 255, 0)
save(6, "thick_core_and_skeleton", core_view)

# 7. 保留原需求中的 marker-controlled watershed，输出边界用于诊断
_, seed_labels = cv2.connectedComponents(core)
markers = seed_labels.astype(np.int32) + 1
sure_background = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
markers[sure_background == 0] = 1
markers[(sure_background > 0) & (core == 0)] = 0
watershed = cv2.watershed(image.copy(), markers)
boundary = image.copy()
boundary[watershed == -1] = (255, 0, 0)
save(7, "watershed_boundary", boundary)

# 合并同一中等厚度连通区内的相邻同向碎段，但保留真正交叉方向
medium = np.uint8(distance >= scale * 0.002) * 255
medium = cv2.morphologyEx(medium, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
_, medium_labels = cv2.connectedComponents(medium)
for candidate in raw_candidates:
    px, py = round(candidate["x"]), round(candidate["y"])
    candidate["group_id"] = int(medium_labels[py, px])

group_counts = {}
for candidate in raw_candidates:
    group_counts[candidate["group_id"]] = group_counts.get(candidate["group_id"], 0) + 1
simple_groups = set()
for group_id, group_count in group_counts.items():
    if group_id == 0 or group_count < 2:
        continue
    ys, xs = np.where(medium_labels == group_id)
    part = np.uint8(medium_labels[ys.min():ys.max() + 1,
                                  xs.min():xs.max() + 1] == group_id) * 255
    group_skeleton = prune_skeleton(skeletonize(part), max(3, round(scale * 0.025)))
    one = np.uint8(group_skeleton > 0)
    degree = cv2.filter2D(one, -1, np.ones((3, 3), np.uint8),
                          borderType=cv2.BORDER_CONSTANT) - one
    endpoints = cv2.connectedComponents(np.uint8((one > 0) & (degree == 1)))[0] - 1
    junctions = cv2.connectedComponents(cv2.dilate(
        np.uint8((one > 0) & (degree >= 3)) * 255, np.ones((3, 3), np.uint8)))[0] - 1
    if endpoints <= 3 and junctions <= 2:
        simple_groups.add(group_id)

parent = list(range(len(raw_candidates)))


def find(index):
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index


for i in range(len(raw_candidates)):
    for j in range(i + 1, len(raw_candidates)):
        first, second = raw_candidates[i], raw_candidates[j]
        if first["group_id"] == 0 or first["group_id"] != second["group_id"]:
            continue
        should_merge = first["group_id"] in simple_groups
        if not should_merge:
            dx, dy = second["x"] - first["x"], second["y"] - first["y"]
            separation = math.hypot(dx, dy)
            connecting_angle = math.degrees(math.atan2(dy, dx))
            alignment = max(angle_diff(first["angle"], connecting_angle),
                            angle_diff(second["angle"], connecting_angle))
            if first["core_id"] != second["core_id"]:
                direction_ok = angle_diff(first["angle"], second["angle"]) < 35
            else:
                direction_ok = angle_diff(first["angle"], second["angle"]) < 20 and alignment < 30
            should_merge = separation < scale * 0.14 and alignment < 48 and direction_ok
        if should_merge:
            root_i, root_j = find(i), find(j)
            if root_i != root_j:
                parent[root_j] = root_i

detections = []
for root_id in sorted({find(index) for index in range(len(raw_candidates))}):
    members = [raw_candidates[index] for index in range(len(raw_candidates))
               if find(index) == root_id]
    best = max(members, key=lambda item: item["length"])
    weights = np.array([item["length"] for item in members], dtype=float)
    cx = sum(item["x"] * weight for item, weight in zip(members, weights)) / weights.sum()
    cy = sum(item["y"] * weight for item, weight in zip(members, weights)) / weights.sum()
    group_id = best["group_id"]
    ys, xs = np.where((medium_labels == group_id) & (core > 0))
    if len(xs):
        cx, cy = project_to_pixels(cx, cy, xs, ys)
    radius = float(distance[round(cy), round(cx)])
    thickness_score = min(1.0, (2 * radius / scale) / 0.025)
    length_score = min(1.0, best["length"] / (scale * 0.15))
    confidence = 0.65 * thickness_score + 0.35 * length_score
    detections.append(dict(x=cx, y=cy, angle=best["angle"],
                           length=best["length"], thickness=2 * radius,
                           confidence=confidence, method=best["method"]))

detections = [item for item in detections if item["confidence"] >= args.min_confidence]
detections.sort(key=lambda item: (item["y"], item["x"]))

# 8. 红点为抓取点，紫线为姿态；低于 0.5 的候选用橙点提示复核
result = image.copy()
for index, item in enumerate(detections, 1):
    x, y = round(item["x"]), round(item["y"])
    angle = item["angle"]
    arm = max(15, round(item["length"] * 0.35))
    color = (0, 0, 255) if item["confidence"] >= 0.5 else (0, 140, 255)
    cv2.circle(result, (x, y), max(4, scale // 250), color, -1)
    cv2.line(result, (x, y),
             (round(x + arm * math.cos(math.radians(angle))),
              round(y + arm * math.sin(math.radians(angle)))),
             (255, 0, 255), max(1, scale // 500))
    cv2.putText(result, str(index), (x + 6, y - 6), 0, 0.42,
                color, 1, cv2.LINE_AA)
    item["id"] = index
save(8, "result", result)

with open(out / "poses.csv", "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow(["id", "center_x_px", "center_y_px", "angle_deg",
                     "confidence", "thickness_px", "length_px", "method"])
    for item in detections:
        writer.writerow([item["id"], f'{item["x"]:.2f}', f'{item["y"]:.2f}',
                         f'{item["angle"]:.2f}', f'{item["confidence"]:.3f}',
                         f'{item["thickness"]:.2f}', f'{item["length"]:.2f}',
                         item["method"]])
print(f"完成：识别 {len(detections)} 个粗壮主体候选，结果保存在 {out}")
