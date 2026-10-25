# 深色长条苗分水岭检测工具 V1.1

兼容性更新（2026-07-31）：兼容 OpenCV 3/4 的 `findContours` 返回格式，
新增 `--version` 便于确认运行的是正确文件。V1 不包含 `HoughLinesP`；如果报错
仍指向 `lines[:, 0]`，说明实际运行的是旧版 V2 文件或混用了源码，请完整替换目录。

这是可直接在 PyCharm 或命令行运行的 OpenCV 程序。程序按 8 个阶段处理工业相机照片，输出候选苗的像素质心、夹爪姿态角、标注图和 CSV。核心算法为欧氏距离变换与 marker-controlled watershed，不依赖示例红点图。

V1 是结构清晰、依赖少的基础版本。对于分散或轻微接触的苗效果较好；严重交叉、密集缠绕时可能过分割或将细根片段当成独立目标。精度优先的现场应用建议同时评估 V2。

## 环境与安装

建议使用 Python 3.10～3.12。在本目录打开 PyCharm Terminal 或 PowerShell：

```powershell
pip install -r requirements.txt
```

## 单张图片

```powershell
python seedling_watershed.py sample_input.png -o sample_result
```

若需要逐步弹窗检查中间结果：

```powershell
python seedling_watershed.py sample_input.png -o sample_result --show
```

`--show` 模式下按任意键进入下一阶段。批量处理时不要添加该参数。

处理自己的照片：

```powershell
python seedling_watershed.py "D:\images\test_001.jpg" -o "D:\images\test_001_result"
```

## 批量图片

```powershell
python batch_run.py "D:\images" "D:\batch_results"
```

支持 PNG、JPG、JPEG、BMP、TIF 和 TIFF。每张原图会生成一个同名结果子目录。

## 输出内容

- `01_original.png`：原图。
- `02_enhanced.png`：灰度与局部对比度增强结果。
- `03_dark_contrast.png`：深色长条响应。
- `04_binary_mask.png`：二值分割掩膜。
- `05_distance_transform.png`：欧氏距离变换热力图。
- `06_watershed_seeds.png`：分水岭种子。
- `07_watershed_boundary.png`：分水岭边界。
- `08_result.png`：红点表示质心，紫线表示夹爪方向。
- `poses.csv`：候选编号、中心坐标、姿态角和区域面积。

坐标原点位于图像左上角，x 向右、y 向下。角度相对图像的 +x 方向，顺时针为正，范围为 `[-90°, 90°)`；平行夹爪旋转 180° 后等价。

## 现场调参位置

参数集中在 `seedling_watershed.py`：

- 候选过多：降低第 6 阶段的 `0.31`，或提高第 8 阶段的长宽比 `1.6`、`minimum_area`、`minimum_length`。
- 候选过少：提高第 6 阶段的 `0.31` 以产生更多种子；若细苗被过滤，可适当降低面积或长度阈值。
- 背景斑点较多：优先改善背光和曝光，也可增大第 3 阶段的背景核比例 `0.075`。
- 托盘边缘误检：固定相机后先裁出机械手可抓取 ROI，再送入程序处理。

## 接入机械手前需要的输入

当前 CSV 只包含视觉像素坐标和图像角度。接入机械手还需要：

1. 代表性原始照片及人工真值，用于统计精确率、召回率和漏检率。
2. 固定的相机曝光、增益、光源和振动盘停止状态。
3. 抓取平面的像素到毫米标定或平面单应矩阵。
4. 相机坐标系到机器人基坐标系的外参或手眼标定结果。
5. 夹爪允许角度、安全边界及目标实际尺寸范围。
6. 镜头畸变明显时所需的相机内参与畸变系数。

在完成坐标标定前，不要把 `poses.csv` 的像素坐标直接当作机器人毫米坐标使用。
