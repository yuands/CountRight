"""
CountRight 生成结果正确率评价脚本
基于 YOLOv9 检测每张图中目标物体的数量，与期望数量比较

用法:
    python evaluation_script.py \
        --images_dir outputs \
        --output_dir eval_output \
        --yolo_weights /home/cx_wchn/yuands/Count/count_right/dataset/yolov9e.pt

支持的图片文件名格式:
    {class}_num={N}_seed={S}.png
    例如: airplane_num=3_seed=152767.png, baseball glove_num=5_seed=740685.png

脚本会自动跳过 _vanilla.png 和 _masks.png 等非最终生成图。
"""
import os
import re
import sys
from typing import List, Tuple, Optional

import PIL.Image
import PIL
import pandas as pd
import supervision as sv
from ultralytics import YOLO
import argparse


# 类名归一化映射
# 将 prompt 中提取的物体名（可能为复数、合成词）映射为 YOLO/COCO 的标准类别名
# YOLO COCO 80 类: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
CLASS_NAME_MAPPING = {
    # 复数 → 单数
    "airplanes": "airplane",
    "backpacks": "backpack",
    "bears": "bear",
    "birds": "bird",
    "boats": "boat",
    "bottles": "bottle",
    "bowls": "bowl",
    "buses": "bus",
    "cars": "car",
    "cats": "cat",
    "chairs": "chair",
    "cows": "cow",
    "dogs": "dog",
    "donuts": "donut",
    "elephants": "elephant",
    "giraffes": "giraffe",
    "horses": "horse",
    "kittens": "cat",         # kittens → cat (YOLO 只有 cat)
    "cats_kitten": "cat",
    "motorcycles": "motorcycle",
    "oranges": "orange",
    "sheeps": "sheep",
    "trucks": "truck",
    "zebras": "zebra",
    # 不规则复数
    "apples": "apple",
    "knives": "knife",
    "sheep": "sheep",
    # 合成词: prompt 中可能是 "baseball gloves", 映射为 COCO 类 "baseball glove"
    "baseball gloves": "baseball glove",
    "tennis rackets": "tennis racket",
    "parking meters": "parking meter",
    "stop signs": "stop sign",
    "sports balls": "sports ball",
    "potted plants": "potted plant",
    "laptop computers": "laptop",
    "cell phones": "cell phone",
    "hot dogs": "hot dog",
    "suit cases": "suitcase",
}


def normalize_class_name(name: str) -> str:
    """
    将类名归一化为 YOLO/COCO 的标准类别名（小写 + 单数）。
    先查显式映射表；查不到则用英文复数规则兜底。
    """
    lower = name.strip().lower()
    if lower in CLASS_NAME_MAPPING:
        return CLASS_NAME_MAPPING[lower]
    # 兜底: 英文简单复数规则
    if lower.endswith("ves"):
        return lower[:-3] + "f"        # knives → knife
    if lower.endswith("ies"):
        return lower[:-3] + "y"        # berries → berry
    if lower.endswith("ses") or lower.endswith("xes") or lower.endswith("zes"):
        return lower[:-2]              # buses → bu(s)
    if lower.endswith("s") and not lower.endswith("ss"):
        return lower[:-1]              # cats → cat
    return lower


class YoloEvaluator:
    def __init__(self, output_dir: str, yolo_weights: str):
        self.model = YOLO(yolo_weights)
        self.bounding_box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.output_dir = output_dir

    def evaluate_example(self, image_path: str, class_name: str, expected_count: int):
        pil_image = PIL.Image.open(image_path)

        # 运行 YOLO 检测
        result = self.model(pil_image)[0]
        detections = sv.Detections.from_ultralytics(result)

        # 归一化后的目标类名
        target_class = normalize_class_name(class_name)

        # 空检测: 直接返回 0
        if len(detections) == 0:
            return {
                "expected_count": expected_count,
                "predicted_count": 0,
                "is_success": expected_count == 0,
                "class_name_original": class_name,
                "class_name_normalized": target_class,
                "diff": -expected_count,
                "all_detections": [],
                "annotated_frame": pil_image,
            }

        # 统计检测到的目标类别 bbox 数（与归一化后的类名精确匹配）
        detections_df = pd.DataFrame(
            [[x for x in detections.xyxy], [*detections.data["class_name"]]]
        ).T
        detections_df = detections_df.rename({0: "box", 1: "class_name"}, axis=1)
        detections_df["is_target"] = detections_df["class_name"].apply(
            lambda detected_class_name: detected_class_name == target_class
        )
        predicted_count = int(detections_df["is_target"].sum())

        # 记录所有检测到的类名（便于排查漏检/误检）
        all_detections = detections_df["class_name"].tolist()

        # 带检测框的可视化
        annotated_frame = self._create_annotated_image(pil_image, detections)

        return {
            "expected_count": expected_count,
            "predicted_count": predicted_count,
            "is_success": predicted_count == expected_count,
            "class_name_original": class_name,
            "class_name_normalized": target_class,
            "diff": predicted_count - expected_count,
            "all_detections": "|".join(all_detections),
            "annotated_frame": annotated_frame,
        }

    def _create_annotated_image(self, pil_image: PIL.Image, detections):
        annotated_frame = pil_image.copy()
        annotated_frame = self.bounding_box_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
        )
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
        )
        return annotated_frame


def extract_image_paths(images_dir: str) -> List[str]:
    """提取目录下所有 PNG 路径，跳过 _vanilla / _masks / _html 等非最终图"""
    skip_suffixes = ("_vanilla.png", "_masks.png")
    return [
        os.path.join(images_dir, img_name)
        for img_name in os.listdir(images_dir)
        if img_name.endswith(".png") and not img_name.endswith(skip_suffixes)
    ]


def analyze_image_name(image_path: str) -> Optional[Tuple[int, str]]:
    """
    解析文件名 {class}_num={N}_seed={S}.png
    返回 (expected_count, class_name); 不匹配返回 None
    """
    basename = os.path.basename(image_path)
    name_no_ext = basename.rsplit(".", 1)[0]  # 去掉 .png

    match = re.match(r"^(.+?)_num=(\d+)_seed=\d+$", name_no_ext)
    if not match:
        return None

    class_name = match.group(1)
    expected_count = int(match.group(2))
    return expected_count, class_name


def save_results(results: List[dict], output_dir: str):
    pd.DataFrame(results).to_csv(f"{output_dir}/results.csv", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="评价 CountRight 生成图像中物体数量的正确率（基于 YOLOv9）"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="/home/cx_wchn/yuands/Count/make-it-count/outputs",
        help="包含 CountRight 生成图像的目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/cx_wchn/yuands/Count/make-it-count/eval_output",
        help="评价结果输出目录",
    )
    parser.add_argument(
        "--yolo_weights",
        type=str,
        default="/home/cx_wchn/yuands/Count/count_right/dataset/yolov9e.pt",
        help="YOLOv9 权重路径",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 检查权重文件
    if not os.path.exists(args.yolo_weights):
        print(f"[ERROR] YOLO 权重文件不存在: {args.yolo_weights}")
        sys.exit(1)

    images_paths = extract_image_paths(args.images_dir)
    total_images = len(images_paths)
    print(f"[INFO] 找到 {total_images} 张待评价图像")

    evaluator = YoloEvaluator(args.output_dir, args.yolo_weights)
    results = []
    skipped = 0

    for i, image_path in enumerate(images_paths, start=1):
        filename = os.path.basename(image_path)
        parsed = analyze_image_name(image_path)
        if parsed is None:
            print(f"[{i}/{total_images}] 跳过（无法解析）: {filename}")
            skipped += 1
            continue

        expected_count, class_name = parsed
        result = evaluator.evaluate_example(image_path, class_name, expected_count)

        # 保存带检测框的可视化
        result.pop("annotated_frame").save(
            os.path.join(args.output_dir, filename)
        )

        result["image_path"] = image_path
        result["filename"] = filename
        results.append(result)

        # 逐张进度
        status = "✓" if result["is_success"] else "✗"
        print(
            f"[{i}/{total_images}] {status} {filename}: "
            f"检测到 {result['predicted_count']} 个 "
            f"{result['class_name_normalized']} "
            f"(期望: {expected_count})"
        )

    if not results:
        print("[ERROR] 没有成功评价任何图像")
        sys.exit(1)

    # ===== 统计汇总 =====
    total = len(results)
    success = sum(1 for r in results if r["is_success"])
    accuracy = success / total

    print("\n" + "=" * 60)
    print(f"总评价图像数: {total}   (跳过格式不符: {skipped})")
    print(f"数量正确数:   {success}/{total}")
    print(f"总体正确率:   {accuracy:.2%}")
    print("=" * 60)

    # Per-class 统计
    df = pd.DataFrame(results)
    per_class = (
        df.groupby("class_name_original")
        .agg(
            total=("is_success", "count"),
            success=("is_success", "sum"),
        )
        .assign(accuracy=lambda x: x["success"] / x["total"])
        .sort_index()
    )
    print("\nPer-class 准确率:")
    print(per_class.to_string())

    save_results(results, args.output_dir)
    print(f"\n[INFO] 详细结果已保存到: {args.output_dir}/results.csv")
    print(f"[INFO] 带检测框的可视化已保存到: {args.output_dir}/")


if __name__ == "__main__":
    main()
