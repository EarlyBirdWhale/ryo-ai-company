"""
stock_judge.py - 在庫状態判定モジュール
"""
from pathlib import Path
from ultralytics import YOLO

from detector import detect_objects


def judge_stock(detections: list[dict]) -> dict:
    """
    detect_objects()の結果を受け取り在庫状態を判定する。

    Args:
        detections: detect_objects()が返す検出結果リスト

    Returns:
        {
            "status": "FULL" | "LOW" | "EMPTY",
            "bottle_count": int,
            "needs_alert": bool,
        }
    """
    bottle_count = sum(1 for d in detections if d["name"] == "bottle")

    if bottle_count >= 10:
        status = "FULL"
        needs_alert = False
    elif bottle_count >= 1:
        status = "LOW"
        needs_alert = True
    else:
        status = "EMPTY"
        needs_alert = True

    return {
        "status": status,
        "bottle_count": bottle_count,
        "needs_alert": needs_alert,
    }


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    image_dir = base_dir / "data" / "sample_images"
    model_path = base_dir / "yolov8n.pt"

    model = YOLO(str(model_path))
    model.to("cpu")

    image_paths = sorted(image_dir.glob("*.png"))
    print(f"モデル: {model_path.name}")
    print(f"画像数: {len(image_paths)} 枚")
    print("=" * 60)

    for image_path in image_paths:
        detections = detect_objects(str(image_path), model)
        result = judge_stock(detections)

        print(f"\n[画像] {image_path.name}")
        print(f"  bottle数    : {result['bottle_count']} 本")
        print(f"  在庫ステータス: {result['status']}")
        print(f"  通知要否    : {'要' if result['needs_alert'] else '不要'}")

    print("\n" + "=" * 60)
    print("判定完了")
