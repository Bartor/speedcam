"""Fine-tune yolo11x (COCO weights) on VisDrone2019-DET for drone-view vehicle detection.

    python tools/train_visdrone.py <work_dir> [hours]

Downloads VisDrone (~2 GB) into <work_dir>/datasets on first run, trains at 1280 px with batch 2
(fits a 12 GB GPU; Ultralytics' AutoBatch misjudged this and ran out of memory), and caps wall
time at `hours` (default 6.5): the learning-rate schedule is fitted to the budget. Copy
<work_dir>/runs/yolo11x-visdrone-1280/weights/best.pt to ~/speedcam-models/yolo11x-visdrone.pt
to use it as the "visdrone" detector.
"""
import sys
from pathlib import Path

import ultralytics
from ultralytics import YOLO

if __name__ == "__main__":
    work = Path(sys.argv[1]).resolve()
    hours = float(sys.argv[2]) if len(sys.argv) > 2 else 6.5
    work.mkdir(parents=True, exist_ok=True)
    src = Path(ultralytics.__file__).parent / "cfg" / "datasets" / "VisDrone.yaml"
    data = work / "visdrone.yaml"
    data.write_text(src.read_text(encoding="utf-8").replace(
        "path: VisDrone", f"path: {(work / 'datasets' / 'VisDrone').as_posix()}", 1), encoding="utf-8")
    YOLO("yolo11x.pt").train(
        data=str(data), imgsz=1280, epochs=40, time=hours, patience=8, batch=2, workers=8, cos_lr=True,
        project=str(work / "runs"), name="yolo11x-visdrone-1280", exist_ok=True, plots=True,
    )
