"""Render YOLO annotations without a GUI."""

from pathlib import Path

from PIL import Image, ImageDraw

CLASS_NAMES = ["open", "short", "mousebite", "spur", "copper", "pin-hole"]


def render_annotation(image_path: Path, label_path: Path, output_path: Path) -> None:
    """Draw one YOLO label file over its image."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        class_id, xc, yc, bw, bh = raw.split()
        cid = int(class_id)
        xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
        box = ((xc - bw / 2) * width, (yc - bh / 2) * height, (xc + bw / 2) * width, (yc + bh / 2) * height)
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), CLASS_NAMES[cid], fill="red")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

