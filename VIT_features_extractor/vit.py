import argparse
import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def build_model(device: torch.device) -> torch.nn.Module:
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    model = timm.create_model("vit_small_patch16_224", pretrained=True)
    model.head = torch.nn.Identity()
    model = model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    return model


def get_image_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )


def collect_image_files(image_dir: Path) -> list[Path]:
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(
        [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_ext]
    )


def extract_features(
    model: torch.nn.Module,
    image_paths: list[Path],
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    transform = get_image_transform()
    image_features: list[np.ndarray] = []
    image_ids: list[str] = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
        batch_files = image_paths[i : i + batch_size]
        batch_imgs = []
        batch_ids = []

        for img_path in batch_files:
            try:
                image = Image.open(img_path).convert("RGB")
                image = transform(image)
                batch_imgs.append(image)
                batch_ids.append(img_path.name)
            except Exception as e:
                print(f"Error reading {img_path.name}: {e}")

        if not batch_imgs:
            continue

        batch_tensor = torch.stack(batch_imgs).to(device)

        with torch.no_grad():
            feats = model(batch_tensor)

        image_features.extend(feats.cpu().numpy())
        image_ids.extend(batch_ids)

    return np.array(image_features), image_ids


def save_outputs(features: np.ndarray, image_ids: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = output_dir / "vit_features_xai.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump({"image_ids": image_ids, "features": features}, f)

    csv_path = output_dir / "image_features.csv"
    df = pd.DataFrame(features)
    df["image_id"] = image_ids
    df.to_csv(csv_path, index=False)

    print(f"Saved pickle: {pkl_path}")
    print(f"Saved csv: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ViT feature extractor for chest X-ray images")
    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/images",
        help="Path to folder containing image files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Path to folder where feature files will be written",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for feature extraction",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)

    if not image_dir.exists() or not image_dir.is_dir():
        print("Image folder not found.")
        print(f"Expected at: {image_dir.resolve()}")
        print("Create the folder or pass --image-dir with your dataset path.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(device)
    print("ViT loaded successfully.")

    image_files = collect_image_files(image_dir)
    if not image_files:
        print("No supported images found in the folder.")
        return

    print(f"Total images: {len(image_files)}")
    features, image_ids = extract_features(
        model=model,
        image_paths=image_files,
        batch_size=args.batch_size,
        device=device,
    )

    if features.size == 0:
        print("No features extracted. Check image files for read issues.")
        return

    print(f"Feature matrix shape: {features.shape}")
    save_outputs(features=features, image_ids=image_ids, output_dir=output_dir)


if __name__ == "__main__":
    main()