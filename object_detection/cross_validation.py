from pathlib import Path
import yaml
import pandas as pd
from object_detection.fifty_one_utils import get_classes, make_yolo_row
from sklearn.model_selection import KFold
import shutil
import datetime
from tqdm import tqdm
#
from mongoengine import connect
from object_detection import annotation_biigle as ab
from fiftyone import ViewField as F
import fiftyone as fo
from object_detection import fifty_one_utils as fou
import fiftyone.utils.data as foud

def k_fold_cross_validation(dataset, export_dir, ksplit = 5):

    classes = get_classes(dataset)
    classes_dict = {c: i for i, c in enumerate(classes)}
    cls_idx = sorted(classes_dict.values())

    index = [sample.id for sample in dataset]
    labels_df = pd.DataFrame([], columns=cls_idx, index=index)
    labels_df = labels_df.fillna(0.0)

    for sample in dataset:
        for detection in sample.detections.detections:
            labels_df.loc[sample.id, classes_dict[detection.label]] += 1

    kf = KFold(n_splits=ksplit, shuffle=True, random_state=20)  # setting random_state for repeatable results

    kfolds = list(kf.split(labels_df))

    folds = [f"split_{n}" for n in range(1, ksplit + 1)]
    folds_df = pd.DataFrame(index=index, columns=folds)

    for i, (train, val) in enumerate(kfolds, start=1):
        folds_df[f"split_{i}"].loc[labels_df.iloc[train].index] = "train"
        folds_df[f"split_{i}"].loc[labels_df.iloc[val].index] = "val"

    fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)

    for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
        train_totals = labels_df.iloc[train_indices].sum()
        val_totals = labels_df.iloc[val_indices].sum()

        # To avoid division by zero, we add a small value (1E-7) to the denominator
        ratio = val_totals / (train_totals + 1e-7)
        fold_lbl_distrb.loc[f"split_{n}"] = ratio

    # Loop through supported extensions and gather image files
    images = [sample.filepath for sample in dataset]

    # Create the necessary directories and dataset YAML files (unchanged)
    save_path = Path(Path(export_dir) / f"{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val")
    save_path.mkdir(parents=True, exist_ok=True)
    ds_yamls = []

    for split in folds_df.columns:
        # Create directories
        split_dir = save_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (split_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

        # Create dataset YAML files
        dataset_yaml = split_dir / f"{split}_dataset.yaml"
        ds_yamls.append(dataset_yaml)

        with open(dataset_yaml, "w") as ds_y:
            yaml.safe_dump(
                {
                    "path": split_dir.as_posix(),
                    "train": "train",
                    "val": "val",
                    "names": list(classes),
                },
                ds_y,
            )

    print("Copying images and exporting labels to new directories (YoloV5)")
    for sample in tqdm(dataset):
        for split, k_split in folds_df.loc[sample.id].items():
            # Destination directory
            img_to_path = save_path / split / k_split / "images"
            lbl_to_path = save_path / split / k_split / "labels"

            # Copy image and label files to new directory (SamefileError if file already exists)
            shutil.copy(sample.filepath, img_to_path / sample.filename)

            label_path = lbl_to_path / f"{Path(sample.filename).stem}.txt"
            with open(label_path, "w") as f:
                for detection in sample.detections.detections:
                    f.write(make_yolo_row(detection, classes_dict[detection.label]) + "\n")

    return ds_yamls


if __name__ == '__main__':
    # Get import image directory path
    image_dir = "D:/KANADEEP/01_Donnee/02_Image/PL01/Annotations/Annotated_Images"

    # Get import BIIGLE csv annotation report file path
    report_file = "U:/10_THESE/Felix Navarro/01_Donnee/01_Plongee/PL01/04_Annotations/91_csv_image_annotation_report/78-catami-classification-scheme.csv"

    # Get export directory for Yolov5 format dataset
    export_dir = "D:/KANADEEP/01_Donnee/02_Image/PL01/Annotations/Kfold_Training"

    # Get name for project
    name = "pl01"

    # Get column to extract label from
    label_col = "label_hierarchy"

    # Get label depth to extract label
    level = 1

    # Import unique classes from BIIGLE csv annotation report file according to asked or lowest annotation level depth
    classes_1 = ab.classes_BIIGLE(report_dir=report_file, label_col=label_col, level=level)

    # %%
    # Connect to MongoDB
    connect(
        db="fiftyone",  # Specify the name of your database (default for FiftyOne is 'fiftyone')
        host="mongodb://localhost:27017",  # MongoDB URI (adjust if using remote or custom ports)
    )

    # Configure FiftyOne to use a specific MongoDB URI
    fo.config.database_uri = "mongodb://localhost:27017"

    # Create fiftyone dataset
    dataset = fo.Dataset()

    # Get annotated images from image dir
    annotated_images = ab.get_annotated_images(report_file, image_dir)
    annotated_images_parser = foud.ImageSampleParser()

    # Add annotated images to dataset
    dataset.add_images(annotated_images, sample_parser=annotated_images_parser)

    # Import BIIGLE csv annotation in fiftyone according to asked or lowest annotation level depth
    annotations = fou.import_image_csv_report(image_dir=image_dir, report_file=report_file, level=level)

    # Add annotation to image dataset
    dataset.add_samples(annotations)

    dataset.default_classes = fou.get_classes(dataset)
    dataset = dataset.match(F("detections.detections").length() != 0)