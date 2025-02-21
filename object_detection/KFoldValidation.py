import pandas as pd
import yaml
from pathlib import Path
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit, KFold
from object_detection import classes_BIIGLE as cB
import shutil

# Import dataset directory path
dataset_path = Path("D:/KANADEEP/01_Donnee/02_Image/PL01/Annotations/Yolo_Training")  # replace with 'path/to/dataset' for your custom data
# Extract all data in labels
labels = sorted(dataset_path.rglob("*labels/train/*.txt"))
# Import BIIGLE csv annotation report file
report_file = "U:/10_THESE/Felix Navarro/01_Donnee/01_Plongee/PL01/04_Annotations/91_csv_image_annotation_report/78-catami-classification-scheme.csv"
report_csv = pd.read_csv(report_file)[['filename','label_name','label_hierarchy']]
level = 1

# Extract classes from label name
if level != "all":
    report_csv["label_name"] = report_csv["label_hierarchy"].dropna().apply(
        lambda x: cB.extract_level(x, level = level)
    )
# Count classes apparitions
class_counts = report_csv['label_name'].value_counts()
# Extract classes with at least two observations
valid_classes = class_counts[class_counts >= 2].index
# Filter classes
report_csv_filtered = report_csv[report_csv['label_name'].isin(valid_classes)]
# Print excluded classes
if len(report_csv) != len(report_csv_filtered):
    print("Les classes suivantes ont été exclues car elles contenaient moins de 2 exemples :")
    print(set(report_csv['label_name']) - set(report_csv_filtered['label_name']))


# Import yaml dataset path
yaml_file = "D:/KANADEEP/01_Donnee/02_Image/PL01/Annotations/Yolo_Training/dataset.yaml"

# Open yaml file
with open(yaml_file, "r", encoding="utf8") as y:
    # Extract classes from yaml file
    classes = yaml.safe_load(y)["names"]
# Get classes index
cls_idx = sorted(classes.keys())

# Extract filenames
index = [label.stem for label in labels]  # uses base filename as ID (no extension)
# Create contengency table
labels_df = pd.DataFrame([], columns=cls_idx, index=index)

# Count number of annotations by classes for each photo
for label in labels:
    # Initialised counter
    lbl_counter = Counter()

    # Open label file and extract annotations
    with open(label, "r") as lf:
        lines = lf.readlines()
    #
    for line in lines:
        # classes for YOLO label uses integer at first position of each line
        lbl_counter[int(line.split(" ")[0])] += 1

    labels_df.loc[label.stem] = lbl_counter

# Replace `nan` values with `0.0`
labels_df = labels_df.fillna(0.0)

# Set numbers of Kfold
ksplit = 5
# Initialise StratifiedKFold for k splits (random_stat set to 1 for repeatability)
skf = StratifiedShuffleSplit(n_splits = ksplit, train_size=0.8, random_state = 1)
kf = KFold(n_splits=ksplit, shuffle=True, random_state=20)

kfolds = list(kf.split(labels_df))

X = report_csv_filtered['filename']
y = report_csv_filtered.pop('label_name')

skfolds = skf.split(X, y)

# folds = [f"split_{n}" for n in range(1, ksplit + 1)]
# folds_df = pd.DataFrame(index=index, columns=folds)

for i, (train, val) in enumerate(skfolds, start=1):
    print("n_split:", i, "TRAIN:", train, "TEST:", val)

folds_df[f"split_{i}"].loc[X.iloc[train].index] = "train"
folds_df[f"split_{i}"].loc[X.iloc[val].index] = "val"

X = report_csv_filtered
y = report_csv_filtered.pop('label_name')


for n, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
    print("n_split:",n,"TRAIN:", train_index, "TEST:", test_index)


fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)

for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
    train_totals = labels_df.iloc[train_indices].sum()
    val_totals = labels_df.iloc[val_indices].sum()

    # To avoid division by zero, we add a small value (1E-7) to the denominator
    ratio = val_totals / (train_totals + 1e-7)
    fold_lbl_distrb.loc[f"split_{n}"] = ratio


import datetime

supported_extensions = [".jpg", ".jpeg", ".png"]

# Initialize an empty list to store image file paths
images = []

# Loop through supported extensions and gather image files
for ext in supported_extensions:
    images.extend(sorted((dataset_path / "images").rglob(f"*{ext}")))

# Create the necessary directories and dataset YAML files (unchanged)
save_path = Path(dataset_path / f"{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val")
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
                "names": classes,
            },
            ds_y,
        )

for image, label in zip(images, labels):
    for split, k_split in folds_df.loc[image.stem].items():
        # Destination directory
        img_to_path = save_path / split / k_split / "images"
        lbl_to_path = save_path / split / k_split / "labels"

        # Copy image and label files to new directory (SamefileError if file already exists)
        shutil.copy(image, img_to_path / image.name)
        shutil.copy(label, lbl_to_path / label.name)


folds_df.to_csv(save_path / "kfold_datasplit.csv")
fold_lbl_distrb.to_csv(save_path / "kfold_label_distribution.csv")