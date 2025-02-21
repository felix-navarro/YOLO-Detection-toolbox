import pandas as pd
import os
import fiftyone as fo
import fiftyone.utils.data as foud
from object_detection import fifty_one_utils as fou
from object_detection import classes_BIIGLE as cb
from mongoengine import connect


# Get annotated images path from Biigle csv annotations report
def get_annotated_images(report_file, image_dir):

    # Validation des entrées
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Le répertoire {image_dir} n'existe pas.")
    if not os.path.isfile(report_file):
        raise FileNotFoundError(f"Le fichier {report_file} n'existe pas.")

    # Import csv annotation report
    report = pd.read_csv(report_file)

    # Vérification de la colonne 'filename'
    if 'filename' not in report.columns:
        raise ValueError(f"Le fichier {report_file} ne contient pas la colonne 'filename'.")

    # Extraction des images annotées
    annotated_images = report['filename'].unique()

    # Création du chemin complet pour chaque image
    return [os.path.join(image_dir, image) for image in annotated_images]


# Convert a BIIGLE csv annotation format dataset to a yolov5 format dataset
def csv2yolo(image_dir, report_file, export_dir, name, label_col, level = "all", train_split = 1):

    # Import unique classes from BIIGLE csv annotation report file according to asked or lowest annotation level depth
    classes = cb.classes_BIIGLE(report_dir = report_file, label_col = label_col, level = level)

    # Connect to MongoDB
    connect(
        db="fiftyone",  # Specify the name of your database (default for FiftyOne is 'fiftyone')
        host="mongodb://localhost:27017",  # MongoDB URI (adjust if using remote or custom ports)
    )

    # Configure FiftyOne to use a specific MongoDB URI
    fo.config.database_uri = "mongodb://localhost:27017"

    # Create fiftyone dataset
    dataset = fo.Dataset(name = name)

    # Get annotated images from image dir
    annotated_images = get_annotated_images(report_file, image_dir)
    annotated_images_parser = foud.ImageSampleParser()

    # Add annotated images to dataset
    dataset.add_images(annotated_images, sample_parser = annotated_images_parser)

    # Import BIIGLE csv annotation in fiftyone according to asked or lowest annotation level depth
    annotations = fou.import_image_csv_report(image_dir = image_dir, report_file = report_file, level = level)

    # Add annotation to image dataset
    dataset.add_samples(annotations)

    # Export annotation and image dataset as yolov5 format
    fou.export_yoloV5_format(dataset = dataset,
                             export_dir = export_dir,
                             classes = classes,
                             train_split= train_split
                             )


if __name__ == '__main__':

    name = "pl01"
    image_dir = "D:/KANADEEP/01_Donnee/02_Image/PL01/Annotations/Annotated_Images"
    report_file = "U:/10_THESE/Felix Navarro/01_Donnee/01_Plongee/PL01/04_Annotations/91_csv_image_annotation_report/78-catami-classification-scheme.csv"
    # classes = ["Porifera","ECH_OPH_BR_Brittle/Snake stars","ECH_CRI_UN_Unstalked crinoid","ECH_CRI_Feather stars","SP_Sponges","CNI_Cnidaria","CNI_CO_Corals","CNI_TRUA_True Anemones","ECH_URC_Sea Urchins","ECH_URC_RE_Regular urchins","Translucent_ball","SP_MA_GM_Globular-massive","White_ball","CRU_PRA_Prawns/Shrimps/Mystids","CNI_TRUAOA_Other Anemones","CRU_Crustacea","MOL_Molluscs","ECH_STAR_Sea Stars","CNI_3DA_Arborescent","FIS_Fishes","SP_ER_ST_Stalked","SP_ER_Erect","SP_ER_1D_1D Erect (simple erect)","CNI_WPC_Complex_whip","UNIDENT_Unidentified","ECH_CRI_ST_Stalked crinoid","CNI_TUBA_Tube Anemones","SP_ER_2D_2D Erect","SP_ER_3D_3D Erect_branching","SP_CUP_TSB_Amphoras_sack-like_bladders","ECH_URC_IR_Irregular urchins","MOL_GAS_Gastropods","SP_CRU_CR_Creeping","SP_EN_ENB_Endolithic_bioeroding","Yellow_ball","SP_MA_CM_Composite-massive_meshes_dense-clusters","SP_MA_SM_Simple-massive","SP_ER_2DLA_Erect_laminar_flabellate","WORMS_POLT_Tube worms","SP_CRU_EN_Encrusting","CRU_CRA_Crabs","SP_CUP_CI_Incomplete_cups","ECH_OPH_BA_Basket stars","CNI_COLZOA_Zoanthids","CNI_HYD_UI_Unidentified_hydrocorals","ECH_SC_Sea Cucumbers","Tabular_sieve","SP_CUP_T_Tube-like forms (narrow cups)","SP_CUP_BA_Barrels (massive cups)","ECH_SC_BE_Benthic","SP_MA_Massive","SP_CUP_C_Cups","CNI_UP_DTn_Thin","CNI_CO_BLK_Black & Octocorals","R1 - Light","CNI_BLK_QU_Quill"]
    export_dir = "D:/KANADEEP/01_Donnee/02_Image/PL01/Annotations/Yolo_Training"
    label_col = "label_hierarchy"
    level = 1

    csv2yolo(image_dir = image_dir,
             report_file = report_file,
             export_dir = export_dir,
             name = name,
             label_col = label_col,
             level = level)