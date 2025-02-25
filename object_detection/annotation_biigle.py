import pandas as pd
import os

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


def extract_level(x, level="all"):
    """
    Extrait un niveau hiérarchique spécifique d'une chaîne donnée.

    Args:
        x (str): Une chaîne représentant une hiérarchie sous forme de niveaux séparés par ' > '.
        level (int ou str): Niveau hiérarchique à extraire ou "all" pour retourner la chaîne entière.

    Returns:
        str: Le niveau hiérarchique extrait ou la chaîne entière si `level="all"`.

    Raises:
        ValueError: Si les arguments sont invalides ou si le niveau est hors limites.
    """
    # Vérification des types et des arguments
    if level != "all" and (not isinstance(level, int) or isinstance(level, bool) or level < 0):
        raise ValueError("Le paramètre 'level' doit être égal à 'all' ou un entier positif.")
    if not isinstance(x, str):
        raise ValueError("La valeur de 'x' doit être une chaîne.")

    # Vérifiez si x est une chaîne vide
    if not x.strip():
        raise ValueError("La chaîne fournie est vide.")

    # Retourner x directement si level == "all"
    if level == "all":
        return x

    # Diviser x en niveaux
    parts = x.split(" > ")
    try:
        # Vérifier si le niveau demandé dépasse les parties
        return parts[-1] if level > len(parts) - 1 else parts[level]
    except IndexError as e:
        raise ValueError(
            f"Le niveau spécifié ({level}) dépasse la profondeur maximale des éléments."
        ) from e




def classes_BIIGLE(report_dir, label_col="label_name", level="all"):
    """
    Extrait les noms uniques des classes depuis un fichier de rapport CSV BIIGLE.
    Args:
        report_dir (str): Chemin vers le fichier CSV de rapport.
        label_col (str): Nom de la colonne contenant les noms des classes.
        level (int ou "all"): Si "all", retourne toutes les classes uniques. Sinon, un entier
                              spécifiant le niveau hiérarchique à extraire de la colonne 'label_hierarchy'.
    Returns:
        numpy.ndarray: Liste des noms uniques de classes.
    Raises:
        FileNotFoundError: Si le fichier demandé n'existe pas ou n'est pas un fichier valide.
        ValueError: Si la colonne spécifiée est manquante ou vide.
    """
    # Vérification du chemin du fichier
    if not os.path.isfile(report_dir):
        raise FileNotFoundError(f"Le fichier {report_dir} n'existe pas ou n'est pas un fichier valide.")

    # Lecture du fichier CSV
    try:
        csv_report = pd.read_csv(report_dir)
    except pd.errors.ParserError as e:
        raise ValueError(f"Erreur de parsing du fichier CSV {report_dir} : {e}")
    except UnicodeDecodeError as e:
        raise ValueError(f"Erreur d'encodage lors de la lecture du fichier CSV {report_dir} : {e}")

    # Vérification de la colonne `label_col`
    if label_col not in csv_report.columns:
        raise ValueError(f"La colonne '{label_col}' est absente du fichier CSV {report_dir}.")
    if csv_report[label_col].isnull().all():
        raise ValueError(f"La colonne '{label_col}' ne contient aucune donnée valide.")


    # Vérification de la colonne `label_hierarchy`
    if "label_hierarchy" not in csv_report.columns:
        raise ValueError("La colonne 'label_hierarchy' est absente du fichier CSV.")
    if csv_report["label_hierarchy"].isnull().all():
        raise ValueError("La colonne 'label_hierarchy' ne contient aucune donnée valide.")

    # Extraction du niveau demandé
    try:
        extracted_labels = csv_report["label_hierarchy"].dropna().apply(
            lambda x: extract_level(x, level)
        )
    except IndexError:
        raise ValueError(f"Le niveau spécifié ({level}) dépasse la profondeur maximale des hiérarchies.")

    # Vérification et retour
    if extracted_labels.isnull().all():
        raise ValueError(f"Les données extraites au niveau {level} sont vides ou invalides.")
    return extracted_labels.dropna().unique()


if __name__ == '__main__':
    report_dir = r"U:\10_THESE\Felix Navarro\01_Donnee\01_Plongee\PL01\04_Annotations\91_csv_image_annotation_report\78-catami-classification-scheme.csv"
    classes = classes_BIIGLE(report_dir, label_col="label_hierarchy", level=2)
    print(classes)