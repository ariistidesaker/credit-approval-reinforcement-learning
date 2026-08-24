import os
import shutil
import kagglehub
import pandas as pd

DATASET_IDENTIFIER = "zvikomborerocmufari/southern-african-banks-lgd-data-simulation"
TARGET_FILENAME = "synthetic_sadc_lgd_dataset.csv"

def download_and_save_data(target_dir: str = "data") -> str:
    """
    Telecharge le dataset depuis Kaggle via kagglehub et l'enregistre dans le dossier local cible.
    """
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, TARGET_FILENAME)
    
    if os.path.exists(target_path):
        print(f"[INFO] Dataset local deja present : {target_path}")
        df = pd.read_csv(target_path)
        print(f"[INFO] Dimensions : {df.shape[0]} lignes, {df.shape[1]} colonnes.")
        return target_path

    print(f"[INFO] Telechargement du dataset '{DATASET_IDENTIFIER}' via Kaggle...")
    download_dir = kagglehub.dataset_download(DATASET_IDENTIFIER)
    source_file = os.path.join(download_dir, TARGET_FILENAME)
    
    if not os.path.exists(source_file):
        csv_files = [f for f in os.listdir(download_dir) if f.endswith('.csv')]
        if csv_files:
            source_file = os.path.join(download_dir, csv_files[0])
        else:
            raise FileNotFoundError(f"Aucun fichier CSV trouve dans {download_dir}")

    shutil.copy(source_file, target_path)
    print(f"[SUCCESS] Dataset copie avec succes dans : {target_path}")
    
    df = pd.read_csv(target_path)
    print(f"[INFO] Apercu du dataset : {df.shape[0]} lignes, {df.shape[1]} colonnes.")
    return target_path

if __name__ == "__main__":
    download_and_save_data()
