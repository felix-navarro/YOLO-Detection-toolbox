# YOLO Classification

Classification script using YOLO (You Only Look Once) model from ultralytics
This script is based upon Marin Marcillat's reprojection toolbox and ultralytics tutorials for YOLO models

## Requirements

```bash
mamba create -y --prefix ./envs -c conda-forge -c pytorch -c nvidia python=3.11 pip pyvista ultralytics pytorch torchvision torchaudio pytorch-cuda=11.8 tqdm pandas geopandas sqlalchemy scipy jupyterlab sahi ipyfilechooser shapely pillow wandb treelib pyqt python-dotenv
```

```bash
conda activate ./envs
pip install fiftyone
```


Ultralytics fonctionne pas 