import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
VISUALIZATION_DIR = PROJECT_ROOT / "visualizations"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

for folder in (DATA_DIR, OUTPUT_DIR, VISUALIZATION_DIR, SCRIPTS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from initmat.main import main


if __name__ == "__main__":
    input_mat = None  # e.g. DATA_DIR / "initial_state.mat"

    sys.exit(main(
        domain_size=250,
        shape_file=DATA_DIR / "particleShapes250rotations.mat",
        pom_shape_file=DATA_DIR / "POMshapes250.mat",
        psd_file=DATA_DIR / "loam_bayreuth.csv",
        output_dir=OUTPUT_DIR,
        visualization_dir=VISUALIZATION_DIR,
        input_mat=input_mat,
        boundary_mode="periodic",
        target_porosity=0.45,
        init_cb_concentration=0.3168 / 5,
        c_n_b=10,
        min_cb_concentration=0.0132 / 8,
        max_cb_cells=216,
    ))