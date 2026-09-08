import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from initmat.main import main


if __name__ == "__main__":
    input_mat = None  # e.g. PROJECT_ROOT / "data" / "initial_state.mat"

    sys.exit(main(
        domain_size=250,
        shape_file=PROJECT_ROOT / "data" / "particleShapes250rotations.mat",
        pom_shape_file=PROJECT_ROOT / "data" / "POMshapes250.mat",
        psd_file=PROJECT_ROOT / "data" / "loam_bayreuth.csv",
        output_dir=PROJECT_ROOT / "output",
        visualization_dir=PROJECT_ROOT / "visualizations",
        input_mat=input_mat,
        boundary_mode="periodic",
        target_porosity=0.45,
    ))