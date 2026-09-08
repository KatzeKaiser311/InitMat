import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from .initial_state_editor import InitialStateEditor


def main(domain_size, shape_file, pom_shape_file, psd_file, output_dir, visualization_dir, input_mat=None, boundary_mode="periodic", target_porosity=0.45):
    app = QApplication.instance() or QApplication(sys.argv)
    editor = InitialStateEditor(domain_size=domain_size, shape_file=shape_file, pom_shape_file=pom_shape_file, psd_file=psd_file, output_dir=output_dir, visualization_dir=visualization_dir, input_mat=input_mat, boundary_mode=boundary_mode, target_porosity=target_porosity)
    editor.show()
    return app.exec()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    sys.exit(main(
        domain_size=250,
        shape_file=project_root / "data" / "particleShapes250rotations.mat",
        pom_shape_file=project_root / "data" / "POMshapes250.mat",
        psd_file=project_root / "data" / "loam_bayreuth.csv",
        output_dir=project_root / "output",
        visualization_dir=project_root / "visualizations",
        input_mat=None,
        boundary_mode="periodic",
        target_porosity=0.45,
    ))