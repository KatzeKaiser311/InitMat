import copy
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget
from scipy.io import loadmat, savemat

from .domain import create_domain_folded


class InitialStateEditor(QMainWindow):
    SIZE_RANGES = {"63-200": (63.0, 200.0), "20-63": (20.0, 63.0), "6.3-20": (6.3, 20.0), "2-6.3": (2.0, 6.3), "1-2": (1.0, 2.0)}

    def __init__(self, domain_size, shape_file, pom_shape_file, psd_file, output_dir, visualization_dir, input_mat=None, boundary_mode="periodic", target_porosity=0.45, shape_library_size=250, reactive_seed=1):
        super().__init__()
        self.domain_size = domain_size
        self.shape_file = Path(shape_file)
        self.pom_shape_file = Path(pom_shape_file)
        self.psd_file = Path(psd_file)
        self.input_mat = Path(input_mat) if input_mat else None
        self.output_dir = Path(output_dir)
        self.visualization_dir = Path(visualization_dir)
        self.boundary_mode = boundary_mode
        self.target_porosity = target_porosity
        self.shape_library_size = shape_library_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visualization_dir.mkdir(parents=True, exist_ok=True)

        n = domain_size
        self.bulk_mask = np.zeros((n, n), dtype=bool)
        self.bulk_type_values = np.zeros((n, n), dtype=float)
        self.particle_type = np.zeros((n, n), dtype=np.int32)
        self.particle_meta = {}
        self.next_particle_id = 1
        self.random_particle_ids = set()

        self.pom_mask = np.zeros((n, n), dtype=bool)
        self.pom_type = np.zeros((n, n), dtype=np.int32)
        self.pom_meta = {}
        self.next_pom_id = 1
        self.pom_conc_values = np.zeros((n, n), dtype=float)
        self.pom_age_values = np.zeros((n, n), dtype=float)

        self.cb_values = np.zeros((n, n), dtype=float)
        self.cb_mask = np.zeros((n, n), dtype=bool)
        self.show_cb = True

        self.g = create_domain_folded(n)
        self.reactive_surface = np.zeros((4 * n * n, 1), dtype=float)
        self.reactive_initialized = False
        self.show_reactive = True
        self.raw_state = {}

        self.undo_stack = []
        self.shape_cache = {}
        self.pom_shape_cache = {}
        self.rng = np.random.default_rng()
        self.reactive_rng = np.random.default_rng(reactive_seed)

        self.drag_particle_id = 0
        self.drag_mouse_cell = None
        self.drag_snapshot = None
        self.drag_changed = False
        self.selected_particle_id = 0
        self.snap_enabled = False

        self.load_particle_library()
        self.load_pom_library()
        self.load_psd()
        self.load_input_state()
        self.build_ui()
        self.draw_domain()

    def load_particle_library(self):
        data = loadmat(self.shape_file, squeeze_me=False, struct_as_record=False)
        self.shape_list = data["fullParticleList"].reshape(-1)
        self.shape_areas = data["fullParticleAreas"].reshape(-1).astype(int)
        self.min_feret = data["fullParticleMinFeretDiameters"].reshape(-1).astype(float)
        self.size_groups = {key: np.where((self.min_feret >= low) & (self.min_feret < high))[0] for key, (low, high) in self.SIZE_RANGES.items()}

    def load_pom_library(self):
        data = loadmat(self.pom_shape_file, squeeze_me=False, struct_as_record=False)
        self.pom_shape_list = data["POMparticleShapesList"].reshape(-1)
        self.pom_shape_areas = data["POMparticleAreas"].reshape(-1).astype(int)
        self.pom_min_feret = data["POMminFeret"].reshape(-1).astype(float)
        self.pom_size_groups = {key: np.where((self.pom_min_feret >= low) & (self.pom_min_feret < high))[0] for key, (low, high) in self.SIZE_RANGES.items()}

    def load_psd(self):
        self.psd = np.atleast_2d(np.genfromtxt(self.psd_file, delimiter=",", dtype=float))

    def vector2d(self, value, default=0.0):
        a = np.asarray(value).reshape(-1)
        if a.size != self.domain_size**2:
            return np.full((self.domain_size, self.domain_size), default, dtype=float)
        return a.reshape((self.domain_size, self.domain_size), order="F").astype(float)

    def type_from_list(self, names, mask):
        result = np.zeros((self.domain_size, self.domain_size), dtype=np.int32)
        for name in names:
            if name not in self.raw_state:
                continue
            for pid, item in enumerate(self.raw_state[name].reshape(-1), start=1):
                inds = np.asarray(item).reshape(-1).astype(int) - 1
                inds = inds[(inds >= 0) & (inds < self.domain_size**2)]
                result[inds % self.domain_size, inds // self.domain_size] = pid
            return result
        result[mask] = np.arange(1, int(mask.sum()) + 1, dtype=np.int32)
        return result

    def load_input_state(self):
        if self.input_mat is None or not self.input_mat.is_file():
            return

        data = loadmat(self.input_mat, squeeze_me=False)
        self.raw_state = {k: v for k, v in data.items() if not k.startswith("__")}

        if "POMVector" in self.raw_state:
            self.pom_mask = self.vector2d(self.raw_state["POMVector"]) > 0

        if "bulkVector" in self.raw_state:
            self.bulk_mask = (self.vector2d(self.raw_state["bulkVector"]) > 0) & ~self.pom_mask

        if "bulkTypeVector" in self.raw_state:
            self.bulk_type_values = self.vector2d(self.raw_state["bulkTypeVector"])
        else:
            self.bulk_type_values[self.bulk_mask | self.pom_mask] = -1

        if "particleTypeVector" in self.raw_state:
            self.particle_type = self.vector2d(self.raw_state["particleTypeVector"]).astype(np.int32)
            self.particle_type[~self.bulk_mask] = 0
        else:
            self.particle_type = self.type_from_list(("solidParticleList", "particleList"), self.bulk_mask)

        self.pom_type = self.type_from_list(("POMParticleList",), self.pom_mask)
        self.pom_type[~self.pom_mask] = 0
        self.pom_conc_values = self.vector2d(self.raw_state.get("POMconcVector", self.pom_mask.astype(float)))
        self.pom_age_values = self.vector2d(self.raw_state.get("POMageVector", self.pom_mask.astype(float)))
        self.cb_values = self.vector2d(self.raw_state.get("C_BVector", np.zeros(self.domain_size**2)))
        self.cb_mask = self.cb_values > 0

        if "reactiveSurfaceVector" in self.raw_state:
            reactive = np.asarray(self.raw_state["reactiveSurfaceVector"]).reshape(-1, 1).astype(float)
            if reactive.size == 4 * self.domain_size**2:
                self.reactive_surface = reactive
                self.reactive_initialized = True

        self.cleanup_bulk_particles()
        self.cleanup_pom_particles()

        feret = np.asarray(self.raw_state.get("particleMinFeret", self.raw_state.get("particleMinFeretDiameters", []))).reshape(-1)
        for pid, value in enumerate(feret[:len(self.particle_meta)], start=1):
            self.particle_meta[pid]["feret"] = float(value)

    def build_ui(self):
        self.setWindowTitle("InitMat Initial State Editor")
        self.resize(1250, 950)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 6, 8, 6)
        main_layout.setSpacing(4)

        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.setHandleWidth(7)
        self.splitter.setChildrenCollapsible(False)
        main_layout.addWidget(self.splitter)

        control_panel = QWidget()
        control_panel.setMinimumHeight(150)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(4)

        info = QGridLayout()
        info.setContentsMargins(0, 0, 0, 0)
        info.setHorizontalSpacing(8)
        info.setVerticalSpacing(2)

        for row, (name, value) in enumerate([
            ("Shape library:", self.shape_file),
            ("POM library:", self.pom_shape_file),
            ("PSD:", self.psd_file),
            ("Input MAT:", self.input_mat if self.input_mat else "None"),
        ]):
            info.addWidget(QLabel(name), row, 0)
            info.addWidget(QLabel(str(value)), row, 1)

        control_layout.addLayout(info)

        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.bulk_tab, self.reactive_tab, self.pom_tab, self.cb_tab = QWidget(), QWidget(), QWidget(), QWidget()

        self.tabs.addTab(self.bulk_tab, "Bulk")
        self.tabs.addTab(self.reactive_tab, "Reactive Surface")
        self.tabs.addTab(self.pom_tab, "POM")
        self.tabs.addTab(self.cb_tab, "CB")
        control_layout.addWidget(self.tabs)

        self.build_bulk_tab()
        self.build_reactive_tab()
        self.build_pom_tab()
        self.build_cb_tab()

        bottom = QHBoxLayout()
        self.bulk_label, self.pom_label, self.cb_label, self.pore_label, self.porosity_label, self.reactive_label = QLabel(), QLabel(), QLabel(), QLabel(), QLabel(), QLabel()

        for widget in (self.bulk_label, self.pom_label, self.cb_label, self.pore_label, self.porosity_label, self.reactive_label):
            bottom.addWidget(widget)

        bottom.addStretch()

        save_mat_button, save_png_button = QPushButton("Save MAT"), QPushButton("Save PNG")
        save_mat_button.clicked.connect(self.save_mat)
        save_png_button.clicked.connect(self.save_png)
        bottom.addWidget(save_mat_button)
        bottom.addWidget(save_png_button)
        control_layout.addLayout(bottom)

        self.message_label = QLabel("Loaded input MAT" if self.raw_state else "Ready")
        control_layout.addWidget(self.message_label)

        self.cell_pixels = max(4, min(20, 1000 // self.domain_size))
        canvas_size = self.domain_size * self.cell_pixels + 1
        self.canvas = QLabel()
        self.canvas.setFixedSize(canvas_size, canvas_size)
        self.canvas.setMouseTracking(True)
        self.canvas.installEventFilter(self)

        self.scroll = QScrollArea()
        self.scroll.setWidget(self.canvas)
        self.scroll.setWidgetResizable(False)

        self.splitter.addWidget(control_panel)
        self.splitter.addWidget(self.scroll)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([210, 740])

    def build_bulk_tab(self):
        layout = QVBoxLayout(self.bulk_tab)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(5)

        row1 = QHBoxLayout()

        self.porosity_box = QDoubleSpinBox()
        self.porosity_box.setRange(0.01, 0.99)
        self.porosity_box.setDecimals(3)
        self.porosity_box.setSingleStep(0.01)
        self.porosity_box.setValue(self.target_porosity)

        randomize_button = QPushButton("Randomize PSD")
        randomize_button.clicked.connect(self.randomize_bulk)

        row1.addWidget(QLabel("Porosity"))
        row1.addWidget(self.porosity_box)
        row1.addWidget(randomize_button)
        row1.addStretch()
        layout.addLayout(row1)

        tools = QHBoxLayout()

        self.bulk_tool_box = QComboBox()
        self.bulk_tool_box.addItems(["Cell", "Particle"])

        self.bulk_action_box = QComboBox()
        self.bulk_action_box.addItems(["Add", "Delete"])

        self.bulk_range_box = QComboBox()
        for key, (low, high) in self.SIZE_RANGES.items():
            self.bulk_range_box.addItem(f">= {low:g} and < {high:g}", key)

        self.boundary_box = QComboBox()
        self.boundary_box.addItems(["Periodic", "Crop"])
        self.boundary_box.setCurrentText("Periodic" if self.boundary_mode == "periodic" else "Crop")

        self.bulk_search_radius = QSpinBox()
        self.bulk_search_radius.setRange(0, 20)
        self.bulk_search_radius.setValue(5)

        self.bulk_shape_trials = QSpinBox()
        self.bulk_shape_trials.setRange(1, 100)
        self.bulk_shape_trials.setValue(20)

        self.bulk_candidate_label = QLabel()

        for label, widget in [
            ("Tool", self.bulk_tool_box),
            ("Action", self.bulk_action_box),
            ("Size", self.bulk_range_box),
            ("Boundary", self.boundary_box),
            ("Search radius", self.bulk_search_radius),
            ("Shape trials", self.bulk_shape_trials),
        ]:
            tools.addWidget(QLabel(label))
            tools.addWidget(widget)

        tools.addWidget(self.bulk_candidate_label)
        tools.addStretch()
        layout.addLayout(tools)

        buttons = QHBoxLayout()

        self.move_button = QPushButton("Move OFF")
        self.move_button.setCheckable(True)

        self.snap_button = QPushButton("Snap OFF")
        self.snap_button.setCheckable(True)

        undo_button = QPushButton("Undo")
        clear_button = QPushButton("Clear Bulk")

        self.move_button.toggled.connect(lambda on: self.move_button.setText("Move ON" if on else "Move OFF"))
        self.snap_button.toggled.connect(self.set_snap)
        undo_button.clicked.connect(self.undo)
        clear_button.clicked.connect(self.clear_bulk)

        for widget in (self.move_button, self.snap_button, undo_button, clear_button):
            buttons.addWidget(widget)

        buttons.addStretch()
        layout.addLayout(buttons)
        layout.addStretch()

        self.bulk_range_box.currentIndexChanged.connect(self.update_statistics)
        self.boundary_box.currentTextChanged.connect(self.set_boundary_mode)

    def build_reactive_tab(self):
        layout = QHBoxLayout(self.reactive_tab)
        layout.setContentsMargins(8, 6, 8, 6)

        randomize_button = QPushButton("Randomize")
        undo_button = QPushButton("Undo")
        clear_button = QPushButton("Clear")
        self.reactive_visibility_button = QPushButton("Hide")

        randomize_button.clicked.connect(self.randomize_reactive_surface)
        undo_button.clicked.connect(self.undo)
        clear_button.clicked.connect(self.clear_reactive_surface)
        self.reactive_visibility_button.clicked.connect(self.toggle_reactive_visibility)

        for widget in (randomize_button, undo_button, clear_button, self.reactive_visibility_button):
            layout.addWidget(widget)

        layout.addStretch()

    def build_pom_tab(self):
        layout = QVBoxLayout(self.pom_tab)
        layout.setContentsMargins(8, 6, 8, 6)

        tools = QHBoxLayout()

        self.pom_action_box = QComboBox()
        self.pom_action_box.addItems(["Add", "Delete"])

        self.pom_range_box = QComboBox()
        for key, (low, high) in self.SIZE_RANGES.items():
            if self.pom_size_groups[key].size:
                self.pom_range_box.addItem(f">= {low:g} and < {high:g}", key)

        self.pom_search_radius = QSpinBox()
        self.pom_search_radius.setRange(1, 30)
        self.pom_search_radius.setValue(10)

        self.pom_shape_trials = QSpinBox()
        self.pom_shape_trials.setRange(1, 100)
        self.pom_shape_trials.setValue(20)

        self.pom_candidate_label = QLabel()

        for label, widget in [
            ("Action", self.pom_action_box),
            ("Size", self.pom_range_box),
            ("Search radius", self.pom_search_radius),
            ("Shape trials", self.pom_shape_trials),
        ]:
            tools.addWidget(QLabel(label))
            tools.addWidget(widget)

        tools.addWidget(self.pom_candidate_label)
        tools.addStretch()
        layout.addLayout(tools)

        buttons = QHBoxLayout()
        undo_button = QPushButton("Undo")
        clear_button = QPushButton("Clear POM")

        undo_button.clicked.connect(self.undo)
        clear_button.clicked.connect(self.clear_pom)

        buttons.addWidget(undo_button)
        buttons.addWidget(clear_button)
        buttons.addStretch()

        layout.addLayout(buttons)
        layout.addStretch()
        self.pom_range_box.currentIndexChanged.connect(self.update_statistics)

    def build_cb_tab(self):
        layout = QHBoxLayout(self.cb_tab)
        layout.setContentsMargins(8, 6, 8, 6)

        self.cb_visibility_button = QPushButton("Hide")
        self.cb_visibility_button.clicked.connect(self.toggle_cb_visibility)

        layout.addWidget(QLabel("C_BVector is loaded and preserved; this layer is display-only."))
        layout.addWidget(self.cb_visibility_button)
        layout.addStretch()

    def mouse_cell(self, event, clamp=False):
        col = int(event.position().x()) // self.cell_pixels
        row = int(event.position().y()) // self.cell_pixels

        if clamp:
            row = min(max(row, 0), self.domain_size - 1)
            col = min(max(col, 0), self.domain_size - 1)

        return row, col

    def eventFilter(self, obj, event):
        if obj is not self.canvas:
            return super().eventFilter(obj, event)

        if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            row, col = self.mouse_cell(event)

            if not (0 <= row < self.domain_size and 0 <= col < self.domain_size):
                return True

            if self.tabs.currentWidget() is self.bulk_tab:
                if self.move_button.isChecked():
                    self.start_drag(row, col)
                else:
                    self.handle_bulk_click(row, col)

            elif self.tabs.currentWidget() is self.pom_tab:
                self.handle_pom_click(row, col)

            return True

        if event.type() == QEvent.Type.MouseMove and self.drag_particle_id:
            self.drag_to(*self.mouse_cell(event, clamp=True))
            return True

        if event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton and self.drag_particle_id:
            self.finish_drag()
            return True

        return super().eventFilter(obj, event)

    def handle_bulk_click(self, row, col):
        tool = self.bulk_tool_box.currentText()
        action = self.bulk_action_box.currentText()

        if tool == "Cell" and action == "Add":
            self.add_bulk_cell(row, col)
        elif tool == "Cell":
            self.delete_bulk_cell(row, col)
        elif action == "Add":
            self.add_bulk_particle(row, col)
        else:
            self.delete_bulk_particle(row, col)

    def handle_pom_click(self, row, col):
        if self.pom_action_box.currentText() == "Add":
            self.add_pom_particle(row, col)
        else:
            self.delete_pom_particle(row, col)

    def snapshot(self):
        return (
            self.bulk_mask.copy(),
            self.bulk_type_values.copy(),
            self.particle_type.copy(),
            copy.deepcopy(self.particle_meta),
            self.next_particle_id,
            set(self.random_particle_ids),
            self.pom_mask.copy(),
            self.pom_type.copy(),
            copy.deepcopy(self.pom_meta),
            self.next_pom_id,
            self.pom_conc_values.copy(),
            self.pom_age_values.copy(),
            self.reactive_surface.copy(),
            self.reactive_initialized,
        )

    def push_undo(self, state=None):
        self.undo_stack.append(state if state is not None else self.snapshot())

        if len(self.undo_stack) > 20:
            self.undo_stack.pop(0)

    def undo(self):
        if not self.undo_stack:
            return

        state = self.undo_stack.pop()

        (
            self.bulk_mask,
            self.bulk_type_values,
            self.particle_type,
            self.particle_meta,
            self.next_particle_id,
            self.random_particle_ids,
            self.pom_mask,
            self.pom_type,
            self.pom_meta,
            self.next_pom_id,
            self.pom_conc_values,
            self.pom_age_values,
            self.reactive_surface,
            self.reactive_initialized,
        ) = state

        self.selected_particle_id = 0
        self.message_label.setText("Undo")
        self.draw_domain()

    def set_snap(self, enabled):
        self.snap_enabled = enabled
        self.snap_button.setText("Snap ON" if enabled else "Snap OFF")

    def clear_particle_reactive(self, particle_id):
        cells = np.flatnonzero((self.particle_type == particle_id).flatten(order="F"))

        if cells.size:
            ce0t = self.g["CE0T"].astype(int) - 1
            self.reactive_surface[ce0t[cells].reshape(-1), 0] = 0

    def clear_bulk(self):
        if not self.bulk_mask.any():
            return

        self.push_undo()
        self.bulk_type_values[self.bulk_mask] = 0
        self.bulk_mask.fill(False)
        self.particle_type.fill(0)
        self.particle_meta = {}
        self.random_particle_ids.clear()
        self.next_particle_id = 1
        self.reactive_surface.fill(0)
        self.reactive_initialized = True
        self.selected_particle_id = 0

        self.message_label.setText("Bulk cleared")
        self.draw_domain()

    def clear_pom(self):
        if not self.pom_mask.any():
            return

        self.push_undo()
        self.bulk_type_values[self.pom_mask] = 0
        self.pom_mask.fill(False)
        self.pom_type.fill(0)
        self.pom_meta = {}
        self.next_pom_id = 1
        self.pom_conc_values.fill(0)
        self.pom_age_values.fill(0)

        self.message_label.setText("POM cleared")
        self.draw_domain()

    def add_bulk_cell(self, row, col):
        if self.bulk_mask[row, col] or self.pom_mask[row, col] or self.cb_mask[row, col]:
            return

        self.push_undo()

        pid = self.next_particle_id
        self.next_particle_id += 1

        self.bulk_mask[row, col] = True
        self.bulk_type_values[row, col] = -1
        self.particle_type[row, col] = pid
        self.particle_meta[pid] = {"shape_id": 0, "feret": 1.0, "center": (row, col), "area": 1}

        self.draw_domain()

    def delete_bulk_cell(self, row, col):
        pid = int(self.particle_type[row, col])

        if pid == 0:
            return

        self.push_undo()
        self.random_particle_ids.discard(pid)
        self.clear_particle_reactive(pid)

        self.bulk_mask[row, col] = False
        self.bulk_type_values[row, col] = 0
        self.particle_type[row, col] = 0

        self.cleanup_bulk_particles()
        self.draw_domain()

    def delete_bulk_particle(self, row, col):
        pid = int(self.particle_type[row, col])

        if pid == 0:
            return

        self.push_undo()
        self.clear_particle_reactive(pid)

        mask = self.particle_type == pid
        self.bulk_mask[mask] = False
        self.bulk_type_values[mask] = 0
        self.particle_type[mask] = 0

        self.random_particle_ids.discard(pid)
        self.cleanup_bulk_particles()
        self.selected_particle_id = 0

        self.draw_domain()

    def delete_pom_particle(self, row, col):
        pid = int(self.pom_type[row, col])

        if pid == 0:
            return

        self.push_undo()

        mask = self.pom_type == pid
        self.pom_mask[mask] = False
        self.pom_type[mask] = 0
        self.pom_conc_values[mask] = 0
        self.pom_age_values[mask] = 0
        self.bulk_type_values[mask] = 0

        self.cleanup_pom_particles()
        self.draw_domain()

    def get_shape(self, shape_list, cache, shape_index):
        shape_index = int(shape_index)

        if shape_index in cache:
            return cache[shape_index]

        indices = np.asarray(shape_list[shape_index]).reshape(-1).astype(np.int64) - 1
        rows = indices % self.shape_library_size
        cols = indices // self.shape_library_size

        rows -= rows.min()
        cols -= cols.min()

        center_row = (rows.max() + rows.min()) // 2
        center_col = (cols.max() + cols.min()) // 2

        cache[shape_index] = (rows - center_row, cols - center_col)

        return cache[shape_index]

    def candidate_cells(self, shape_list, cache, shape_index, center):
        row_offsets, col_offsets = self.get_shape(shape_list, cache, shape_index)

        rows = center[0] + row_offsets
        cols = center[1] + col_offsets

        if self.boundary_mode == "periodic":
            rows %= self.domain_size
            cols %= self.domain_size

            linear = rows * self.domain_size + cols

            if np.unique(linear).size != linear.size:
                return None

        else:
            inside = (rows >= 0) & (rows < self.domain_size) & (cols >= 0) & (cols < self.domain_size)
            rows, cols = rows[inside], cols[inside]

            if rows.size == 0:
                return None

        return rows.astype(int), cols.astype(int)

    def ring_centers(self, row, col, radius):
        centers = [(row, col)] if radius == 0 else [(row + dr, col + dc) for dr in range(-radius, radius + 1) for dc in range(-radius, radius + 1) if max(abs(dr), abs(dc)) == radius]

        if self.boundary_mode == "periodic":
            centers = list({(r % self.domain_size, c % self.domain_size) for r, c in centers})
        else:
            centers = [(r, c) for r, c in centers if 0 <= r < self.domain_size and 0 <= c < self.domain_size]

        if centers:
            order = self.rng.permutation(len(centers))
            centers = [centers[i] for i in order]

        return centers

    def count_bulk_contacts(self, rows, cols):
        candidate = set((rows * self.domain_size + cols).tolist())
        count = 0

        for row, col in zip(rows, cols):
            for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if self.boundary_mode == "periodic":
                    nr %= self.domain_size
                    nc %= self.domain_size
                elif not (0 <= nr < self.domain_size and 0 <= nc < self.domain_size):
                    continue

                if nr * self.domain_size + nc not in candidate and self.bulk_mask[nr, nc]:
                    count += 1

        return count

    def find_bulk_position(self, shape_index, desired_center, prefer_contact):
        occupied = self.bulk_mask | self.pom_mask | self.cb_mask
        fallback = None

        for radius in range(self.bulk_search_radius.value() + 1):
            valid = []

            for center in self.ring_centers(*desired_center, radius):
                cells = self.candidate_cells(self.shape_list, self.shape_cache, shape_index, center)

                if cells is None:
                    continue

                rows, cols = cells

                if occupied[rows, cols].any():
                    continue

                valid.append((center, rows, cols, self.count_bulk_contacts(rows, cols)))

            if not valid:
                continue

            if prefer_contact:
                touching = [item for item in valid if item[3] > 0]

                if touching:
                    weights = np.array([1 + item[3] for item in touching], dtype=float)
                    return touching[int(self.rng.choice(len(touching), p=weights / weights.sum()))]

                if fallback is None:
                    fallback = valid[int(self.rng.integers(len(valid)))]

            else:
                return valid[int(self.rng.integers(len(valid)))]

        return fallback

    def add_bulk_particle(self, row, col):
        key = self.bulk_range_box.currentData()
        group = self.size_groups[key]

        if group.size == 0:
            QMessageBox.warning(self, "No particles", f"No real particle shapes exist in range {key}.")
            return

        prefer_contact = self.snap_enabled and self.SIZE_RANGES[key][1] <= 20 and bool(self.bulk_mask.any())
        fallback = None

        for _ in range(self.bulk_shape_trials.value()):
            shape_index = int(group[self.rng.integers(group.size)])
            placement = self.find_bulk_position(shape_index, (row, col), prefer_contact)

            if placement is None:
                continue

            if not prefer_contact or placement[3] > 0:
                self.commit_bulk_particle(shape_index, placement)
                return

            if fallback is None:
                fallback = (shape_index, placement)

        if fallback is not None:
            self.commit_bulk_particle(*fallback)
        else:
            self.message_label.setText("No collision-free bulk position found")

    def commit_bulk_particle(self, shape_index, placement, generated=False, push=True):
        center, rows, cols, contacts = placement

        if push:
            self.push_undo()

        pid = self.next_particle_id
        self.next_particle_id += 1

        self.bulk_mask[rows, cols] = True
        self.bulk_type_values[rows, cols] = -1
        self.particle_type[rows, cols] = pid
        self.particle_meta[pid] = {"shape_id": shape_index + 1, "feret": float(self.min_feret[shape_index]), "center": center, "area": len(rows)}

        if generated:
            self.random_particle_ids.add(pid)

        self.message_label.setText(f"Bulk shape {shape_index + 1}, MinFeret={self.min_feret[shape_index]:.2f}, cells={len(rows)}, contacts={contacts}")

        if push:
            self.draw_domain()

    def remove_random_bulk(self):
        for pid in list(self.random_particle_ids):
            self.clear_particle_reactive(pid)

            mask = self.particle_type == pid
            self.bulk_mask[mask] = False
            self.bulk_type_values[mask] = 0
            self.particle_type[mask] = 0

        self.random_particle_ids.clear()
        self.cleanup_bulk_particles()

    def psd_bin(self, feret):
        for i, (low, high, _) in enumerate(self.psd):
            if low <= feret < high or i == len(self.psd) - 1 and feret == high:
                return i

        return None

    def place_random_shape(self, shape_index, trials=300):
        occupied = self.bulk_mask | self.pom_mask | self.cb_mask

        for _ in range(trials):
            center = (int(self.rng.integers(self.domain_size)), int(self.rng.integers(self.domain_size)))
            cells = self.candidate_cells(self.shape_list, self.shape_cache, shape_index, center)

            if cells is None:
                continue

            rows, cols = cells

            if len(rows) != int(self.shape_areas[shape_index]) or occupied[rows, cols].any():
                continue

            self.commit_bulk_particle(shape_index, (center, rows, cols, self.count_bulk_contacts(rows, cols)), generated=True, push=False)

            return len(rows)

        return 0

    def fill_psd_bin(self, bin_index, target_cells, global_remaining):
        low, high, _ = self.psd[bin_index]
        group = np.where((self.min_feret >= low) & (self.min_feret < high))[0]

        placed = 0
        failures = 0

        while placed < target_cells and placed < global_remaining and failures < 80:
            room = min(target_cells - placed, global_remaining - placed)
            candidates = group[self.shape_areas[group] <= room]

            if candidates.size == 0:
                break

            shape_index = int(candidates[self.rng.integers(candidates.size)])
            added = self.place_random_shape(shape_index)

            if added:
                placed += added
                failures = 0
            else:
                failures += 1

        return placed

    def randomize_bulk(self):
        target = int(round(self.domain_size**2 * (1 - self.porosity_box.value())))

        self.push_undo()
        self.remove_random_bulk()
        self.selected_particle_id = 0

        fixed = int(self.bulk_mask.sum())

        if fixed >= target:
            self.message_label.setText(f"Fixed bulk already has {fixed} cells; target is {target}")
            self.draw_domain()
            return

        fractions = self.psd[:, 2].astype(float)
        fractions /= fractions.sum()

        quotas = fractions * target
        fixed_bins = np.zeros(len(self.psd), dtype=float)

        for pid, meta in self.particle_meta.items():
            i = self.psd_bin(float(meta["feret"]))

            if i is not None:
                fixed_bins[i] += int((self.particle_type == pid).sum())

        deficits = np.maximum(quotas - fixed_bins, 0)
        remaining = target - fixed
        weights = deficits / deficits.sum() if deficits.sum() else fractions
        targets = weights * remaining

        for i in range(len(self.psd) - 1, -1, -1):
            desired = min(remaining, int(round(targets[i])))

            if desired <= 0:
                continue

            remaining -= self.fill_psd_bin(i, desired, remaining)

        all_candidates = np.argsort(self.shape_areas)
        failures = 0

        while remaining > 0 and failures < 100:
            candidates = all_candidates[self.shape_areas[all_candidates] <= remaining]

            if candidates.size == 0:
                break

            added = self.place_random_shape(int(candidates[self.rng.integers(candidates.size)]))

            if added:
                remaining -= added
                failures = 0
            else:
                failures += 1

        actual = int(self.bulk_mask.sum())
        self.message_label.setText(f"PSD randomized: bulk={actual}, target={target}, porosity={1 - actual / self.domain_size**2:.4f}")
        self.draw_domain()

    def start_drag(self, row, col):
        pid = int(self.particle_type[row, col])

        if pid == 0:
            self.selected_particle_id = 0
            self.draw_domain()
            return

        self.drag_particle_id = pid
        self.selected_particle_id = pid
        self.drag_mouse_cell = (row, col)
        self.drag_snapshot = self.snapshot()
        self.drag_changed = False

        self.draw_domain()

    def try_move_particle(self, pid, dr, dc):
        rows, cols = np.where(self.particle_type == pid)
        new_rows = rows + dr
        new_cols = cols + dc

        if np.any(new_rows < 0) or np.any(new_rows >= self.domain_size) or np.any(new_cols < 0) or np.any(new_cols >= self.domain_size):
            return False

        other_bulk = self.bulk_mask.copy()
        other_bulk[rows, cols] = False

        if (other_bulk[new_rows, new_cols] | self.pom_mask[new_rows, new_cols] | self.cb_mask[new_rows, new_cols]).any():
            return False

        ce0t = self.g["CE0T"].astype(int) - 1
        old_cells = rows + cols * self.domain_size
        new_cells = new_rows + new_cols * self.domain_size

        reactive_flags = self.reactive_surface[ce0t[old_cells], 0].copy()
        type_values = self.bulk_type_values[rows, cols].copy()

        self.reactive_surface[ce0t[old_cells].reshape(-1), 0] = 0

        self.bulk_mask[rows, cols] = False
        self.bulk_type_values[rows, cols] = 0
        self.particle_type[rows, cols] = 0

        self.bulk_mask[new_rows, new_cols] = True
        self.bulk_type_values[new_rows, new_cols] = type_values
        self.particle_type[new_rows, new_cols] = pid
        self.reactive_surface[ce0t[new_cells], 0] = reactive_flags

        center = self.particle_meta[pid]["center"]
        self.particle_meta[pid]["center"] = (center[0] + dr, center[1] + dc)

        self.random_particle_ids.discard(pid)

        return True

    def drag_to(self, row, col):
        old_row, old_col = self.drag_mouse_cell
        dr_total = row - old_row
        dc_total = col - old_col
        steps = max(abs(dr_total), abs(dc_total))

        if steps == 0:
            return

        previous_row = old_row
        previous_col = old_col

        for step in range(1, steps + 1):
            current_row = int(round(old_row + dr_total * step / steps))
            current_col = int(round(old_col + dc_total * step / steps))
            dr = current_row - previous_row
            dc = current_col - previous_col

            if (dr or dc) and self.try_move_particle(self.drag_particle_id, dr, dc):
                if not self.drag_changed:
                    self.push_undo(self.drag_snapshot)
                    self.drag_changed = True

            previous_row = current_row
            previous_col = current_col

        self.drag_mouse_cell = (row, col)
        self.draw_domain()

    def finish_drag(self):
        self.drag_particle_id = 0
        self.drag_mouse_cell = None
        self.drag_snapshot = None

        self.message_label.setText("Particle moved" if self.drag_changed else "Particle selected")
        self.drag_changed = False

        self.draw_domain()

    def find_pom_position(self, shape_index, desired_center, require_bulk_contact):
        for radius in range(1, self.pom_search_radius.value() + 1):
            valid = []

            for center in self.ring_centers(*desired_center, radius):
                cells = self.candidate_cells(self.pom_shape_list, self.pom_shape_cache, shape_index, center)

                if cells is None:
                    continue

                rows, cols = cells

                if self.bulk_mask[rows, cols].any() or self.pom_mask[rows, cols].any() or self.cb_mask[rows, cols].any():
                    continue

                valid.append((center, rows, cols, self.count_bulk_contacts(rows, cols)))

            if not valid:
                continue

            if require_bulk_contact:
                touching = [item for item in valid if item[3] > 0]

                if not touching:
                    continue

                max_contacts = max(item[3] for item in touching)
                best = [item for item in touching if item[3] == max_contacts]

                return best[int(self.rng.integers(len(best)))]

            return valid[int(self.rng.integers(len(valid)))]

        return None

    def add_pom_particle(self, row, col):
        if self.pom_range_box.count() == 0:
            return

        key = self.pom_range_box.currentData()
        group = self.pom_size_groups[key]

        for _ in range(self.pom_shape_trials.value()):
            shape_index = int(group[self.rng.integers(group.size)])
            direct = self.candidate_cells(self.pom_shape_list, self.pom_shape_cache, shape_index, (row, col))

            if direct is None:
                continue

            rows, cols = direct
            overlap_bulk = self.bulk_mask[rows, cols].any()

            if not overlap_bulk and not self.pom_mask[rows, cols].any() and not self.cb_mask[rows, cols].any():
                self.commit_pom_particle(shape_index, ((row, col), rows, cols, self.count_bulk_contacts(rows, cols)))
                return

            placement = self.find_pom_position(shape_index, (row, col), require_bulk_contact=bool(overlap_bulk))

            if placement is not None:
                self.commit_pom_particle(shape_index, placement)
                return

        self.message_label.setText(f"No valid POM position found for {key}")

    def commit_pom_particle(self, shape_index, placement):
        center, rows, cols, contacts = placement

        self.push_undo()

        pid = self.next_pom_id
        self.next_pom_id += 1

        self.pom_mask[rows, cols] = True
        self.pom_type[rows, cols] = pid
        self.pom_conc_values[rows, cols] = 1
        self.pom_age_values[rows, cols] = 1
        self.bulk_type_values[rows, cols] = -1
        self.pom_meta[pid] = {"shape_id": shape_index + 1, "feret": float(self.pom_min_feret[shape_index]), "center": center, "area": len(rows)}

        self.message_label.setText(f"POM shape {shape_index + 1}, cells={len(rows)}, bulk contacts={contacts}")
        self.draw_domain()

    def cleanup_bulk_particles(self):
        old_ids = np.unique(self.particle_type)
        old_ids = old_ids[old_ids > 0]

        new_type = np.zeros_like(self.particle_type)
        new_meta = {}
        new_random = set()

        for new_id, old_id in enumerate(old_ids, start=1):
            mask = self.particle_type == old_id
            new_type[mask] = new_id

            rows, cols = np.where(mask)
            old = self.particle_meta.get(int(old_id), {})

            new_meta[new_id] = {
                "shape_id": old.get("shape_id", 0),
                "feret": old.get("feret", float(np.sqrt(mask.sum()))),
                "center": old.get("center", (float(rows.mean()), float(cols.mean()))),
                "area": int(mask.sum()),
            }

            if int(old_id) in self.random_particle_ids:
                new_random.add(new_id)

        self.particle_type = new_type
        self.particle_meta = new_meta
        self.random_particle_ids = new_random
        self.next_particle_id = len(old_ids) + 1

    def cleanup_pom_particles(self):
        old_ids = np.unique(self.pom_type)
        old_ids = old_ids[old_ids > 0]

        new_type = np.zeros_like(self.pom_type)
        new_meta = {}

        for new_id, old_id in enumerate(old_ids, start=1):
            mask = self.pom_type == old_id
            new_type[mask] = new_id

            rows, cols = np.where(mask)
            old = self.pom_meta.get(int(old_id), {})

            new_meta[new_id] = {
                "shape_id": old.get("shape_id", 0),
                "feret": old.get("feret", float(np.sqrt(mask.sum()))),
                "center": old.get("center", (float(rows.mean()), float(cols.mean()))),
                "area": int(mask.sum()),
            }

        self.pom_type = new_type
        self.pom_meta = new_meta
        self.next_pom_id = len(old_ids) + 1

    def set_boundary_mode(self, text):
        self.boundary_mode = text.lower()

    def particle_surface_edges(self):
        particle_vector = self.particle_type.flatten(order="F")
        ce0t = self.g["CE0T"].astype(int)
        result = []

        for pid in range(1, self.next_particle_id):
            edges = []

            for cell in np.where(particle_vector == pid)[0]:
                row = cell % self.domain_size
                col = cell // self.domain_size

                neighbors = [
                    ((col - 1) % self.domain_size) * self.domain_size + row,
                    col * self.domain_size + (row - 1) % self.domain_size,
                    col * self.domain_size + (row + 1) % self.domain_size,
                    ((col + 1) % self.domain_size) * self.domain_size + row,
                ]

                for neighbor, direction in zip(neighbors, (0, 3, 2, 1)):
                    if particle_vector[neighbor] != pid:
                        edges.append(ce0t[cell, direction] - 1)

            result.append(np.asarray(edges, dtype=int))

        return result

    def generate_reactive_surface(self):
        reactive = np.zeros_like(self.reactive_surface)

        for pid, edges in enumerate(self.particle_surface_edges(), start=1):
            if edges.size == 0:
                continue

            feret = float(self.particle_meta[pid]["feret"])
            fraction = 1.0 if feret < 6.3 else 0.5 if feret < 20 else 0.25 if feret < 63 else 0.1
            count = int(np.floor(fraction * edges.size + 0.5))

            selected = edges if count == edges.size else self.reactive_rng.choice(edges, count, replace=False)
            reactive[np.asarray(selected, dtype=int), 0] = 1

        return reactive

    def randomize_reactive_surface(self):
        self.push_undo()
        self.reactive_surface = self.generate_reactive_surface()
        self.reactive_initialized = True

        self.message_label.setText("Reactive surface randomized")
        self.draw_domain()

    def clear_reactive_surface(self):
        self.push_undo()
        self.reactive_surface.fill(0)
        self.reactive_initialized = True

        self.message_label.setText("Reactive surface cleared")
        self.draw_domain()

    def toggle_reactive_visibility(self):
        self.show_reactive = not self.show_reactive
        self.reactive_visibility_button.setText("Hide" if self.show_reactive else "Show")
        self.draw_domain()

    def toggle_cb_visibility(self):
        self.show_cb = not self.show_cb
        self.cb_visibility_button.setText("Hide" if self.show_cb else "Show")
        self.draw_domain()

    def update_statistics(self):
        bulk = int(self.bulk_mask.sum())
        pom = int(self.pom_mask.sum())
        cb = int(self.cb_mask.sum())
        pore = self.domain_size**2 - int((self.bulk_mask | self.pom_mask).sum())

        self.bulk_label.setText(f"Bulk: {bulk}")
        self.pom_label.setText(f"POM: {pom}")
        self.cb_label.setText(f"CB: {cb}")
        self.pore_label.setText(f"Pore: {pore}")
        self.porosity_label.setText(f"Bulk porosity: {1 - bulk / self.domain_size**2:.4f}")
        self.reactive_label.setText(f"Reactive edges: {int(self.reactive_surface.sum())}")

        if hasattr(self, "bulk_candidate_label"):
            self.bulk_candidate_label.setText(f"Shapes: {len(self.size_groups[self.bulk_range_box.currentData()]):,}")

        if hasattr(self, "pom_candidate_label") and self.pom_range_box.count():
            self.pom_candidate_label.setText(f"Shapes: {len(self.pom_size_groups[self.pom_range_box.currentData()]):,}")

    def draw_domain(self):
        size = self.domain_size * self.cell_pixels

        image = QImage(size + 1, size + 1, QImage.Format.Format_RGB32)
        image.fill(QColor(255, 255, 255))

        painter = QPainter(image)
        painter.setPen(Qt.PenStyle.NoPen)

        painter.setBrush(QColor(139, 69, 19))
        for row, col in zip(*np.where(self.bulk_mask)):
            painter.drawRect(col * self.cell_pixels, row * self.cell_pixels, self.cell_pixels, self.cell_pixels)

        if self.show_cb:
            painter.setBrush(QColor(230, 159, 0))
            for row, col in zip(*np.where(self.cb_mask & ~self.bulk_mask & ~self.pom_mask)):
                painter.drawRect(col * self.cell_pixels, row * self.cell_pixels, self.cell_pixels, self.cell_pixels)

        painter.setBrush(QColor(255, 215, 0))
        for row, col in zip(*np.where(self.pom_mask)):
            painter.drawRect(col * self.cell_pixels, row * self.cell_pixels, self.cell_pixels, self.cell_pixels)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(190, 190, 190), 1))

        for i in range(self.domain_size + 1):
            p = i * self.cell_pixels
            painter.drawLine(p, 0, p, size)
            painter.drawLine(0, p, size, p)

        if self.show_reactive:
            self.draw_reactive_edges_qt(painter)

        if self.selected_particle_id:
            painter.setPen(QPen(QColor(0, 90, 255), max(2, self.cell_pixels // 3)))

            for row, col in zip(*np.where(self.particle_type == self.selected_particle_id)):
                painter.drawRect(col * self.cell_pixels, row * self.cell_pixels, self.cell_pixels, self.cell_pixels)

        painter.end()

        self.canvas.setPixmap(QPixmap.fromImage(image))
        self.update_statistics()

    def draw_reactive_edges_qt(self, painter):
        active = self.reactive_surface[:, 0] > 0
        ce0t = self.g["CE0T"].astype(int) - 1
        cells = np.where(np.any(active[ce0t], axis=1))[0]

        painter.setPen(QPen(QColor(0, 180, 0), max(2, self.cell_pixels // 3)))

        for cell in cells:
            row = cell % self.domain_size
            col = cell // self.domain_size
            x = col * self.cell_pixels
            y = row * self.cell_pixels
            p = self.cell_pixels
            flags = active[ce0t[cell]]

            if flags[0]:
                painter.drawLine(x, y, x, y + p)
            if flags[1]:
                painter.drawLine(x + p, y, x + p, y + p)
            if flags[2]:
                painter.drawLine(x, y + p, x + p, y + p)
            if flags[3]:
                painter.drawLine(x, y, x + p, y)

    def create_particle_list(self, type_array):
        num_particles = int(type_array.max())
        result = np.empty((1, num_particles), dtype=object)

        for pid in range(1, num_particles + 1):
            indices = np.flatnonzero((type_array == pid).flatten(order="F")) + 1
            result[0, pid - 1] = indices.astype(float).reshape(1, -1)

        return result

    def save_mat(self):
        self.cleanup_bulk_particles()
        self.cleanup_pom_particles()

        path, _ = QFileDialog.getSaveFileName(self, "Save MAT", str(self.output_dir / "initial_state.mat"), "MATLAB (*.mat)")

        if not path:
            return

        path = Path(path)

        if path.suffix.lower() != ".mat":
            path = path.with_suffix(".mat")

        if not self.reactive_initialized:
            self.reactive_surface = self.generate_reactive_surface()
            self.reactive_initialized = True

        bulk_vector = (self.bulk_mask | self.pom_mask).flatten(order="F").reshape(-1, 1).astype(float)

        bulk_type = self.bulk_type_values.copy()
        bulk_type[~(self.bulk_mask | self.pom_mask)] = 0
        bulk_type[(self.bulk_mask | self.pom_mask) & (bulk_type == 0)] = -1

        particle_list = self.create_particle_list(self.particle_type)

        output = dict(self.raw_state)

        output.update({
            "g": self.g,
            "bulkVector": bulk_vector,
            "bulkTypeVector": bulk_type.flatten(order="F").reshape(-1, 1),
            "particleTypeVector": self.particle_type.flatten(order="F").reshape(-1, 1).astype(float),
            "particleList": particle_list,
            "reactiveSurfaceVector": self.reactive_surface.copy(),
            "POMVector": self.pom_mask.flatten(order="F").reshape(-1, 1).astype(float),
            "POMconcVector": self.pom_conc_values.flatten(order="F").reshape(-1, 1),
            "POMageVector": self.pom_age_values.flatten(order="F").reshape(-1, 1),
            "POMParticleList": self.create_particle_list(self.pom_type),
            "C_BVector": self.cb_values.flatten(order="F").reshape(-1, 1),
            "particleShapeIDs": np.array([self.particle_meta[i]["shape_id"] for i in range(1, self.next_particle_id)], dtype=np.int32).reshape(1, -1),
            "particleMinFeret": np.array([self.particle_meta[i]["feret"] for i in range(1, self.next_particle_id)], dtype=float).reshape(1, -1),
            "particleCenters": np.array([self.particle_meta[i]["center"] for i in range(1, self.next_particle_id)], dtype=float).reshape(-1, 2),
        })

        if "solidParticleList" in output:
            output["solidParticleList"] = particle_list

        if "edgeChargeVector" not in output:
            output["edgeChargeVector"] = np.zeros((4 * self.domain_size**2, 1))

        savemat(path, output, do_compression=True, oned_as="column")

        self.message_label.setText(f"Saved MAT: {path}")
        self.draw_domain()

    def save_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save PNG", str(self.visualization_dir / "initial_state.png"), "PNG (*.png)")

        if not path:
            return

        path = Path(path)

        if path.suffix.lower() != ".png":
            path = path.with_suffix(".png")

        p = self.cell_pixels
        size = self.domain_size * self.cell_pixels

        image = Image.new("RGB", (size + 1, size + 1), (255, 255, 255))
        draw = ImageDraw.Draw(image)

        for row, col in zip(*np.where(self.bulk_mask)):
            draw.rectangle([col * p, row * p, (col + 1) * p, (row + 1) * p], fill=(139, 69, 19))

        if self.show_cb:
            for row, col in zip(*np.where(self.cb_mask & ~self.bulk_mask & ~self.pom_mask)):
                draw.rectangle([col * p, row * p, (col + 1) * p, (row + 1) * p], fill=(230, 159, 0))

        for row, col in zip(*np.where(self.pom_mask)):
            draw.rectangle([col * p, row * p, (col + 1) * p, (row + 1) * p], fill=(255, 215, 0))

        for i in range(self.domain_size + 1):
            q = i * p
            draw.line([(q, 0), (q, size)], fill=(190, 190, 190), width=1)
            draw.line([(0, q), (size, q)], fill=(190, 190, 190), width=1)

        if self.show_reactive:
            self.draw_reactive_edges_pil(draw)

        image.save(path)
        self.message_label.setText(f"Saved PNG: {path}")

    def draw_reactive_edges_pil(self, draw):
        active = self.reactive_surface[:, 0] > 0
        ce0t = self.g["CE0T"].astype(int) - 1
        cells = np.where(np.any(active[ce0t], axis=1))[0]

        p = self.cell_pixels
        width = max(2, self.cell_pixels // 3)

        for cell in cells:
            row = cell % self.domain_size
            col = cell // self.domain_size
            x = col * p
            y = row * p
            flags = active[ce0t[cell]]

            if flags[0]:
                draw.line([(x, y), (x, y + p)], fill=(0, 180, 0), width=width)
            if flags[1]:
                draw.line([(x + p, y), (x + p, y + p)], fill=(0, 180, 0), width=width)
            if flags[2]:
                draw.line([(x, y + p), (x + p, y + p)], fill=(0, 180, 0), width=width)
            if flags[3]:
                draw.line([(x, y), (x + p, y)], fill=(0, 180, 0), width=width)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    app = QApplication(sys.argv)

    editor = InitialStateEditor(
        domain_size=250,
        shape_file=project_root / "data" / "particleShapes250rotations.mat",
        pom_shape_file=project_root / "data" / "POMshapes250.mat",
        psd_file=project_root / "data" / "loam_bayreuth.csv",
        output_dir=project_root / "output",
        visualization_dir=project_root / "visualizations",
        input_mat=None,
        boundary_mode="periodic",
    )

    editor.show()
    sys.exit(app.exec())