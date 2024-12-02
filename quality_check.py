import sys
import os
# import numpy as np
import nibabel as nib
import csv

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSpinBox, QRadioButton, QButtonGroup, QGroupBox, QHBoxLayout, QPushButton, QFrame
# from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MainWindow(QMainWindow):
    def __init__(self, nifti_image_path, mask_image_path):
        super().__init__()
        self.setWindowTitle("1.3 - FatViT - Quality Check")
        self.setWindowIcon(QIcon("assets/ai_icon1.png"))
        self.resize(800, 500)  # Set the initial size of the window (Width, Height)
        
        self.nifti_image_path = nifti_image_path
        self.mask_image_path = mask_image_path
        self.save_folder = os.path.dirname(mask_image_path)  # Save folder from predicted volume
        
        # Load NIfTI files
        self.volume = nib.load(self.nifti_image_path).get_fdata()
        self.mask = nib.load(self.mask_image_path).get_fdata()
        self.slice_ratings_vat = [0] * self.volume.shape[2]  # Initialize VAT ratings with 0 (not rated)
        self.slice_ratings_sat = [0] * self.volume.shape[2]  # Initialize SAT ratings with 0 (not rated)
        self.final_rating_vat = None  # Initialize final VAT rating
        self.final_rating_sat = None  # Initialize final SAT rating
        
        # Create a figure instance to plot on
        self.figure = Figure(figsize=(10, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        
        # Create a spinbox to select slices
        self.slice_selector = QSpinBox()
        self.slice_selector.setRange(0, self.volume.shape[2] - 1)
        self.slice_selector.valueChanged.connect(self.update_plot)
        
        # Create Prev and Next buttons
        self.btn_prev = QPushButton("Prev")
        self.btn_prev.clicked.connect(self.prev_slice)
        
        self.btn_next = QPushButton("Next")
        self.btn_next.clicked.connect(self.next_slice)
        
        # Create radio buttons for VAT quality rating
        self.radio_group_vat = QButtonGroup()
        self.radio_buttons_vat = []
        radio_layout_vat = QHBoxLayout()
        
        for i in range(1, 6):
            radio_btn = QRadioButton(str(i))
            radio_layout_vat.addWidget(radio_btn)
            self.radio_group_vat.addButton(radio_btn, i)
            self.radio_buttons_vat.append(radio_btn)
        
        self.radio_group_vat.buttonClicked.connect(self.save_rating_vat)
        
        radio_group_box_vat = QGroupBox("VAT Quality Rating")
        radio_group_box_vat.setLayout(radio_layout_vat)
        
        # Create radio buttons for SAT quality rating
        self.radio_group_sat = QButtonGroup()
        self.radio_buttons_sat = []
        radio_layout_sat = QHBoxLayout()
        
        for i in range(1, 6):
            radio_btn = QRadioButton(str(i))
            radio_layout_sat.addWidget(radio_btn)
            self.radio_group_sat.addButton(radio_btn, i)
            self.radio_buttons_sat.append(radio_btn)
        
        self.radio_group_sat.buttonClicked.connect(self.save_rating_sat)
        
        radio_group_box_sat = QGroupBox("SAT Quality Rating")
        radio_group_box_sat.setLayout(radio_layout_sat)
        
        # Create final check radio buttons for VAT
        self.final_radio_group_vat = QButtonGroup()
        self.final_radio_buttons_vat = []
        final_radio_layout_vat = QVBoxLayout()
        
        for id, label in enumerate(["Pass", "Fail"], start=1):
            radio_btn = QRadioButton(label)
            final_radio_layout_vat.addWidget(radio_btn)
            self.final_radio_group_vat.addButton(radio_btn, id)
            self.final_radio_buttons_vat.append(radio_btn)
        
        self.final_radio_group_vat.buttonClicked.connect(self.save_final_rating_vat)
        
        final_radio_group_box_vat = QGroupBox("Final VAT Check")
        final_radio_group_box_vat.setLayout(final_radio_layout_vat)
        
        # Create final check radio buttons for SAT
        self.final_radio_group_sat = QButtonGroup()
        self.final_radio_buttons_sat = []
        final_radio_layout_sat = QVBoxLayout()
        
        for id, label in enumerate(["Pass", "Fail"], start=1):
            radio_btn = QRadioButton(label)
            final_radio_layout_sat.addWidget(radio_btn)
            self.final_radio_group_sat.addButton(radio_btn, id)
            self.final_radio_buttons_sat.append(radio_btn)
        
        self.final_radio_group_sat.buttonClicked.connect(self.save_final_rating_sat)
        
        final_radio_group_box_sat = QGroupBox("Final SAT Check")
        final_radio_group_box_sat.setLayout(final_radio_layout_sat)
        
        # Create a save button
        self.save_button = QPushButton("Save Results")
        self.save_button.clicked.connect(self.save_results)
        
        # Create the main layout with vertical spacer and line
        main_layout = QVBoxLayout()
        navigation_layout = QHBoxLayout()
        radio_layout = QHBoxLayout()
        final_check_layout = QHBoxLayout()
        
        # Vertical line
        # line = QFrame()
        # line.setFrameShape(QFrame.VLine)
        # line.setFrameShadow(QFrame.Sunken)
        # line.setStyleSheet("background-color: gray;")
        
        navigation_layout.addWidget(self.btn_prev)
        navigation_layout.addWidget(self.slice_selector)
        navigation_layout.addWidget(self.btn_next)
        
        radio_layout.addWidget(radio_group_box_vat)
        # radio_layout.addWidget(line)
        radio_layout.addWidget(radio_group_box_sat)
        
        final_check_layout.addWidget(final_radio_group_box_vat)
        # final_check_layout.addWidget(line)
        final_check_layout.addWidget(final_radio_group_box_sat)
        
        main_layout.addWidget(self.canvas)
        main_layout.addLayout(navigation_layout)
        main_layout.addLayout(radio_layout)
        main_layout.addLayout(final_check_layout)
        main_layout.addWidget(self.save_button)
        
        # Create a QWidget and set it as the central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Initialize plot
        self.plot(self.volume.shape[2] // 2)
        
    def plot(self, slice_idx):
        # Clear previous figure
        self.figure.clear()
        
        # Extract the specified slice
        img_slice = self.volume[:, :, slice_idx]
        mask_slice = self.mask[:, :, slice_idx]
        
        # Adjust subplots to remove margins
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.01, hspace=0.01)
    
        
        # Create subplots with no margins
        ax1 = self.figure.add_subplot(131)
        ax2 = self.figure.add_subplot(132)
        ax3 = self.figure.add_subplot(133)
        
        # Plot volume slice
        ax1.imshow(img_slice.T, cmap='gray', origin='lower')
        ax1.axis('off')
    
        # Plot mask slice
        ax2.imshow(mask_slice.T, cmap='gray', origin='lower')
        ax2.axis('off')
    
        # Plot overlay
        ax3.imshow(img_slice.T, cmap='gray', origin='lower', interpolation='none')
        ax3.imshow(mask_slice.T, cmap='jet', origin='lower', interpolation='none', alpha=0.7)
        ax3.axis('off')
    
        # Refresh canvas
        self.canvas.draw_idle()
    
        # Update radio button selection based on saved rating
        rating_vat = self.slice_ratings_vat[slice_idx]
        rating_sat = self.slice_ratings_sat[slice_idx]
    
        if rating_vat != 0:
            self.radio_group_vat.button(rating_vat).setChecked(True)
        else:
            self.radio_group_vat.setExclusive(False)
            for radio_btn in self.radio_buttons_vat:
                radio_btn.setChecked(False)
            self.radio_group_vat.setExclusive(True)
    
        if rating_sat != 0:
            self.radio_group_sat.button(rating_sat).setChecked(True)
        else:
            self.radio_group_sat.setExclusive(False)
            for radio_btn in self.radio_buttons_sat:
                radio_btn.setChecked(False)
            self.radio_group_sat.setExclusive(True)

    
    def update_plot(self, value):
        self.plot(value)
    
    def prev_slice(self):
        current_value = self.slice_selector.value()
        if current_value > self.slice_selector.minimum():
            self.slice_selector.setValue(current_value - 1)
    
    def next_slice(self):
        current_value = self.slice_selector.value()
        if current_value < self.slice_selector.maximum():
            self.slice_selector.setValue(current_value + 1)
    
    def save_rating_vat(self):
        current_slice = self.slice_selector.value()
        selected_rating = self.radio_group_vat.checkedId()
        self.slice_ratings_vat[current_slice] = selected_rating
    
    def save_rating_sat(self):
        current_slice = self.slice_selector.value()
        selected_rating = self.radio_group_sat.checkedId()
        self.slice_ratings_sat[current_slice] = selected_rating
    
    def save_final_rating_vat(self):
        self.final_rating_vat = self.final_radio_group_vat.checkedId()
    
    def save_final_rating_sat(self):
        self.final_rating_sat = self.final_radio_group_sat.checkedId()
    
    def save_results(self):
        csv_file_name = os.path.join(self.save_folder, os.path.basename(self.mask_image_path).replace('_pred.nii', '_quality_check.csv'))
        
        with open(csv_file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write header
            header = ["file_name", "vat_final", "sat_final"]
            for i in range(self.volume.shape[2]):
                header.append(f"slice_vat_{i}")
                header.append(f"slice_sat_{i}")
            writer.writerow(header)
            
            # Write data
            row = [os.path.basename(self.nifti_image_path)]
            row.append("Pass" if self.final_rating_vat == 1 else "Fail" if self.final_rating_vat == 2 else "")
            row.append("Pass" if self.final_rating_sat == 1 else "Fail" if self.final_rating_sat == 2 else "")
            for i in range(self.volume.shape[2]):
                row.append(self.slice_ratings_vat[i])
                row.append(self.slice_ratings_sat[i])
            writer.writerow(row)
        
        print(f"Quality check results saved to {csv_file_name}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    nifti_image_path = sys.argv[1]  # Path to NIfTI file passed as argument
    mask_image_path = sys.argv[2]   # Path to mask file passed as argument
    window = MainWindow(nifti_image_path, mask_image_path)
    with open("assets/qss/light3.qss", "r") as file:
        stylesheet = file.read()
        app.setStyleSheet(stylesheet)
        window.show()
    sys.exit(app.exec_())
