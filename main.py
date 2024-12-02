import sys
import torch
import time
import nibabel as nib
import subprocess

from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFileDialog
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt

from monai.data import DataLoader, Dataset, decollate_batch
from monai.transforms import Compose, LoadImaged, SaveImaged, EnsureChannelFirstd, Orientationd, Invertd, AsDiscreted
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("1.3 - FatViT - Automated VAT/SAT Segmentation")
        self.setWindowIcon(QIcon("assets/ai_icon1.png"))
        self.resize(545, 144)  # Set the initial size of the window (Width, Height)
        
        mainLayout = QVBoxLayout()
        # Buttons
        buttonLayout = QHBoxLayout()
        self.btnLoadVolume = QPushButton("Load volume")
        self.btnLoadVolume.clicked.connect(self.show_dialog_open_mr)
        
        self.btnLoadModel = QPushButton("Load model")
        self.btnLoadModel.clicked.connect(self.show_dialog_model_predict)
        
        self.btnAutomatedSeg = QPushButton("Automated segmentation")
        self.btnAutomatedSeg.clicked.connect(self.make_prediction)
        
        self.btnSaveSeg = QPushButton("Select save folder")
        self.btnSaveSeg.clicked.connect(self.show_dialog_save_dir)
        
        self.btnQualityCheck = QPushButton("Quality check")
        self.btnQualityCheck.setEnabled(False)  # Initially disabled
        self.btnQualityCheck.clicked.connect(self.launch_quality_check)
        
        buttonLayout.addWidget(self.btnLoadVolume)
        buttonLayout.addWidget(self.btnLoadModel)
        buttonLayout.addWidget(self.btnSaveSeg)
        buttonLayout.addWidget(self.btnAutomatedSeg)
        buttonLayout.addWidget(self.btnQualityCheck)

        # Model Path Label
        self.lblModelPath = QLabel("No model selected")
        self.lblModelPath.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Results
        volumeLayout = QHBoxLayout()
        resultGroup = QGroupBox("Abdominal Fat Quantification")
        resultLayout = QHBoxLayout(resultGroup)
        
        self.lblVAT = QLabel("VAT (cm³)\n")
        self.lblVAT.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.lblSAT = QLabel("SAT (cm³)\n")
        self.lblSAT.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.lblVATtoSAT = QLabel("VAT/SAT Ratio(cm³)\n")
        self.lblVATtoSAT.setAlignment(Qt.AlignmentFlag.AlignLeft)        
        
        resultLayout.addWidget(self.lblVAT)
        resultLayout.addWidget(self.lblSAT)
        resultLayout.addWidget(self.lblVATtoSAT)
        
        volumeLayout.addWidget(resultGroup)

        mainLayout.addLayout(buttonLayout)
        mainLayout.addWidget(self.lblModelPath)
        mainLayout.addLayout(volumeLayout)
        
        centralWidget = QWidget()
        centralWidget.setLayout(mainLayout)
        self.setCentralWidget(centralWidget)
        
    # def showEvent(self, event):
    #     super().showEvent(event)
    #     print(f"Window size: {self.size().width()} x {self.size().height()}")
        
    def show_dialog_open_mr(self):
        self.nii_path, _ = QFileDialog.getOpenFileName(self, "Open MRI File", "", "Nifti Files (*.nii *.nii.gz);;All Files (*)")
        if self.nii_path:
            self.load_nii(self.nii_path)
        
    def show_dialog_save_dir(self):
        options = QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks
        save_path = QFileDialog.getExistingDirectory(self, "Select Folder", options=options)
        if save_path:
            self.save_path = save_path

    def show_dialog_model_predict(self):
        self.dl_model_path, _ = QFileDialog.getOpenFileName(self, "Open trained DL model", "", "PyTorch model (*.pth)")
        if self.dl_model_path:
            self.lblModelPath.setText(f"{self.inference_time_str if hasattr(self, 'inference_time_str') else ''}")
              
    def load_nii(self, nii_path):
        self.nii_obj = nib.load(self.nii_path)
        self.nii_data = self.nii_obj.get_fdata()
            
    def format_inference_time(self, inference_time):
        if inference_time < 60:
            sec = int(inference_time)
            msec = (inference_time % 1) * 1000
            return f"{sec} seconds:{msec:.2f} milliseconds" 
        else:
            min = int(inference_time / 60)
            sec = round(inference_time % 60)  # Round the seconds part to the nearest whole number
            return f"{min} minutes,{sec} seconds"

    def make_prediction(self):
        try:
            test_files = [self.nii_path]
            print(test_files)
        
            test_transforms = Compose([
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
            ])

            test_files = [{"image": file} for file in test_files]

            test_ds = Dataset(data=test_files, transform=test_transforms)
            test_loader = DataLoader(test_ds, batch_size=1, num_workers=1)
            post_transforms = Compose([
                Invertd(
                    keys="pred",
                    transform=test_transforms,
                    orig_keys="image",
                    meta_keys="pred_meta_dict",
                    orig_meta_keys="image_meta_dict",
                    meta_key_postfix="meta_dict",
                    nearest_interp=False,
                    to_tensor=True,
                    device=device,
                ),
                AsDiscreted(keys="pred", argmax=True),
                SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=self.save_path, output_ext=".nii", output_postfix="pred", separate_folder=False, resample=False),
            ])
        
            num_classes = 3
            patch_size = (64, 64, 32)

            self.model = SwinUNETR(
                img_size=patch_size, 
                in_channels=1,
                out_channels=num_classes,
                feature_size=48,
                spatial_dims=3
            )
            state_dict = torch.load(self.dl_model_path, map_location=device)  # Use the device variable here
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            self.model.load_state_dict(new_state_dict)
            self.model = self.model.to(device)  # Use the device variable here
        
            self.model.eval()

            with torch.no_grad():
                for test_data in test_loader:
                    test_inputs = test_data["image"].to(device)  # Use the device variable here
                    sw_batch_size = 4
                
                    start_time = time.time()
                    test_data["pred"] = sliding_window_inference(test_inputs, patch_size, sw_batch_size, self.model)
                    inference_time = time.time() - start_time
                    test_data = [post_transforms(i) for i in decollate_batch(test_data)]
                
                    self.inference_time_str = self.format_inference_time(inference_time)
                    print(f"Inference time: {self.inference_time_str}")
                    
                    # Calculate SAT and VAT volumes
                    pred_data = test_data[0]["pred"].cpu().numpy()
                    sat_volume = round(((pred_data == 2).sum() * self.nii_obj.header["pixdim"][1:4].prod()) / 1000, 4)  ## mm3 to cm3
                    vat_volume = round(((pred_data == 1).sum() * self.nii_obj.header["pixdim"][1:4].prod()) / 1000, 4)
                    self.lblSAT.setText(f"SAT (cm³)\n{sat_volume:.2f}")
                    self.lblVAT.setText(f"VAT (cm³)\n{vat_volume:.2f}")
                    self.lblVATtoSAT.setText(f"VAT/SAT (cm³)\n{vat_volume/sat_volume:.2f}")
                    
                    self.lblModelPath.setText(f"Inference time: {self.inference_time_str}")
                    self.predicted_mask_path = f"{self.save_path}/{self.nii_path.split('/')[-1].split('.')[0]}_pred.nii"  # Save path for predicted mask
                    self.btnQualityCheck.setEnabled(True)  # Enable the Quality Check button
                    
        except Exception as e:
            print(f"Error during prediction: {e}")
    
    def launch_quality_check(self):
        if hasattr(self, 'nii_path') and hasattr(self, 'predicted_mask_path'):
            subprocess.Popen([sys.executable, "quality_check.py", self.nii_path, self.predicted_mask_path])

if __name__ == "__main__":
    try:
        app = QApplication((sys.argv))
        window = MainWindow()
    
        with open("assets/qss/light3.qss", "r") as file:
            stylesheet = file.read()
            app.setStyleSheet(stylesheet)
            window.show()
        
        sys.exit(app.exec())
    except Exception as e:
        print(f"Error starting application: {e}")
