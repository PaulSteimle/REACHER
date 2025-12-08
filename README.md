# REACHER - **R**aw image **E**xtraction and **C**alibration or **H**igh-contrast **E**xoplanet **R**eductions

A Python-based pipeline for the pre-processing and analysis of high-contrast imaging (HCI) data currently only applicable to **VLT SPHERE/IRDIS**, designed to facilitate the **detection of circumstellar disks and exoplanets**.
REACHER allows for automated pre-procssing of SPHERE IRDIS datasets collected with and without star-hopping in the DBI mode. It takes users from raw archive data to science-ready results, integrating seamlessly with the [VIP HCI](https://github.com/vortex-exoplanet/VIP) package for advanced post-processing.

## ✨ Features

- Preprocessing of raw SPHERE/IRDIS frames
  - Flat, Dark, Background, Sky and Distortion and Bad-Pixel correction 
  - Frame registration and centering
  - Parallactic angle extraction for each frame 
  - Temporal filtering
    
- Ready postprocessing integration with VIP for:
  - PSF subtraction
  - Circumstellar disk detection
  - Exoplanet detection
  - Exoplanet Astrometry & Photometry using negative injection recovery

## 🧰 Used Software

- Python 3.8
- [VIP HCI](https://github.com/vortex-exoplanet/VIP)
- NumPy
- Astropy
- SciPy
- Matplotlib (optional for plotting)
- [ESO Raw Data Archive](http://archive.eso.org/eso/eso_archive_main.html) (data source)

## 📥 Input Data

The pipeline is designed to work with **raw SPHERE/IRDIS .fits files** from the **ESO Archive**. You'll need to manually download the science and calibration data for your target of interest.

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/PaulSteimle/REACHER.git
cd SIRP
````

### 2. Set Up the Environment

We recommend using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


### 3. Prepare Your Data
Download your data from the [SPHERE ARCHIVE](https://archive.eso.org/wdb/wdb/eso/sphere/form) into a designated folder and use 

```python
import REACHER

rename_dict = {
    'epsSgr' : 'HD169022_epsSgr',
}

download_directory = '/path/to/downloaded/data/'
new_directory = '/path/to/structured/data/'

REACHER.move_files_from_download_dir(download_directory, new_directory, rename_dict)
´´´

This will structure your data according to:

```css
data/
├── science/
│   └── IRDIS/
│       └── obs_night/
│           └── filter/
├── darks/
│   └── IRDIS/
│       └── obs_night/
│           └── filter/
├── flats/
│   └── IRDIS/
│       └── obs_night/
│           └── filter/


```
from where then REACHER will grab the needed files


### 4. Run the Pipeline

Start the reduction process by setting up the main Python file and then run it e.g.:

```bash
python run_sirp.py
```

[!IMPORTANT]  
Note That currently only the processing in survey format is supported. e.g the master files like master-darks are created before touching the data. In future versions REACHER will feature a single target mode where these are created on the go.

Make sure to configure your reduction settings in the provided config.yaml.
📊 Output

- The pipeline will produce:

- Calibrated and aligned science frames

- Diagnostic plots (optional)

Post-processed images using VIP (e.g., PCA residuals, SNR maps)

📄 License

MIT License. See LICENSE for details.
🙌 Acknowledgements

ESO for access to the SPHERE/IRDIS data

VIP HCI for powerful HCI post-processing tools

📬 Contact

For questions or suggestions, feel free to open an issue or contact me via [email](mailto:pasteimle@mpia.de)



