# SIRP - VLT SPHERE/IRDIS Reduction Pipeline

A Python-based pipeline for the preprocessing and analysis of high-contrast imaging (HCI) data from **VLT SPHERE/IRDIS**, designed to facilitate the **detection of circumstellar disks and exoplanets**.
This pipeline takes users from raw archive data to science-ready results, integrating seamlessly with the [VIP HCI](https://github.com/vortex-exoplanet/VIP) package for advanced post-processing.

## ✨ Features

- Preprocessing of raw SPHERE/IRDIS frames (flat, dark, background subtraction, etc.)
- Frame registration and centering
- Temporal filtering and derotation
- Ready integration with VIP for:
  - PSF subtraction
  - Disk enhancement
  - Exoplanet detection

## 🧰 Technologies

- Python 3.8
- [VIP HCI](https://github.com/vortex-exoplanet/VIP)
- NumPy
- Astropy
- SciPy
- Matplotlib (optional for plotting)
- [ESO Raw Data Archive](http://archive.eso.org/eso/eso_archive_main.html) (data source)

## 📥 Input Data

The pipeline is designed to work with **raw SPHERE/IRDIS FITS files** from the **ESO Archive**. You'll need to manually download the science and calibration data for your target of interest.

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/hci-reduction-pipeline.git
cd hci-reduction-pipeline
````

### 2. Set Up the Environment

We recommend using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


### 3. Prepare Your Data

Place your raw science and calibration files in the data/ directory. The folder structure should follow:


data/
├── science/
├── darks/
├── flats/
└── backgrounds/



### 4. Run the Pipeline

Start the reduction process with:

```python
python run_pipeline.py --config config.yaml
```

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

For questions or suggestions, feel free to open an issue or contact me via email at your.email@example.com.



