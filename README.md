# PPG/PPW Signal Extraction and Blood Pressure Estimation Project

This project provides a comprehensive framework for the extraction, preprocessing, analysis, and estimation of blood pressure using signals derived from video data. Leveraging computer vision, signal processing, and statistical analysis techniques, the project extracts Photoplethysmography (PPG) and Pulse Pressure Wave (PPW) signals from the face (forehead) and neck, computes the Blood Volume Pulse (BVP) signal, and estimates hemodynamic parameters such as Systolic Blood Pressure (SBP) and Diastolic Blood Pressure (DBP) using multiple methods (e.g., direct methods, cross-correlation, and AIx-based approaches).

---

## Key Features

- **Video Signal Extraction:**  
  - Utilizes MediaPipe Face Mesh for detecting facial landmarks, especially for the forehead.
  - Enables manual selection and adaptive tracking of the Region of Interest (ROI) for the neck.

- **Video Management:**  
  - Calculation of video sampling rate (both nominal and actual FPS).
  - Reading and management of external data files (e.g., blood pressure data).

- **PPG and PPW Signal Extraction:**  
  - Extracts the RGB signal from both the forehead (using MediaPipe) and the neck.
  - Computes the PPW signal from the neck using optical flow techniques (Farneback method).

- **Preprocessing and Filtering:**  
  - Removes trends using smoothness priors detrending.
  - Applies bandpass and lowpass Butterworth filters to isolate the relevant frequency band (typically 0.5–3.0 Hz).

- **BVP Extraction and Peak Detection:**  
  - Computes the BVP signal via the CHROM method, which combines the R, G, and B channels.
  - Detects peaks using dedicated functions that implement IQR filtering and adjustments based on heart rate.

- **Pulse Transit Time (PTT) Calculation:**  
  - Implements both cross-correlation and direct (with quadratic interpolation) methods to estimate PTT.
  - Validates PTT values against physiologically plausible ranges.

- **Blood Pressure Estimation:**  
  - Estimates SBP and DBP using linear relationships based on PTT values and AIx (Augmentation Index) methods.
  - Saves results and generates visualizations (e.g., Bland-Altman plots, pair plots, box plots, scatter plots, and correlation heatmaps).

- **Interpolation and Method Comparison:**  
  - Provides interpolation using polynomial resampling and cubic spline methods to adjust signals to a new sampling rate (e.g., 100 FPS).
  - Compares and correlates PTT values from different calculation methods and blood pressure estimates from various approaches.

- **Statistical Analysis:**  
  - Computes Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Pearson correlation.
  - Generates statistical plots for assessing the performance of the models.

---

## Project Structure

The code is organized into functional modules covering the following areas:

- **Library Imports:**  
  Initialization of standard, scientific (numpy, pandas, scipy), computer vision (OpenCV, MediaPipe), and data visualization (matplotlib, seaborn) libraries.

- **Video Management & ROI Handling:**  
  Functions to extract video frame rate, select the ROI (both manual and via tracking), and save ROI data.

- **Signal Extraction:**  
  - Functions to extract the RGB signals from the forehead (using MediaPipe Face Mesh) and neck.
  - Extraction of the PPW signal from the neck using optical flow calculation.

- **Preprocessing & Filtering:**  
  Functions for detrending and applying filters (bandpass, lowpass) to enhance signal quality.

- **BVP Calculation & Peak Detection:**  
  Utilizes the CHROM method to compute the BVP from the combined RGB channels and includes functions for peak detection using adaptive IQR filters and temporal validations.

- **PTT and AIx Calculation:**  
  Functions implementing PTT estimation via cross-correlation and direct methods (with quadratic interpolation) along with validation. AIx values are computed for indirect blood pressure estimation.

- **Blood Pressure Estimation & Statistical Analysis:**  
  Dedicated functions to estimate SBP and DBP, generate visualizations (Bland-Altman, scatter plots, pair plots, box plots, heatmaps), and conduct statistical analyses (MAE, RMSE, Pearson correlation).

- **Main Execution Interface:**  
  The `main()` function coordinates the entire workflow—from setting paths and configurations, extracting and preprocessing signals, to final analysis and visualization. All outputs (text files, CSVs, and images) are saved in the designated output directory.

---

## Requirements and Dependencies

This project is written in Python and requires the following libraries:

- **Standard Libraries:** `os`, `json`, `time`
- **Scientific Computing:** `numpy`, `pandas`, `scipy`
- **Computer Vision:** `opencv-python` (cv2), `mediapipe`
- **Data Visualization:** `matplotlib`, `seaborn`

Ensure you have installed the required libraries, for example by running:

```bash
pip install numpy pandas scipy opencv-python mediapipe matplotlib seaborn
```

# Usage Guide

1. **Configuration:**  
   - Set the initial configuration to define dataset paths, the subject identifier, and the task (e.g., `M005/T1`).
   - Specify the gender parameter (`male` or `female`) as required by the blood pressure estimation formulas.

2. **Running the Main Script:**  
   - The project entry point is the `main()` function.
   - Run the project by executing the Python file (e.g., `python main.py`).
   - The script initializes paths, reads video and blood pressure data, extracts and tracks the ROI, processes the signals, performs analyses, and saves all results in the designated output folder.

3. **Output and Results:**  
   - All outputs, including text files, CSV files, and graphs (PNG/JPG formats), are saved in a subject/task-specific output directory.
   - Generated plots include overlays for peak detection, Bland-Altman plots, scatter plots, box plots, and heatmaps.
   - CSV files contain estimation results and statistical analyses for further review.

4. **Customization:**  
   - The interpolation functions allow you to adjust the sampling rate (e.g., to 100 FPS) to enhance signal resolution.
   - Filter parameters (cutoff frequencies, filter order) and peak detection settings can be modified to better suit your dataset.
   - The modular structure of the code allows easy updates or replacement of preprocessing, filtering, and analysis functions without altering the overall workflow.

# Technical Notes

- **ROI Tracking:**  
  Adaptive ROI tracking for the neck is implemented using OpenCV’s CSRT tracker. If a file with ROI dimensions exists, it is loaded automatically; otherwise, manual ROI selection is prompted.

- **Signal Extraction Methods:**  
  Two methods for calculating Pulse Transit Time (PTT) are provided:
  - **Cross-Correlation Method:** Computes the optimal lag between the forehead and neck signals using cross-correlation.
  - **Direct Method:** Measures the time difference between the detected peaks with quadratic interpolation for enhanced sub-sample accuracy.

- **Statistical Analysis:**  
  - Linear regression, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) are calculated to evaluate the accuracy of blood pressure estimations.
  - Pearson correlation is used to assess the relationship between estimated and reference measurements.
  - Bland-Altman plots are generated to visually assess agreement between the estimated and actual blood pressure values.

- **Modularity:**  
  The code is designed in a modular way, enabling independent updates or replacements of components (e.g., signal preprocessing, filtering, interpolation, and analysis) without impacting the overall processing pipeline.

# Conclusion

This project offers a comprehensive and flexible solution for analyzing video-derived PPG/PPW signals and estimating blood pressure. It integrates advanced techniques from computer vision, signal processing, and statistical analysis, making it a valuable tool for research and development in medical and physiological applications. The modular design and extensive visualization capabilities ensure that the system can be easily adapted and extended to meet specific research needs. For further inquiries or contributions, please contact the development team or refer to the internal documentation.