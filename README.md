# IMU Sensor Data Filtering & Visualization

This project implements a Python-based IMU signal processing pipeline for accelerometer and gyroscope data. It applies multiple filtering methods, evaluates noise reduction performance, computes sensor magnitude metrics, and compares frequency-domain behavior using FFT analysis.

## Features
- CSV-based IMU data input
- Moving Average filtering
- Exponential Moving Average (EMA) filtering
- Butterworth low-pass filtering
- Raw vs filtered signal visualization
- Accelerometer and gyroscope magnitude analysis
- Standard deviation based filter comparison
- FFT-based frequency-domain comparison
- Processed data export to CSV
- Summary report export to text file

## Tools Used
- Python
- Pandas
- NumPy
- Matplotlib
- SciPy

## Project Files
- `main.py` → signal processing and visualization script
- `imu_data.csv` → raw IMU dataset
- `filtered_output.csv` → processed output dataset
- `summary_results.txt` → numerical summary report
- `signal_plot_acc_x.png` → raw vs filtered acc_x signals
- `signal_plot_magnitude.png` → accelerometer magnitude comparison
- `filter_comparison.png` → standard deviation comparison of filters
- `fft_comparison.png` → frequency-domain comparison using FFT
- `requirements.txt` → project dependencies
- `README.md` → documentation

## Method
The project applies three common filtering methods:
- Moving Average
- Exponential Moving Average (EMA)
- Butterworth Low-Pass Filter

Filtering effectiveness is evaluated using:
- standard deviation reduction
- magnitude-based signal interpretation
- FFT-based frequency response comparison

## Engineering Insight
The Butterworth filter generally preserves low-frequency motion trends while attenuating higher-frequency noise components more effectively than basic smoothing approaches. Comparing multiple filters provides a more rigorous signal quality assessment and better reflects real-world IMU preprocessing workflows.

## Output
The script:
- reads raw IMU data from CSV
- applies multiple filtering techniques
- computes raw and filtered sensor magnitudes
- exports the processed dataset
- generates multiple engineering plots
- writes summary metrics to a text report

## How to Run
```bash
python main.py


Author

Burak Bekir ÇAKIRER