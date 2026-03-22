import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ============================================
# IMU SENSOR DATA FILTERING & VISUALIZATION
# Final Professional Version
# ============================================

INPUT_FILE = "imu_data.csv"
OUTPUT_FILE = "filtered_output.csv"
SUMMARY_FILE = "summary_results.txt"

PLOT_ACC_X = "signal_plot_acc_x.png"
PLOT_MAG = "signal_plot_magnitude.png"
PLOT_COMPARISON = "filter_comparison.png"
PLOT_FFT = "fft_comparison.png"

WINDOW_SIZE = 5
EMA_ALPHA = 0.25
BUTTER_ORDER = 2
CUTOFF_HZ = 2.0  # low-pass cutoff
SAMPLE_RATE = 100.0  # Hz, based on time step of 0.01 s


def moving_average(signal: pd.Series, window: int) -> pd.Series:
    return signal.rolling(window=window, center=True, min_periods=1).mean()


def exponential_moving_average(signal: pd.Series, alpha: float) -> pd.Series:
    return signal.ewm(alpha=alpha, adjust=False).mean()


def butter_lowpass_filter(signal: pd.Series, cutoff: float, fs: float, order: int) -> np.ndarray:
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, signal.to_numpy())


def compute_magnitude(x: pd.Series, y: pd.Series, z: pd.Series) -> pd.Series:
    return np.sqrt(x**2 + y**2 + z**2)


def compute_fft(signal: pd.Series, fs: float):
    signal_array = signal.to_numpy()
    n = len(signal_array)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    fft_vals = np.abs(np.fft.rfft(signal_array))
    return freqs, fft_vals


def dominant_frequency(signal: pd.Series, fs: float) -> float:
    freqs, fft_vals = compute_fft(signal, fs)
    if len(freqs) <= 1:
        return 0.0
    idx = np.argmax(fft_vals[1:]) + 1
    return freqs[idx]


def main():
    # 1) Read data
    df = pd.read_csv(INPUT_FILE)

    required_columns = ["time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing columns in CSV: {missing_columns}")

    # 2) Apply filters
    for axis in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]:
        df[f"{axis}_ma"] = moving_average(df[axis], WINDOW_SIZE)
        df[f"{axis}_ema"] = exponential_moving_average(df[axis], EMA_ALPHA)
        df[f"{axis}_butter"] = butter_lowpass_filter(df[axis], CUTOFF_HZ, SAMPLE_RATE, BUTTER_ORDER)

    # 3) Compute magnitudes
    df["acc_mag_raw"] = compute_magnitude(df["acc_x"], df["acc_y"], df["acc_z"])
    df["acc_mag_ma"] = compute_magnitude(df["acc_x_ma"], df["acc_y_ma"], df["acc_z_ma"])
    df["acc_mag_ema"] = compute_magnitude(df["acc_x_ema"], df["acc_y_ema"], df["acc_z_ema"])
    df["acc_mag_butter"] = compute_magnitude(df["acc_x_butter"], df["acc_y_butter"], df["acc_z_butter"])

    df["gyro_mag_raw"] = compute_magnitude(df["gyro_x"], df["gyro_y"], df["gyro_z"])
    df["gyro_mag_ma"] = compute_magnitude(df["gyro_x_ma"], df["gyro_y_ma"], df["gyro_z_ma"])
    df["gyro_mag_ema"] = compute_magnitude(df["gyro_x_ema"], df["gyro_y_ema"], df["gyro_z_ema"])
    df["gyro_mag_butter"] = compute_magnitude(df["gyro_x_butter"], df["gyro_y_butter"], df["gyro_z_butter"])

    # 4) Statistics
    acc_x_raw_std = df["acc_x"].std()
    acc_x_ma_std = df["acc_x_ma"].std()
    acc_x_ema_std = df["acc_x_ema"].std()
    acc_x_butter_std = df["acc_x_butter"].std()

    ma_noise_reduction = 100 * (acc_x_raw_std - acc_x_ma_std) / acc_x_raw_std if acc_x_raw_std != 0 else 0
    ema_noise_reduction = 100 * (acc_x_raw_std - acc_x_ema_std) / acc_x_raw_std if acc_x_raw_std != 0 else 0
    butter_noise_reduction = 100 * (acc_x_raw_std - acc_x_butter_std) / acc_x_raw_std if acc_x_raw_std != 0 else 0

    raw_dom_freq = dominant_frequency(df["acc_x"], SAMPLE_RATE)
    ma_dom_freq = dominant_frequency(df["acc_x_ma"], SAMPLE_RATE)
    ema_dom_freq = dominant_frequency(df["acc_x_ema"], SAMPLE_RATE)
    butter_dom_freq = dominant_frequency(df["acc_x_butter"], SAMPLE_RATE)

    # 5) Save processed output
    df.to_csv(OUTPUT_FILE, index=False)

    # 6) Plot raw vs filtered signals
    plt.figure(figsize=(10, 6))
    plt.plot(df["time"], df["acc_x"], label="Raw acc_x")
    plt.plot(df["time"], df["acc_x_ma"], label="Moving Average acc_x")
    plt.plot(df["time"], df["acc_x_ema"], label="EMA acc_x")
    plt.plot(df["time"], df["acc_x_butter"], label="Butterworth acc_x")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (g)")
    plt.title("IMU Accelerometer X-Axis Signal Filtering")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_ACC_X, dpi=300)
    plt.show()

    # 7) Plot magnitude comparison
    plt.figure(figsize=(10, 6))
    plt.plot(df["time"], df["acc_mag_raw"], label="Raw Acc Magnitude")
    plt.plot(df["time"], df["acc_mag_ma"], label="MA Acc Magnitude")
    plt.plot(df["time"], df["acc_mag_ema"], label="EMA Acc Magnitude")
    plt.plot(df["time"], df["acc_mag_butter"], label="Butterworth Acc Magnitude")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration Magnitude")
    plt.title("Accelerometer Magnitude Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_MAG, dpi=300)
    plt.show()

    # 8) Filter performance comparison
    plt.figure(figsize=(10, 6))
    methods = ["Raw", "Moving Average", "EMA", "Butterworth"]
    std_values = [acc_x_raw_std, acc_x_ma_std, acc_x_ema_std, acc_x_butter_std]
    plt.bar(methods, std_values)
    plt.xlabel("Method")
    plt.ylabel("Standard Deviation")
    plt.title("Filter Performance Comparison for acc_x")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(PLOT_COMPARISON, dpi=300)
    plt.show()

    # 9) FFT comparison
    freqs_raw, fft_raw = compute_fft(df["acc_x"], SAMPLE_RATE)
    freqs_butter, fft_butter = compute_fft(df["acc_x_butter"], SAMPLE_RATE)

    plt.figure(figsize=(10, 6))
    plt.plot(freqs_raw, fft_raw, label="Raw acc_x FFT")
    plt.plot(freqs_butter, fft_butter, label="Butterworth acc_x FFT")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("FFT Comparison: Raw vs Butterworth Filtered acc_x")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_FFT, dpi=300)
    plt.show()

    # 10) Summary report
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write("IMU Sensor Data Filtering & Visualization - Final Summary\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Input file: {INPUT_FILE}\n")
        f.write(f"Output file: {OUTPUT_FILE}\n")
        f.write(f"Moving average window size: {WINDOW_SIZE}\n")
        f.write(f"EMA alpha: {EMA_ALPHA}\n")
        f.write(f"Butterworth order: {BUTTER_ORDER}\n")
        f.write(f"Butterworth cutoff frequency: {CUTOFF_HZ} Hz\n")
        f.write(f"Sampling rate: {SAMPLE_RATE} Hz\n\n")

        f.write("acc_x Standard Deviation Comparison\n")
        f.write("-" * 45 + "\n")
        f.write(f"Raw acc_x std           : {acc_x_raw_std:.6f}\n")
        f.write(f"Moving Average acc_x std: {acc_x_ma_std:.6f}\n")
        f.write(f"EMA acc_x std           : {acc_x_ema_std:.6f}\n")
        f.write(f"Butterworth acc_x std   : {acc_x_butter_std:.6f}\n\n")

        f.write("Estimated Noise Reduction\n")
        f.write("-" * 45 + "\n")
        f.write(f"Moving Average: {ma_noise_reduction:.2f}%\n")
        f.write(f"EMA           : {ema_noise_reduction:.2f}%\n")
        f.write(f"Butterworth   : {butter_noise_reduction:.2f}%\n\n")

        f.write("Dominant Frequency Comparison (acc_x)\n")
        f.write("-" * 45 + "\n")
        f.write(f"Raw         : {raw_dom_freq:.4f} Hz\n")
        f.write(f"Moving Avg  : {ma_dom_freq:.4f} Hz\n")
        f.write(f"EMA         : {ema_dom_freq:.4f} Hz\n")
        f.write(f"Butterworth : {butter_dom_freq:.4f} Hz\n\n")

        f.write("Accelerometer Magnitude Statistics\n")
        f.write("-" * 45 + "\n")
        f.write(f"Raw magnitude mean        : {df['acc_mag_raw'].mean():.6f}\n")
        f.write(f"Moving Average mean       : {df['acc_mag_ma'].mean():.6f}\n")
        f.write(f"EMA mean                  : {df['acc_mag_ema'].mean():.6f}\n")
        f.write(f"Butterworth mean          : {df['acc_mag_butter'].mean():.6f}\n\n")

        f.write("Gyroscope Magnitude Statistics\n")
        f.write("-" * 45 + "\n")
        f.write(f"Raw magnitude mean        : {df['gyro_mag_raw'].mean():.6f}\n")
        f.write(f"Moving Average mean       : {df['gyro_mag_ma'].mean():.6f}\n")
        f.write(f"EMA mean                  : {df['gyro_mag_ema'].mean():.6f}\n")
        f.write(f"Butterworth mean          : {df['gyro_mag_butter'].mean():.6f}\n")

    # 11) Console output
    print("IMU Data Filtering Project Results")
    print("=" * 45)
    print(f"Raw acc_x std           : {acc_x_raw_std:.6f}")
    print(f"Moving Average acc_x std: {acc_x_ma_std:.6f}")
    print(f"EMA acc_x std           : {acc_x_ema_std:.6f}")
    print(f"Butterworth acc_x std   : {acc_x_butter_std:.6f}")
    print()
    print(f"Moving Average noise reduction: {ma_noise_reduction:.2f}%")
    print(f"EMA noise reduction          : {ema_noise_reduction:.2f}%")
    print(f"Butterworth noise reduction  : {butter_noise_reduction:.2f}%")
    print()
    print("Files generated successfully:")
    print(f"- {OUTPUT_FILE}")
    print(f"- {SUMMARY_FILE}")
    print(f"- {PLOT_ACC_X}")
    print(f"- {PLOT_MAG}")
    print(f"- {PLOT_COMPARISON}")
    print(f"- {PLOT_FFT}")


if __name__ == "__main__":
    main()