import sounddevice as sd
import numpy as np
import time
import csv
from scipy.signal import find_peaks

# 采样参数
SAMPLE_RATE = 44100  # 采样率 (Hz)
DURATION = 0.04  # 每次采样持续时间 (秒)
DEVICE_INDEX = 0  # 设备索引（默认 0）
LOG_FILE = "frequency_log.csv"  # 记录文件

# 记录最近一次检测的时间
last_detection_time = 0
min_time_between_detections = 1  # 1 秒内的多次检测视为同一次
long_interval_threshold = 30  # 记录必须间隔 30 秒以上

def detect_peak_frequency(audio_data, rate):
    """计算音频数据的FFT并返回最高峰值的频率"""
    fft_data = np.fft.rfft(audio_data)
    freqs = np.fft.rfftfreq(len(audio_data), d=1.0 / rate)
    magnitude = np.abs(fft_data)

    # 找到峰值
    peaks, _ = find_peaks(magnitude, height=0, distance=5)  # 设定阈值，防止误检
    if len(peaks) == 0:
        return None

    # 获取最高峰值对应的频率
    peak_freqs = freqs[peaks]
    peak_mags = magnitude[peaks]
    max_peak_index = np.argmax(peak_mags)
    max_freq = peak_freqs[max_peak_index]

    return max_freq

def calculate_decibels(audio_data):
    """计算音频信号的分贝（dB SPL），使用标准 20μPa 参考值"""
    ref_pressure = 2e-5  # 20μPa（人耳听力阈值）
    rms = np.sqrt(np.mean(audio_data**2))  # 计算 RMS（均方根）

    if rms > 0:
        db = 20 * np.log10(rms / ref_pressure)
    else:
        db = -100  # 代表完全静音

    return db

def log_detection(time_diff, frequency, decibels):
    """记录检测到的频率间隔、分贝大小"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([str(time.time()), timestamp, time_diff, frequency, decibels])
    print(f"🔥 记录：{timestamp} | 上次间隔: {time_diff:.2f} 秒 | 频率: {frequency:.2f} Hz | 分贝: {decibels:.2f} dB")

def callback(indata, frames, time_info, status):
    """实时处理音频输入"""
    global last_detection_time
    if status:
        print(f"Error: {status}")
        return

    audio_data = indata[:, 0]  # 获取单声道数据
    max_freq = detect_peak_frequency(audio_data, SAMPLE_RATE)
    decibels = calculate_decibels(audio_data)  # 计算当前分贝大小

    if max_freq and 3000 < max_freq < 3300:  # 只记录 3000~3300 Hz
        current_time = time.time()
        time_diff = current_time - last_detection_time

        # 1 秒内的多次检测视为同一次
        if time_diff < min_time_between_detections:
            return

        # 30 秒内不重复记录
        if time_diff >= long_interval_threshold:
            log_detection(time_diff, max_freq, decibels)

        last_detection_time = current_time  # 更新上次检测时间

def main():
    """主程序，持续监听麦克风"""
    print("检测可用音频输入设备...")
    print(sd.query_devices())  # 打印所有可用音频设备

    print(f"使用设备索引: {DEVICE_INDEX}")
    print("开始监听麦克风... 按 Ctrl+C 停止")

    try:
        with sd.InputStream(callback=callback, device=DEVICE_INDEX, channels=1, samplerate=SAMPLE_RATE, blocksize=int(SAMPLE_RATE * DURATION)):
            print("音频流已启动")
            while True:
                time.sleep(1)  # 让程序持续运行
    except KeyboardInterrupt:
        print("\n停止监听")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()
