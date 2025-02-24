import sounddevice as sd
import numpy as np
import time
from scipy.signal import find_peaks

# 采样参数
SAMPLE_RATE = 44100  # 采样率 (Hz)
DURATION = 0.04 # 每次采样持续时间 (秒)
DEVICE_INDEX = 0  # 设备索引（默认 0，如果不工作请运行 print(sd.query_devices()) 检查设备索引）

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

def callback(indata, frames, time_info, status):
    """实时处理音频输入"""
    if status:
        print(f"Error: {status}")
        return

    audio_data = indata[:, 0]  # 获取单声道数据
    max_freq = detect_peak_frequency(audio_data, SAMPLE_RATE)
    try:
        if 3300>max_freq>3000:
            print(f"当前最高峰频率: {max_freq:.2f} Hz")  # 实时输出频率
    except Exception as E:
        pass
    #else:
    #    print("未检测到峰值频率")

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
