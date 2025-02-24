import numpy as np
import sounddevice as sd

fs = 44100  # 采样率
duration = 30  # 持续时间（秒）
freq = 3160  # 目标频率

while 114:
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    white_noise = np.random.normal(0, 1, t.shape)  # 生成白噪音
    narrow_band_noise = np.sin(2 * np.pi * freq * t) + white_noise * 0.2  # 叠加 3160Hz

    sd.play(narrow_band_noise, samplerate=fs)  # 播放
    sd.wait()

