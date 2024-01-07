import numpy as np
import scipy.io.wavfile as wav
import numpy.fft as nff
import matplotlib.pyplot as plt

def read_file(fileName):
    sample_rate, signs = wav.read(fileName)
    signs = signs/2**15
    times = np.arange(len(signs)) /sample_rate
    return  signs,sample_rate,times

def time2freq(sigs, sample_rate):
    freqs = nff.fftfreq(len(sigs),d = 1/sample_rate) #获得傅里叶变换的频率样本
    fft = nff.fft(sigs) #获取震动幅度,根据时间域的数组获得频率域数值，由复数构成
    amps = np.abs(fft) #频率振幅/能量
    return  freqs,fft,amps

def read_signals(fileName):
    sample_rate , sigs = wav.read(fileName) # sample_rate单位为HZ
    # show_signal(sigs,sample_rate)
    sigs = sigs/2**15  #音频文件中的值进过归一化处理，介于 -1-1 之间，这个值才是声音强度值，以16位2进制数存放一个样本值，所以初一2**15 是对样本值线性缩放
    times = np.arange(len(sigs))/sample_rate #得到采样周期，采样频率是一秒钟多少个采样样本，采样周期等于等于采样频率的倒数，采样周期也是两个采样样本之间的时间间隔， np.arange(30)表示30个采样点
    sigs = np.array(sigs)
    return sigs,sample_rate,times


def init_signals():
    plt.gcf().set_facecolor(np.ones(3) * 240 / 255)
    plt.subplot(121)
    plt.title("Audio Signal", fontsize=20)
    plt.xlabel("Time(ms)", fontsize=14)
    plt.ylabel("Signal", fontsize=14)
    plt.tick_params(axis='both', right=True, top=True, labelright=True, labelsize=10)
    plt.grid(True)

def init_amplitudes():
    plt.subplot(122)
    plt.title("Audo Amplitude",fontsize = 20)
    plt.xlabel("Frequency (KHz)",fontsize = 14)
    plt.ylabel("Amplitude",fontsize = 14)
    plt.tick_params(axis='both',right=True,top = True  ,labelright = True,labelsize = 10)
    plt.grid(True)


def _draw_amplitude(freqs,amps):
    amps = amps[freqs>=0]
    freqs = freqs[freqs>=0]/1000 # KHz
    # plt.plot(freqs, amps, label="Amplitude", c="orange")
    plt.semilogy(freqs,amps,label="Amplitude", c="orange" ) #对y轴取对数划图
    plt.plot()

def draw_signals(times,signs):
    times = times * 1000
    plt.plot(times, signs, label="Signal", c="dodgerblue")
    # plt.scatter(times, signs, edgecolors='orangered', label='Samples', facecolor='white', s=80, zorder=1)
    plt.legend()

def show_diagram():
    plt.show()

def main():
    fileName = r"E:\python\study\terana_Learning\AI\data\freq.wav"
    signs, sample_rate, times = read_signals(fileName)
    freqs, fft, amps = time2freq(signs,sample_rate)
    init_signals()
    draw_signals(times,signs)
    init_amplitudes()
    _draw_amplitude(freqs,amps)
    show_diagram()
    # time2freq(signs,sample_rate)
main()