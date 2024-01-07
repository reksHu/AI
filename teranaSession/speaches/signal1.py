import numpy as np
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt
import pdb
#sigs 音频文件的强度值
def show_signal(sigs,sample_rage):
    print(sigs)  #(132300,) 表示一共是132300个样本
    print(sample_rage/1000) #打印出音频的千赫兹,表示每秒钟有多少个强度的样本值，叫采样频率。 44.1KHZ 表示对sigs个样本每秒中采的样本数为44.1K个
    print(len(sigs) *1000 / sample_rage) #音频文件持续时间 毫秒， 3000 ms表示视频长度为3s

#获得信号，采样率，和时间点
#采样率：sample_rate，信号强度：sigs，周期/时间点：times
def read_signals(fileName):
    sample_rate , sigs = wf.read(fileName) # sample_rate单位为HZ
    # show_signal(sigs,sample_rate)
    sigs = sigs[:30]/2**15  #音频文件中的值进过归一化处理，介于 -1-1 之间，这个值才是声音强度值，以16位2进制数存放一个样本值，所以初一2**15 是对样本值线性缩放
    times = np.arange(30)/sample_rate #得到采样周期，采样频率是一秒钟多少个采样样本，采样周期等于等于采样频率的倒数，采样周期也是两个采样样本之间的时间间隔， np.arange(30)表示30个采样点
    print(sigs)
    print(times)
    sigs = np.array(sigs)
    return sigs,sample_rate,times

def init_chart():
    plt.gcf().set_facecolor(np.ones(3)*240/255)
    plt.title("Audio Signal",fontsize = 20)
    plt.xlabel("Time(ms)",fontsize = 14)
    plt.ylabel("Signal",fontsize = 14)
    plt.tick_params(axis='both',right=True,top = True, labelright = True,labelsize = 10)
    plt.grid(linestyle=":")

def draw_chart(times,signs):
    times *=1000
    plt.plot(times,signs,label = "Signal",c = "dodgerblue",zorder = 0)
    plt.scatter(times,signs,edgecolors='orangered',label ='Samples',facecolor = 'white',s = 80,zorder = 1)
    plt.legend()

def show_chart():
    plt.show()

def main():
    fileName = r"E:\python\study\terana_Learning\AI\data\signal.wav"
    signs, sample_rage,times = read_signals(fileName)
    # init_chart()
    # draw_chart(times,signs)
    # show_chart()

main()