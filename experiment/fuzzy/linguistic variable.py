import numpy as np
import matplotlib.pyplot as plt

# 定義模糊隸屬函數
def slow(x):
    if x < 35:
        return 1
    elif 35 <= x <= 55:
        return (55 - x) / 20
    else:
        return 0

def medium(x):
    if 35 <= x <= 55:
        return (x - 35) / 20
    elif 55 <= x <= 75:
        return (75 - x) / 20
    else:
        return 0

def fast(x):
    if 55 <= x <= 75:
        return (x - 55) / 20
    elif x > 75:
        return 1
    else:
        return 0

# 準備數據
x = np.linspace(0, 100, 1000)
slow_y = [slow(xi) for xi in x]
medium_y = [medium(xi) for xi in x]
fast_y = [fast(xi) for xi in x]

# 繪製圖形
plt.figure(figsize=(8, 6))
plt.plot(x, slow_y, label="slow", color="red")
plt.plot(x, medium_y, label="medium", color="blue")
plt.plot(x, fast_y, label="fast", color="green")

# 添加標籤和圖例
plt.title("Fuzzy Membership Functions for Car Speed")
plt.xlabel("Car Speed (km/hr)")
plt.ylabel("Membership Degree")
plt.legend()

plt.xticks(np.arange(0, 101, 10))
plt.grid(False)

# 顯示圖形
plt.show()
