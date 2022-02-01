import matplotlib.pyplot as plt
# c = ['#192C3C', '#274862', '#995054', '#D96831', '#E6B33D','#B2BE80','#593D43']
c = ['#823935', '#89BEB2', '#C9BA83', '#DED38C', '#DE9C53','#B2BE80','#593D43']
# c = ['#E3A05D', '#B2BE80', '#726F80', '#380D31', '#593D43','#B2BE80', '#593D43']
# c = ['#0f98CB', '#995054', '#00B1B7', '#FFBAD1', '#FFD040', '#B2BE80', '#595959']


x = [6, 24, 48, 72]
y1 = [87, 174, 225, 254]
y2 = [24, 97, 202, 225]
y3 = [110, 138, 177, 205]
y4 = [95, 68, 83, 105]
y5 = [72, 74, 76, 67]
y6 = [93, 65, 63, 235]
y7 = [34, 54, 75, 145]
plt.title('扩散速度')  # 折线图标题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
plt.xlabel('时间')  # x轴标题
plt.ylabel('差值')  # y轴标题
plt.plot(x, y1, marker='o', markersize=3, color=c[0])  # 绘制折线图，添加数据点，设置点的大小
plt.plot(x, y2, marker='o', markersize=3, color=c[1])
plt.plot(x, y3, marker='o', markersize=3, color=c[2])
plt.plot(x, y4, marker='o', markersize=3, color=c[3])
plt.plot(x, y5, marker='o', markersize=3, color=c[4])
plt.plot(x, y6, marker='o', markersize=3, color=c[5])
plt.plot(x, y7, marker='o', markersize=3, color=c[6])
for a, b in zip(x, y1):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)  # 设置数据标签位置及大小
for a, b in zip(x, y2):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
for a, b in zip(x, y3):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
for a, b in zip(x, y4):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
for a, b in zip(x, y5):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
for a, b in zip(x, y6):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
for a, b in zip(x, y7):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

plt.legend(['1', '2', '3', '4', '5', '6', '7'])  # 设置折线名称

plt.show()  # 显示折线图


plt.show()
