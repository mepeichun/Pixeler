from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

# setting
OUTPUT_SIZE = 50
COLOR_NUM = 16
ADD_COORD = True  # 是否考虑像素的坐标位置
lambda_ = 0.15  # 0~1之间取值，坐标的重要性


image = Image.open("./images/usagi.png")  # 替换为你的图片路径
if image.mode == 'RGBA':
    image = image.convert('RGB')
resized = image.resize((OUTPUT_SIZE*2, OUTPUT_SIZE*2), Image.BILINEAR)

# 2. 创建像素化图像
pixelated = Image.new("RGB", (OUTPUT_SIZE, OUTPUT_SIZE))
pixel_array = np.array(pixelated)

# 3. 将图像分割为2x2块，计算平均颜色
resized_array = np.array(resized)
for y in range(OUTPUT_SIZE):
    for x in range(OUTPUT_SIZE):
        # 提取2x2像素块
        block = resized_array[y*2:y*2+2, x*2:x*2+2]
        # 计算RGB平均值
        avg_color = np.mean(block, axis=(0, 1)).astype(np.uint8)
        pixel_array[y, x] = avg_color

# 4. 准备KMeans聚类数据
pixels_rgb = pixel_array.reshape(OUTPUT_SIZE*OUTPUT_SIZE, 3) / 255.0  # 归一化到[0,1]
height, width = pixel_array.shape[:2]

# 创建归一化的坐标网格
x_coords = np.tile(np.linspace(0, 1, width), height)
y_coords = np.repeat(np.linspace(0, 1, height), width)
coords = np.column_stack((x_coords, y_coords)) * lambda_

# 合并RGB和坐标信息
pixels_with_coords = np.hstack((pixels_rgb, coords))

# 5. 执行KMeans聚类
kmeans = KMeans(n_clusters=COLOR_NUM, random_state=0, n_init=COLOR_NUM)
if ADD_COORD:
    kmeans.fit(pixels_with_coords)
else:
    kmeans.fit(pixels_rgb)

# 6. 替换为聚类中心颜色
new_colors = kmeans.cluster_centers_[kmeans.labels_, :3]
new_colors = (new_colors * 255).astype(np.uint8)  # 转回0-255范围

# 7. 创建结果图像
result_array = new_colors.reshape(OUTPUT_SIZE, OUTPUT_SIZE, 3)
result_image = Image.fromarray(result_array)

# 保存并显示结果
output_path = "./output/pixelated_reduced.png"
result_image.save(output_path)
print(f"[OK] 已导出像素图: {output_path}")
