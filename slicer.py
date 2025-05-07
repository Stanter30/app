import os
from PIL import Image

# Параметры
input_tif = r"1.tif"  # Путь к вашему TIF изображению
img = Image.open(input_tif).convert('RGB')

output_folder = r"Sliced_images/"  # Папка для сохранения разрезанных изображений
tile_size = 512  # Размер тайла (размер каждой части)

# Создаем папку для выходных изображений, если она не существует
#os.makedirs(output_folder, exist_ok=True)

# Открываем изображение
with Image.open(input_tif) as img:
    width, height = img.size
    # Итерируем по окнам изображения
    for i in range(0, height, tile_size):
        for j in range(0, width, tile_size):
            # Определяем координаты окна
            box = (j, i, min(j + tile_size, width), min(i + tile_size, height))
            # Обрезаем изображение
            tile = img.crop(box)
            # Определяем имя выходного файла
            output_tif = os.path.join(output_folder, f'tile_{i}_{j}.tif')
            # Сохраняем обрезанное изображение
            tile.save(output_tif, format='TIFF')

print("Разрезка завершена.")