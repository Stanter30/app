import os
from PIL import Image

input_folder = r"Input_images/"
output_folder = r"Sliced_images/"  # Папка для сохранения разрезанных изображений
tile_size = 256  # Размер тайла (размер каждой части)


files = os.listdir(input_folder)

cnt = 0
for filename in files:
    input_tif = os.path.join(input_folder, filename)
    # Открываем изображение
    with Image.open(input_tif).convert('RGB') as img:
        width, height = img.size
        # Итерируем по окнам изображения
        for i in range(0, height, tile_size):
            for j in range(0, width, tile_size):
                # Определяем координаты окна
                box = (j, i, min(j + tile_size, width), min(i + tile_size, height))
                # Обрезаем изображение
                tile = img.crop(box)
                # Определяем имя выходного файла
                cnt += 1
                output_tif = os.path.join(output_folder, f'{cnt}.tif')
                # Сохраняем обрезанное изображение
                tile.save(output_tif, format='TIFF')

print("Разрезка завершена.")