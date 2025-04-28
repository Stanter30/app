import numpy as np
import rasterio
from dbscan import DBSCAN
#import geopandas as gpd
#from shapely.geometry import Polygon
#from scipy.ndimage import label


with rasterio.open("C:/Users/tereshkinsa/Desktop/test.tif") as src:
    image = src.read()  # Чтение всех каналов
    height, width = image.shape[1], image.shape[2]

x = np.arange(0, height, dtype='uint32')
y = np.arange(0, width, dtype='uint32')

mW, nW = np.meshgrid(x, y)

X = np.reshape(mW, mW.size) * 10
Y = np.reshape(nW, nW.size) * 10

R = image[0]
G = image[1]
B = image[2]

RF = R.flatten()
GF = G.flatten()
BF = B.flatten()


data = np.stack((X, Y, RF), axis=1, dtype='int32')

CL, mask_1 = DBSCAN(data, eps=10, min_samples=5)


CL_data = CL.reshape((height, width))



# Преобразование меток в 2D массив
labelled_array, num_features = label(labels.reshape((height, width)))

polygons = []
for i in range(num_features):
    # Получите координаты каждого кластера
    coords = np.column_stack(np.where(labelled_array == i))
    if len(coords) > 0:
        # Создайте полигон из координат
        poly = Polygon(coords)
        polygons.append(poly)

gdf = gpd.GeoDataFrame({'geometry': polygons})
gdf.to_file("C:/Users/tereshkinsa/Desktop/gdf/test.shp", driver='ESRI Shapefile')