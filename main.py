import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import cv2
import numpy as np

#Configuración del ImageSegmenter
options = vision.ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path="./Models/deeplab_v3.tflite"), output_category_mask=True, running_mode=vision.RunningMode.IMAGE)
segmenter = vision.ImageSegmenter.create_from_options(options)

#Categorias
categories={0: 'background',
            1: 'aeroplane',
            2: 'bicycle',
            3: 'bird',
            4: 'boat',
            5: 'bottle',
            6: 'bus',
            7: 'car',
            8: 'cat',
            9: 'chair',
            10: 'cow',
            11: 'dining table',
            12: 'dog',
            13: 'horse',
            14: 'motorbike',
            15: 'person',
            16: 'potted plant',
            17: 'sheep',
            18: 'sofa',
            19: 'train',
            20: 'tv/monitor'
            }

#Asignas colores únicos a las categorias
category_colors = np.random.randint(0, 255, size=(len(categories),3), dtype="uint8")
#print(category_colors)

#Aplicando el modelo sobre una imagen
#Leer imagen entrada
image=cv2.imread("./Images/Televisor.jpg")

# Verificar si la imagen fue cargada correctamente
if image is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta.")
    exit()

#Transformas el category mask a 3 canales


#Convertir la imagen a RGB
image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image_rgc=mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

#Obtener los resultados del segmentador
segmetation_result = segmenter.segment(image_rgc)
#print(segmetation_result)

#Convertir la máscara de categorias en un array de Numpy
category_mask=segmetation_result.category_mask
category_mask_np=category_mask.numpy_view()
print(np.unique(category_mask_np))

#Transformar el category mask a 3 canales
category_mask_bgr=cv2.cvtColor(category_mask_np,cv2.COLOR_GRAY2BGR)
#print(category_mask_bgr.shape)

# Colorear cada segmento con su color correspondiente
for category_id in np.unique(category_mask_np):
    # Verificar si el category_id está dentro de los índices de category_colors
    if category_id < len(category_colors):
        color = category_colors[category_id]
    else:
        color = (128, 128, 128)  # Color gris por defecto para categorías no definidas
    category_mask_bgr[np.where(category_mask_np == category_id)] = color

#Transparencia
alpha=0.5
final_image=cv2.addWeighted(image, 1 - alpha, category_mask_bgr, alpha,0)

#Visualizar lista categorias
black_image = np.zeros((430, 200, 3), dtype="uint8")

y_offset=20
font_scale=0.6
line_thickness=2

for category_id, name in categories.items():
    print(category_id, name)
    if category_id in np.unique(category_mask_np):
        color=tuple(map(int, category_colors[category_id]))
    else:
        color=(128, 128,128)
    cv2.putText(black_image,
                f"{category_id}:{name}",
                (30, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                line_thickness,
                cv2.LINE_AA)
    y_offset+=20

#Visualización
#cv2.imshow("Image", image)
#cv2.imshow("category_mask_np", category_mask_np)
#cv2.imshow("category_mask_bgr", category_mask_bgr)
cv2.imshow("final_image", final_image)
cv2.imshow("black_image", black_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



