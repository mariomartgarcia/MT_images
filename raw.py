from PIL import Image
import matplotlib.pyplot as plt

# Open the PNG file
image = Image.open('v_2/split/train/grassland_s2/ROIs1970_fall_s2_145_p1034.png')

# Get the number of bands (channels)
bands = image.getbands()

print(f"Number of bands: {len(bands)}")
print(f"Bands: {bands}")


plt.imshow(image)
plt.axis('off')  # Ocultar los ejes
plt.show()



import os
from PIL import Image

def process_images(base_dir):
    # Subcarpetas a procesar
    activities = ["YesActivity", "NoActivity"]
    for activity in activities:
        for folder in ["train", "test"]:
            input_dir = os.path.join(base_dir, folder, activity)
            rgb_dir = os.path.join(base_dir, folder, f"{activity}RGB")
            alpha_dir = os.path.join(base_dir, folder, f"{activity}A")

            # Crear carpetas de salida si no existen
            os.makedirs(rgb_dir, exist_ok=True)
            os.makedirs(alpha_dir, exist_ok=True)

            # Procesar cada imagen
            for filename in os.listdir(input_dir):
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    input_path = os.path.join(input_dir, filename)
                    
                    # Abrir la imagen
                    with Image.open(input_path) as img:
                        # Verificar si tiene banda alfa
                        if img.mode == 'RGBA':
                            r, g, b, a = img.split()
                            
                            # Crear imagen RGB
                            rgb_image = Image.merge("RGB", (r, g, b))
                            rgb_image.save(os.path.join(rgb_dir, filename))
                            
                            # Guardar capa A
                            a.save(os.path.join(alpha_dir, filename))
                        else:
                            print(f"Image {filename} in {input_dir} does not have an alpha channel.")

if __name__ == "__main__":
    # Ruta base
    base_directory = "LastUpdate"
    process_images(base_directory)



import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_image_volcano(data_dir, classes, input_size=(224, 224)):
    rgb_paths = []
    alpha_paths = []
    labels = []  # Guardar etiquetas: 0 para 'NoActivity', 1 para 'YesActivity'

    for label, class_name in enumerate(classes):
        rgb_folder = os.path.join(data_dir, f'{class_name}RGB')
        alpha_folder = os.path.join(data_dir, f'{class_name}A')

        for file_name in os.listdir(rgb_folder):
            rgb_path = os.path.join(rgb_folder, file_name)
            alpha_path = os.path.join(alpha_folder, file_name)

            if os.path.exists(rgb_path) and os.path.exists(alpha_path):
                rgb_paths.append(rgb_path)
                alpha_paths.append(alpha_path)
                labels.append(label)  # Etiqueta basada en el índice de `classes`
            else:
                print('Warning: mismatch')
                print(f'Missing file: {rgb_path}' if not os.path.exists(rgb_path) else f'Missing file: {alpha_path}')

    # Preprocesar imágenes
    def preprocess_image(image_path, target_size=input_size):
        img = load_img(image_path, target_size=target_size)
        return img_to_array(img) / 255.0  # Normalizar a [0, 1]

    rgb_images = np.array([preprocess_image(path) for path in rgb_paths])
    alpha_images = np.array([preprocess_image(path)[:, :, 0:1] for path in alpha_paths])  # Grayscale alpha channel
    labels = np.array(labels)  # Convertir etiquetas a numpy array

    return rgb_images, alpha_images, labels



import os
import shutil
import random

# Directorio raíz
root_dir = 'v_2'
# Directorio de destino para la división
split_dir = os.path.join(root_dir, 'split')

# Clases y subcarpetas (s1, s2)
classes = ['agri', 'barrenland', 'grassland', 'urban']
subfolders = ['s1', 's2']

# Crear la estructura de carpetas (train y test para cada clase y s1/s2)
for split in ['train', 'test']:
    for cls in classes:
        for sub in subfolders:
            path = os.path.join(split_dir, split, f"{cls}_{sub}")
            os.makedirs(path, exist_ok=True)

# Función para dividir y mover las imágenes
def split_images():
    # Recorre cada clase
    for cls in classes:
        # Listas para almacenar las imágenes de cada subcarpeta
        s1_images = []
        s2_images = []
        
        # Recorre las subcarpetas s1 y s2 de cada clase
        for sub in subfolders:
            subfolder_path = os.path.join(root_dir, cls, sub)
            images = os.listdir(subfolder_path)
            # Guardamos las imágenes de cada subcarpeta
            if sub == 's1':
                s1_images = images
            elif sub == 's2':
                s2_images = images
        
        # Aseguramos que las imágenes en s1 y s2 coinciden por los últimos 7 caracteres del nombre
        paired_images = []
        for img1 in s1_images:
            # Extraemos los últimos 7 caracteres antes de la extensión
            #print(img1)
            #base_name1 = os.path.splitext(img1)[0][-7:]


            parts = img1.split('_')
            base_name1 = "_".join(parts[3:]) 
            #print(base_name1)
            # Buscamos la imagen en s2 con el mismo sufijo
            for img2 in s2_images:
                #base_name2 = os.path.splitext(img2)[0][-7:]

                parts = img2.split('_')
                base_name2 = "_".join(parts[3:]) 
                
                if base_name1 == base_name2:
                    print(base_name1)
                    print(base_name2)
                    paired_images.append((img1, img2))
                    s2_images.remove(img2)  # Eliminar la imagen emparejada de s2
                    break
        
        # Mezclamos aleatoriamente las imágenes emparejadas
        random.shuffle(paired_images)
        
        # 80% para entrenamiento, 20% para prueba
        num_train = int(0.8 * len(paired_images))
        train_images = paired_images[:num_train]
        test_images = paired_images[num_train:]

        # Mover imágenes a la carpeta correspondiente (train y test)
        for img1, img2 in train_images:
            # Mover imagen de s1 a train/agri_s1 y s2 a train/agri_s2
            shutil.copy(os.path.join(root_dir, cls, 's1', img1), os.path.join(split_dir, 'train', f"{cls}_s1", img1))
            shutil.copy(os.path.join(root_dir, cls, 's2', img2), os.path.join(split_dir, 'train', f"{cls}_s2", img2))
        
        for img1, img2 in test_images:
            # Mover imagen de s1 a test/agri_s1 y s2 a test/agri_s2
            shutil.copy(os.path.join(root_dir, cls, 's1', img1), os.path.join(split_dir, 'test', f"{cls}_s1", img1))
            shutil.copy(os.path.join(root_dir, cls, 's2', img2), os.path.join(split_dir, 'test', f"{cls}_s2", img2))

if __name__ == "__main__":
    split_images()
    print("División completada.")


def load_image_pairs_with_labels_s2_s1(data_dir, classes, input_size=(256, 256)):
    s2_paths = []
    s1_paths = []
    labels = []  # Etiquetas: por ejemplo, 0 para la primera clase, 1 para la segunda

    for label, class_name in enumerate(classes):
        s2_folder = os.path.join(data_dir, f'{class_name}_s2')
        s1_folder = os.path.join(data_dir, f'{class_name}_s1')

        for file_name in os.listdir(s2_folder):

            parts = file_name.split('_')
            p1 = "_".join(parts[:2]) 
            p2 = "_".join(parts[3:]) 
            

            s2_path = os.path.join(s2_folder, p1 + '_s2_' +  p2)
            s1_path = os.path.join(s1_folder, p1 + '_s1_' +  p2)

            if os.path.exists(s2_path) and os.path.exists(s1_path):
                s2_paths.append(s2_path)
                s1_paths.append(s1_path)
                labels.append(label)  # Etiqueta basada en el índice de `classes`
            else:
                print('Warning mismatch')
                print(s2_path)
                print(s1_path)

    # Preprocesar imágenes
    def preprocess_image(image_path):
        print(image_path)
        img = load_img(image_path, target_size=input_size)
        return img_to_array(img) / 255.0  # Normalizar a [0, 1]

    # Cargar y procesar imágenes
    s2_images = np.array([preprocess_image(path) for path in s2_paths])  # Imágenes S2
    s1_images = np.array([preprocess_image(path)[:, :, 0:1] for path in s1_paths])  # Imágenes S1 en escala de grises
    labels = np.array(labels)  # Convertir etiquetas a array numpy
    
    return s2_images, s1_images, labels



#s2_images, s1_images, 

s2_images, s1_images, labels  = load_image_pairs_with_labels_s2_s1('v_2/split/train', ['agri', 'grassland'], input_size=(256, 256))