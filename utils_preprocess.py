import os
import numpy as np
from PIL import Image
import tifffile as tiff



def extract_and_save_rgb_ir_eurosat(input_folder):
    """
    Extrae las imágenes RGB (bandas 4, 3, 2) e infrarrojo cercano (banda 8)
    de imágenes TIFF de Sentinel-2 en el directorio de entrada y las guarda
    en carpetas con nombres específicos en el nivel de 'EuroSATallBands'.

    Args:
    - input_folder (str): Carpeta de entrada con imágenes TIFF.
    """
    # Obtener el nombre del subdirectorio (e.g., "River")
    base_dir = os.path.dirname(input_folder)  # Ruta hasta 'EuroSATallBands'
    subfolder_name = os.path.basename(input_folder)  # Nombre del subdirectorio (e.g., "River")
    
    # Crear nuevas carpetas en el nivel de EuroSATallBands
    rgb_folder = os.path.join(base_dir, f"{subfolder_name}RGB")
    ir_folder = os.path.join(base_dir, f"{subfolder_name}NIR")
    os.makedirs(rgb_folder, exist_ok=True)
    os.makedirs(ir_folder, exist_ok=True)

    # Procesar todas las imágenes en la carpeta de entrada
    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):  # Asegurarse de que solo se procesen archivos .tif
            file_path = os.path.join(input_folder, filename)

            # Leer la imagen TIFF
            tiff_image = tiff.imread(file_path)

            # Verificar que la imagen tiene al menos 13 bandas
            if len(tiff_image.shape) == 3 and tiff_image.shape[2] >= 13:
                # Extraer las bandas para RGB (4, 3, 2 en Sentinel-2, índices 3, 2, 1)
                band_4 = tiff_image[:, :, 3]  # Banda Roja (Band 4)
                band_3 = tiff_image[:, :, 2]  # Banda Verde (Band 3)
                band_2 = tiff_image[:, :, 1]  # Banda Azul (Band 2)

                # Normalizar cada banda de forma separada (usando la fórmula band - band.min() / (band.max() - band.min()) * 255)
                band_4_normalized = (band_4 - band_4.min()) / (band_4.max() - band_4.min()) * 255  # Banda Roja
                band_3_normalized = (band_3 - band_3.min()) / (band_3.max() - band_3.min()) * 255  # Banda Verde
                band_2_normalized = (band_2 - band_2.min()) / (band_2.max() - band_2.min()) * 255  # Banda Azul

                # Convertir las bandas normalizadas a tipo uint8 (rango [0, 255])
                band_4_normalized = np.clip(band_4_normalized, 0, 255).astype(np.uint8)
                band_3_normalized = np.clip(band_3_normalized, 0, 255).astype(np.uint8)
                band_2_normalized = np.clip(band_2_normalized, 0, 255).astype(np.uint8)

                # Concatenar las bandas normalizadas en una imagen RGB
                rgb_image = np.stack((band_4_normalized, band_3_normalized, band_2_normalized), axis=-1)  # Orden: Rojo, Verde, Azul

                rgb_output_path = os.path.join(rgb_folder, f"{os.path.splitext(filename)[0]}_RGB.png")
                Image.fromarray(rgb_image).save(rgb_output_path)
                print(f"Guardada RGB: {rgb_output_path}")

                # Extraer la banda infrarroja cercana (banda 8 en Sentinel-2, índice 7)
                ir_band = tiff_image[:, :, 7]  # Índice para la banda 8
                ir_normalized = (ir_band - ir_band.min()) / (ir_band.max() - ir_band.min()) * 255
                ir_normalized = ir_normalized.astype(np.uint8)
                ir_output_path = os.path.join(ir_folder, f"{os.path.splitext(filename)[0]}_NIR.png")
                Image.fromarray(ir_normalized).save(ir_output_path)
                print(f"Guardada NIR: {ir_output_path}")
            else:
                print(f"Saltando archivo (estructura no válida o no tiene suficientes bandas): {file_path}")

    print("Proceso completado.")



import os
import numpy as np
from PIL import Image
import tifffile as tiff



def extract_and_save_rgb_nir_swir_eurosat(input_folder):

    def normalize_band(band):
        norm_band = (band - band.min()) / (band.max() - band.min()) * 255
        return np.clip(norm_band, 0, 255).astype(np.uint8)

    base_dir = os.path.dirname(input_folder)
    subfolder_name = os.path.basename(input_folder)

    # Crear carpetas de salida
    rgb_folder = os.path.join(base_dir, f"{subfolder_name}RGB")
    nir_folder = os.path.join(base_dir, f"{subfolder_name}NIR")
    swir_folder = os.path.join(base_dir, f"{subfolder_name}SWIR")
    os.makedirs(rgb_folder, exist_ok=True)
    os.makedirs(nir_folder, exist_ok=True)
    os.makedirs(swir_folder, exist_ok=True)

    # Procesar cada archivo TIFF
    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):
            file_path = os.path.join(input_folder, filename)

            try:
                tiff_image = tiff.imread(file_path)

                if len(tiff_image.shape) == 3 and tiff_image.shape[2] >= 13:
                    # Extraer bandas RGB
                    band_4 = normalize_band(tiff_image[:, :, 3])  # Rojo
                    band_3 = normalize_band(tiff_image[:, :, 2])  # Verde
                    band_2 = normalize_band(tiff_image[:, :, 1])  # Azul

                    rgb_image = np.stack((band_4, band_3, band_2), axis=-1)
                    rgb_output_path = os.path.join(rgb_folder, f"{os.path.splitext(filename)[0]}_RGB.png")
                    Image.fromarray(rgb_image).save(rgb_output_path)
                    print(f"Guardada RGB: {rgb_output_path}")

                    # Extraer banda NIR
                    nir_band = normalize_band(tiff_image[:, :, 7])  # Infrarrojo cercano
                    nir_output_path = os.path.join(nir_folder, f"{os.path.splitext(filename)[0]}_NIR.png")
                    Image.fromarray(nir_band).save(nir_output_path)
                    print(f"Guardada NIR: {nir_output_path}")

                    # Extraer banda SWIR
                    swir_band = normalize_band(tiff_image[:, :, 10])  # SWIR
                    swir_output_path = os.path.join(swir_folder, f"{os.path.splitext(filename)[0]}_SWIR.png")
                    Image.fromarray(swir_band).save(swir_output_path)
                    print(f"Guardada SWIR: {swir_output_path}")
                else:
                    print(f"Saltando archivo (estructura no válida o insuficientes bandas): {file_path}")

            except Exception as e:
                print(f"Error procesando archivo {file_path}: {e}")

    print("Proceso completado.")



import os
import shutil

def filter_and_copy_matching_files(rgb_folder, swir_folder, output_folder):
    # Crear carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)

    # Obtener nombres base de archivos en la carpeta RGB
    rgb_files = os.listdir(rgb_folder)
    rgb_base_names = {
        "_".join(filename.split("_")[:-1]) for filename in rgb_files if "_" in filename
    }

    # Filtrar archivos de la carpeta SWIR basados en nombres base
    swir_files = os.listdir(swir_folder)
    matched_files = [
        filename for filename in swir_files
        if "_".join(filename.split("_")[:-1]) in rgb_base_names
    ]

    # Copiar archivos coincidentes a la carpeta de salida
    for filename in matched_files:
        source_path = os.path.join(swir_folder, filename)
        destination_path = os.path.join(output_folder, filename)
        shutil.copy(source_path, destination_path)

    print(f"Archivos copiados: {len(matched_files)}")
    return matched_files





for i in ['River', 'Forest', 'AnnualCrop', 'PermanentCrop', 'Highway', 'Pasture']:
    for j in ['train', 'test']:
        # Directorios de ejemplo
        rgb_folder = "eurosat/split/" + j + "/" + i  + "RGB"
        swir_folder = "eurosat/EuroSATallBands/" + i + "SWIR"
        output_folder = "eurosat/split/" + j + "/" + i + "SWIR"

        # Ejecutar la función
        copied_files = filter_and_copy_matching_files(rgb_folder, swir_folder, output_folder)
        print("Archivos copiados:", copied_files)




def extract_and_save_swir2(input_folder):

    def normalize_band(band):
        norm_band = (band - band.min()) / (band.max() - band.min()) * 255
        return np.clip(norm_band, 0, 255).astype(np.uint8)

    base_dir = os.path.dirname(input_folder)
    subfolder_name = os.path.basename(input_folder)

    # Crear carpetas de salida
    # rgb_folder = os.path.join(base_dir, f"{subfolder_name}RGB")
    # nir_folder = os.path.join(base_dir, f"{subfolder_name}NIR")
    # swir_folder = os.path.join(base_dir, f"{subfolder_name}SWIR")
    swir2_folder = os.path.join(base_dir, f"{subfolder_name}SWIR2")  # Nueva carpeta SWIR2

    # os.makedirs(rgb_folder, exist_ok=True)
    # os.makedirs(nir_folder, exist_ok=True)
    # os.makedirs(swir_folder, exist_ok=True)
    os.makedirs(swir2_folder, exist_ok=True)  # Crear carpeta para SWIR2

    # Procesar cada archivo TIFF
    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):
            file_path = os.path.join(input_folder, filename)

            try:
                tiff_image = tiff.imread(file_path)

                if len(tiff_image.shape) == 3 and tiff_image.shape[2] >= 13:
                    # # Extraer bandas RGB
                    # band_4 = normalize_band(tiff_image[:, :, 3])  # Rojo
                    # band_3 = normalize_band(tiff_image[:, :, 2])  # Verde
                    # band_2 = normalize_band(tiff_image[:, :, 1])  # Azul

                    # rgb_image = np.stack((band_4, band_3, band_2), axis=-1)
                    # rgb_output_path = os.path.join(rgb_folder, f"{os.path.splitext(filename)[0]}_RGB.png")
                    # Image.fromarray(rgb_image).save(rgb_output_path)
                    # print(f"Guardada RGB: {rgb_output_path}")

                    # # Extraer banda NIR
                    # nir_band = normalize_band(tiff_image[:, :, 7])  # Infrarrojo cercano (NIR)
                    # nir_output_path = os.path.join(nir_folder, f"{os.path.splitext(filename)[0]}_NIR.png")
                    # Image.fromarray(nir_band).save(nir_output_path)
                    # print(f"Guardada NIR: {nir_output_path}")

                    # # Extraer banda SWIR
                    # swir_band = normalize_band(tiff_image[:, :, 10])  # SWIR
                    # swir_output_path = os.path.join(swir_folder, f"{os.path.splitext(filename)[0]}_SWIR.png")
                    # Image.fromarray(swir_band).save(swir_output_path)
                    # print(f"Guardada SWIR: {swir_output_path}")

                    # Extraer banda SWIR2
                    swir2_band = normalize_band(tiff_image[:, :, 12])  # SWIR2
                    swir2_output_path = os.path.join(swir2_folder, f"{os.path.splitext(filename)[0]}_SWIR2.png")
                    Image.fromarray(swir2_band).save(swir2_output_path)
                    print(f"Guardada SWIR2: {swir2_output_path}")

                else:
                    print(f"Saltando archivo (estructura no válida o insuficientes bandas): {file_path}")

            except Exception as e:
                print(f"Error procesando archivo {file_path}: {e}")

    print("Proceso completado.")
