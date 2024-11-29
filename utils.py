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
