import os
import shutil

# Carpeta principal
main_dir = "coffee"

# Subcarpetas con las imágenes
input_dirs = [os.path.join(main_dir, "RUST"), os.path.join(main_dir, "NORUST")]

# Subcarpetas de salida
output_dirs = [os.path.join(main_dir, "RUST_filtered"), os.path.join(main_dir, "NORUST_filtered")]

# Crear las carpetas de salida si no existen
for output_dir in output_dirs:
    os.makedirs(output_dir, exist_ok=True)

# Función para verificar si un archivo termina en 0 o 5
def ends_with_0_or_5(filename):
    # Extraer el nombre sin extensión
    name, _ = os.path.splitext(filename)
    # Verificar si termina en 0 o 5
    return name[-1] in {'0', '5'}

# Procesar cada carpeta
for input_dir, output_dir in zip(input_dirs, output_dirs):
    for filename in os.listdir(input_dir):
        # Verificar que sea una imagen con extensión válida
        if filename.lower().endswith((".jpg", ".tif", ".jpeg")):
            if ends_with_0_or_5(filename):
                # Copiar el archivo que cumple la condición a la carpeta de salida
                shutil.copy(os.path.join(input_dir, filename), os.path.join(output_dir, filename))

print("Filtrado completado. Archivos filtrados guardados en las subcarpetas *_filtered dentro de 'coffee'.")


# %%


from PIL import Image
import matplotlib.pyplot as plt

# Ruta de la imagen
ruta_imagen = "/Users/mmartinez/Desktop/MT_images/coffee/NORUST_filtered/15.TIF"

# Cargar la imagen usando Pillow
imagen = Image.open(ruta_imagen)

# Mostrar la imagen con Matplotlib
plt.imshow(imagen)
plt.axis('off')  # Ocultar ejes
plt.show()

# %%
import os
import shutil

# Directorios principales
carpetas_principales = ["coffee/NORUST_filtered", "coffee/RUST_filtered"]
nombres_subcarpetas = ["NIR", "RGB"]

# Función para organizar imágenes
def organizar_imagenes(carpeta_principal):
    print(f"Procesando la carpeta: {carpeta_principal}")
    
    # Crear subcarpetas NIR y RGB dentro de la carpeta principal
    for subcarpeta in nombres_subcarpetas:
        ruta_subcarpeta = os.path.join(carpeta_principal, subcarpeta)
        os.makedirs(ruta_subcarpeta, exist_ok=True)

    # Obtener la lista de archivos en la carpeta principal
    archivos = [archivo for archivo in os.listdir(carpeta_principal) if os.path.isfile(os.path.join(carpeta_principal, archivo))]

    if not archivos:
        print(f"La carpeta '{carpeta_principal}' está vacía. No se encontraron imágenes para organizar.")
        return

    # Iterar sobre los archivos
    for archivo in archivos:
        ruta_archivo = os.path.join(carpeta_principal, archivo)
        
        # Verifica si es una imagen que termina en 5 o 0

        if archivo[-5] == "5":  # Imágenes NIR
            destino = os.path.join(carpeta_principal, "NIR", archivo)
            shutil.move(ruta_archivo, destino)
            print(f"Movido a NIR: {archivo}")
        elif archivo[-5] == "0":  # Imágenes RGB
            destino = os.path.join(carpeta_principal, "RGB", archivo)
            shutil.move(ruta_archivo, destino)
            print(f"Movido a RGB: {archivo}")
        else:
            print(f"No coincide con las reglas: {archivo}")

# Aplicar la función a ambas carpetas principales
for carpeta in carpetas_principales:
    if os.path.exists(carpeta):
        organizar_imagenes(carpeta)
    else:
        print(f"La carpeta '{carpeta}' no existe. Verifica las rutas.")

print("Organización finalizada.")



# %%
import os
import shutil
import random

# Directorio principal
ruta_rust = "coffee/RUST_filtered"
subcarpetas_originales = ["NIR", "RGB"]
subcarpetas_copia = ["NIRs", "RGBs"]
num_muestras = 273
rango_posiciones = 849  # Máximo valor es 848, rango es de 0 a 848

def seleccionar_muestras_aleatorias(ruta_carpeta, num_muestras):
    """Selecciona num_muestras archivos aleatorios de una carpeta."""
    archivos = [archivo for archivo in os.listdir(ruta_carpeta) if os.path.isfile(os.path.join(ruta_carpeta, archivo))]
    if len(archivos) < num_muestras:
        raise ValueError(f"La carpeta '{ruta_carpeta}' contiene menos de {num_muestras} elementos.")
    
    # Seleccionar índices aleatorios únicos
    indices_aleatorios = random.sample(range(len(archivos)), num_muestras)
    archivos_seleccionados = [archivos[i] for i in indices_aleatorios]
    return archivos_seleccionados

def crear_copia_con_muestras(ruta_principal, subcarpeta_origen, subcarpeta_destino, num_muestras):
    """Crea una copia de los archivos seleccionados aleatoriamente en una nueva subcarpeta."""
    ruta_origen = os.path.join(ruta_principal, subcarpeta_origen)
    ruta_destino = os.path.join(ruta_principal, subcarpeta_destino)
    
    # Crear subcarpeta destino
    os.makedirs(ruta_destino, exist_ok=True)
    
    # Seleccionar muestras
    muestras = seleccionar_muestras_aleatorias(ruta_origen, num_muestras)
    
    # Copiar muestras a la nueva carpeta
    for archivo in muestras:
        origen = os.path.join(ruta_origen, archivo)
        destino = os.path.join(ruta_destino, archivo)
        shutil.copy(origen, destino)
        print(f"Copiado: {archivo} -> {ruta_destino}")

# Crear las carpetas NIRs y RGBs
for subcarpeta_origen, subcarpeta_destino in zip(subcarpetas_originales, subcarpetas_copia):
    crear_copia_con_muestras(ruta_rust, subcarpeta_origen, subcarpeta_destino, num_muestras)

print("Copia de muestras aleatorias completada.")


# %%
