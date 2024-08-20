from matplotlib import pyplot as plt
import cv2
import numpy as np

def subplot_images(img_array):
    """
    Muestra la imagen original, la imagen en escala de grises, y la imagen en espacio HSV 
    para cada imagen en img_array en un solo gráfico con subplots.
    
    Args:
        img_array: Lista de paths de las imágenes.
    """
    for image_path in img_array:
        img = cv2.imread(image_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        # Convertir la imagen a escala de grises
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)          # Convertir la imagen a espacio HSV
        plt.figure(figsize=(15, 5))
        
        # Subplot 1: Imagen original
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Imagen Original')
        plt.axis('off')
        
        # Subplot 2: Imagen en escala de grises
        plt.subplot(1, 3, 2)
        plt.imshow(gray_img, cmap='gray')
        plt.title('Escala de Grises')
        plt.axis('off')
        
        # Subplot 3: Imagen en espacio HSV
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB))
        plt.title('Espacio HSV')
        plt.axis('off')
        
        plt.suptitle(f'Visualizaciones para {image_path.split("/")[-1]}', fontsize=16)
        plt.show()



def subplot_points(hsv_img, X, Y):
    """
    Muestra una imagen con puntos en las coordenadas dadas y una leyenda que muestra el color correspondiente
    en formato HSV.
    
    Args:
        hsv_img: Imagen en formato HSV.
        X: Lista de coordenadas X de los puntos.
        Y: Lista de coordenadas Y de los puntos.
    """
    colors = ['yellow', 'orange', 'red', 'blue', 'green']
    markers = ['o', 's', 'D', 'v', '^']
    
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB))
    for i, (x, y) in enumerate(zip(X, Y)):
        color = colors[i]
        marker = markers[i]
        hsv_value = hsv_img[y, x]  # porque las coordenadas están en formato (y, x)
        plt.scatter(x, y, color=color, marker=marker, label=f'{color} - HSV: {hsv_value}', edgecolors='black', s=100)
    plt.legend(loc='lower left')
    plt.axis('off')
    plt.show()



def subplots_by_color(imgs_array, color_ranges):
    """
    Muestra una imagen original y las máscaras binarias para cada rango de color especificado en color_ranges.
    Devuelve un diccionario con las máscaras binarias para cada color en cada imagen.
    
    Args:
        imgs_array: Lista de rutas de imágenes.
        color_ranges: Diccionario con los rangos de color HSV.
    """
    masks = {}

    for image_path in imgs_array:
        img = cv2.imread(image_path)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        plt.figure(figsize=(20, 6)) 
        
        # Subplot 1: Imagen original
        plt.subplot(1, len(color_ranges) + 1, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Original')
        plt.axis('off')
        masks[image_path] = {}
        
        # Subplots para las máscaras de cada color
        for i, (color, (lower, upper)) in enumerate(color_ranges.items(), 2):
            lower_bound = np.array(lower)
            upper_bound = np.array(upper)
            mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
            plt.subplot(1, len(color_ranges) + 1, i)
            plt.imshow(mask, cmap='gray')
            plt.title(color)
            plt.axis('off')
            masks[image_path][color] = mask
        plt.suptitle(f'Máscaras para {image_path.split("/")[-1]}', fontsize=16)
        plt.show()
    return masks



def threshold_image_for_color(img, color):
    """
    Genera una máscara para un color específico en una imagen.
    
    Args:
        img: Imagen en formato BGR.
        color: Tupla con los valores HSV del color.
        
    Returns:
        mask: Máscara binaria para el color especificado.
    """
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_bound = np.array(color[0])
    upper_bound = np.array(color[1])
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
    return mask



def box_by_color(imagenes_binarizadas, colores, imagen_original, component_min_area=100):
    """
    Combina todas las imágenes binarizadas en una sola imagen, dibujando los bounding boxes y 
    etiquetas para cada color y número de confite sobre la imagen original.

    Args:
        imagenes_binarizadas: Diccionario con arrays binarizados por color.
        colores: Lista de nombres de colores en el mismo orden que las imágenes binarizadas.
        imagen_original: Imagen original sobre la cual se dibujarán los bounding boxes y etiquetas.

    Returns:
        img_bbox: Imagen original con todos los confites enmarcados y etiquetados.
    """
    # Verifica que el número de imágenes binarizadas coincida con el número de colores
    if len(imagenes_binarizadas) != len(colores):
        raise ValueError("El número de imágenes binarizadas debe coincidir con el número de colores.")
    
    # Asegúrate de que la imagen original esté en formato BGR
    if len(imagen_original.shape) == 2:  # Si la imagen es en escala de grises
        imagen_original = cv2.cvtColor(imagen_original, cv2.COLOR_GRAY2BGR)
    
    # Contador para llevar la cuenta de los confites válidos por color
    confite_num_total = 1

    # Iterar sobre cada color y su imagen binarizada correspondiente
    for color in colores:
        img_bin = imagenes_binarizadas[color]
        img_bin = cv2.threshold(img_bin, 128, 255, cv2.THRESH_BINARY)[1]  # Asegúrate de que la imagen esté correctamente binarizada
        
        # Detectar componentes conectadas
        num_labels, img_labels, values, centroids = cv2.connectedComponentsWithStats(img_bin)

        # Iterar sobre cada componente conectada y marcarla con un bounding box
        for i in range(1, num_labels):  # Ignoramos el fondo (etiqueta 0)
            area = values[i, cv2.CC_STAT_AREA]
            if area >= component_min_area:  # Solo consideramos componentes suficientemente grandes
                x, y, w, h = values[i, cv2.CC_STAT_LEFT], values[i, cv2.CC_STAT_TOP], values[i, cv2.CC_STAT_WIDTH], values[i, cv2.CC_STAT_HEIGHT]
                cv2.rectangle(imagen_original, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Dibujar el bounding box en rojo
                cv2.putText(imagen_original, f"{color.capitalize()} {confite_num_total}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # Añadir el label en verde
                confite_num_total += 1 

    # Mostrar la imagen con los bounding boxes y etiquetas
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(imagen_original, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"{confite_num_total - 1} Confites detectados")  # Restar 1 para el conteo correcto
    plt.show()

    return imagen_original