from matplotlib import pyplot as plt
import cv2
import numpy as np
import utils



def subplot_images(img_array):
    """
    Muestra la imagen original, la imagen en escala de grises, y la imagen en espacio HSV 
    para cada imagen en img_array en un solo gráfico con subplots.
    
    Args:
        img_array: Lista de paths de las imágenes.
    """
    for image_path in img_array:
        img = cv2.imread(image_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  
        plt.figure(figsize=(15, 5))
        
        # Subplot 1: Imagen original
        plt.subplot(1, 3, 1)
        utils.imshow(img)
        plt.title('Imagen Original')
        
        # Subplot 2: Imagen en escala de grises
        plt.subplot(1, 3, 2)
        utils.imshow(gray_img, cmap='gray')
        plt.title('Escala de Grises')
        
        # Subplot 3: Imagen en espacio HSV
        plt.subplot(1, 3, 3)
        utils.imshow(hsv_img, cmap='hsv')
        plt.title('Espacio HSV')
        
        plt.suptitle(image_path.split("/")[-1], fontsize=16)
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
    utils.imshow(hsv_img, cmap='hsv')
    for i, (x, y) in enumerate(zip(X, Y)):
        color = colors[i]
        marker = markers[i]
        hsv_value = hsv_img[y, x] 
        plt.scatter(x, y, color=color, marker=marker, label=f'{color} - HSV: {hsv_value}', edgecolors='black', s=100)
    plt.legend(loc='lower left')
    plt.axis('off')
    plt.show()



def subplots_by_color(imgs_array, color_ranges):
    """
    Muestra la imagen original y las máscaras binarias para cada rango de color especificado en color_ranges.
    Devuelve un diccionario con las máscaras binarias para cada color en cada imagen.
    
    Args:
        imgs_array: Lista de paths de las imágenes.
        color_ranges: Diccionario con los rangos de color HSV.
    """
    masks = {}

    for image_path in imgs_array:
        img = cv2.imread(image_path)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        plt.figure(figsize=(20, 6)) 
        
        # Subplot 1: Imagen original
        plt.subplot(1, len(color_ranges) + 1, 1)
        utils.imshow(img)
        plt.title('Original')
        masks[image_path] = {}
        
        # Subplots para las máscaras de cada color
        for i, (color, (lower, upper)) in enumerate(color_ranges.items(), 2):
            lower_bound = np.array(lower)
            upper_bound = np.array(upper)
            mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
            plt.subplot(1, len(color_ranges) + 1, i)
            utils.imshow(mask, cmap='gray')
            plt.title(color)
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



def box_by_color(imagenes_binarizadas, colores, imagen_original, component_min_area=100, draw_mode="box"):
    """
    Devuelve la imagen original, dibujando sobre esta los bounding boxes con etiquetas o cruces
    para cada color y el número de objetos detectados en la imagen original.

    Args:
        imagenes_binarizadas: Diccionario con arrays binarizados por color.
        colores: Lista de nombres de colores en el mismo orden que las imágenes binarizadas.
        imagen_original: Imagen original sobre la cual se dibujarán los bounding boxes o cruces y etiquetas.
        component_min_area: Área mínima de la componente conectada para ser considerada válida.
        draw_mode: Modo de dibujo, puede ser "box" para cajas o "cross" para cruces.

    Returns:
        imagen_original: Imagen original con todos los objetos enmarcados o cruzados y etiquetados.
    """
    # verificaciones
    if len(imagenes_binarizadas) != len(colores):
        raise ValueError("El número de imágenes binarizadas debe coincidir con el número de colores.")
    
    if len(imagen_original.shape) == 2:  
        imagen_original = cv2.cvtColor(imagen_original, cv2.COLOR_GRAY2BGR)
    
    objeto_num_total = 1        # Contador para llevar la cuenta de los objetos válidos totales

    for color in colores:
        img_bin = imagenes_binarizadas[color]
        img_bin = cv2.threshold(img_bin, 128, 255, cv2.THRESH_BINARY)[1] 
        
        
        num_labels, img_labels, values, centroids = cv2.connectedComponentsWithStats(img_bin)       # Detectar componentes conexas

        for i in range(1, num_labels):  # Ignora el fondo (etiqueta 0)
            area = values[i, cv2.CC_STAT_AREA]
            if area >= component_min_area:  # Solo considero componentes suficientemente grandes
                x, y, w, h = values[i, cv2.CC_STAT_LEFT], values[i, cv2.CC_STAT_TOP], values[i, cv2.CC_STAT_WIDTH], values[i, cv2.CC_STAT_HEIGHT]
                
                if draw_mode == "box":
                    cv2.rectangle(imagen_original, (x, y), (x + w, y + h), (0, 0, 255), 2) 
                    cv2.putText(imagen_original, f"{color.capitalize()} {objeto_num_total}", (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1) 
                
                elif draw_mode == "cross":
                    cx, cy = int(centroids[i][0]), int(centroids[i][1])
                    cross_size = min(w, h) // 2
                    cross_color = (0, 0, 0) 
                    cross_thickness = 1  
                    cv2.line(imagen_original, (cx - cross_size, cy), (cx + cross_size, cy), cross_color, cross_thickness)  
                    cv2.line(imagen_original, (cx, cy - cross_size), (cx, cy + cross_size), cross_color, cross_thickness)
                
                else:
                    raise ValueError("El modo de dibujo debe ser 'box' o 'cross'.")
                objeto_num_total += 1
    

    # Mostrar la imagen con bounding boxes o cruces
    plt.figure(figsize=(10, 10))
    utils.imshow(imagen_original)
    plt.title(f'{objeto_num_total-1} objetos detectados')
    plt.axis('off')
    plt.show()
    
    return imagen_original