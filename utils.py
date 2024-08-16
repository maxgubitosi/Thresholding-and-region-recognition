from matplotlib import pyplot as plt
import cv2
import numpy as np

"""
A set of utility functions handling opencv image formats and color-spaces
"""

def patch(img, kwargs):
  """
  A patching function that:
    - Defaults cmap to grayscale if detects images with only 1 channel.
    - Defaults cmap to rgb if detects images with 3 channels and no cmap defined.
    - Converts opencv default BGR format to RGB if detects images with 3 channels.
    - Converts opencv HSV to RGB
  """

  grayscale = {'cmap':'gray', 'vmin':0, 'vmax':255}
  
  cmap_patched = kwargs.copy()
  if len(img.shape) == 2:
    # num channels == 1
    # Defaulting cmap to grayscale
    if 'cmap' not in kwargs:
      cmap_patched.update(grayscale)

  img_patched = img
  if len(img.shape) == 3:
    if 'cmap' not in kwargs:
      # Changing BGR opencv format to RGB
      if img.shape[2] == 4:
        img_patched = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
      else:
        img_patched = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # matplotlib expects hsv in [0, 1] range, simply convert opencv HSV to RGB format
    if 'cmap' in kwargs and kwargs['cmap'] == 'hsv':
      img_patched = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

  return img_patched, cmap_patched


def imshow(img, **kwargs):
  """
  imshow wrapper for matplotlib.pyplot.imshow
  """
  patched_img , patched_cmap = patch(img, kwargs)
  plt.imshow(patched_img, **patched_cmap)
  plt.axis('off')

def show_images(images, titles=None, **kwargs):
  num_images = len(images)
  fig, axs = plt.subplots(1, num_images, figsize=(12, 6))
  if titles is None:
    titles = [None for _ in images]
  for ax, img, title in zip(axs, images, titles):

    patched_img , patched_cmap = patch(img, kwargs)
    ax.imshow(patched_img, **patched_cmap)
    ax.axis('off')
    ax.set_title(title)

def plot_transform(r, s, label=None, title=None, fig=None):
  if fig is None:
    plt.figure(figsize=(5, 5))
  if not isinstance(s, list):
    ss = [s]
  else:
    ss = s

  legend = True
  if label is None:
    legend = False
    ls = [None] * len(ss)
  else:
    if not isinstance(label, list):
      ls = [label] * len(ss)
    else:
      ls = label

  for s, lbl in zip(ss, ls):
    plt.plot(r, s, label=lbl)

  plt.grid()
  plt.xlabel("r")
  plt.ylabel("s")
  plt.title(title)
  if legend:
    plt.legend()
  plt.ylim(0, 256)
  plt.xlim(0, 256)



# Funciones agregadas por mi

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
    Genera subplots para cada color especificado en color_ranges para cada imagen en imgs_array.
    
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