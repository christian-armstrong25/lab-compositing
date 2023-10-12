import cv2
import numpy as np
from skimage.filters import gaussian
import matplotlib.pyplot as plt
from alpha_blending import create_vert_alpha_matte


SIGMA = 2.0

def build_gaussian_pyramid(img, num_levels, scale_factor):
  '''
  Creates a pyramid of images with num_levels
  '''
  pyramid = [img]
  downsampled = img
  for _ in range(1, num_levels):
    filtered = gaussian(downsampled, sigma=SIGMA, preserve_range=True)
    downsampled = filtered[::scale_factor, ::scale_factor]
    pyramid.append(downsampled)
  return pyramid

def build_laplacian_pyramid(img, num_levels, scale_factor):
  '''
  Creates a pyramid of images with num_levels
  '''
  pyramid = []
  downsampled = img
  for _ in range(1, num_levels):
    filtered = gaussian(downsampled, sigma=SIGMA, preserve_range=True)
    pyramid.append(downsampled - filtered)
    downsampled = filtered[::scale_factor, ::scale_factor]
  pyramid.append(downsampled)
  return pyramid

def construct_combined_pyramid(im1, im2, mask, num_levels, scale_factor, is_grayscale=False):
  L_1 = build_laplacian_pyramid(im1, num_levels, scale_factor)
  L_2 = build_laplacian_pyramid(im2, num_levels, scale_factor)
  G_m = build_gaussian_pyramid(mask, num_levels, scale_factor)

  if not is_grayscale:
    G_m = [level[..., np.newaxis] for level in G_m]

  L_o = []
  for i in range(num_levels):
    L_o.append(G_m[i] * L_1[i] + (1 - G_m[i]) * L_2[i])

  print(L_o)

  return L_o

def reconstruct_image_from_laplacian(pyramid, scale_factor):
  '''
  Reconstructs an image from a pyramid
  '''
  pyramid = pyramid.reverse()
  upsampled = pyramid[0]
  for i in range(1, len(pyramid)):
    upsampled = cv2.resize(upsampled, None, fx=scale_factor, fy=scale_factor)
    upsampled = gaussian(upsampled, sigma=SIGMA, preserve_range=True) 
    upsampled += pyramid[i]

  return upsampled

def blend_images_laplacian(im1, im2, mask, num_levels, scale_factor, is_grayscale=False):
  L_o = construct_combined_pyramid(im1, im2, mask, num_levels, scale_factor, is_grayscale)
  return reconstruct_image_from_laplacian(L_o, scale_factor)

if __name__ == '__main__':
  apple = cv2.cvtColor(cv2.imread('images/burt_apple.png'), cv2.COLOR_BGR2RGB)
  orange = cv2.cvtColor(cv2.imread('images/burt_orange.png'), cv2.COLOR_BGR2RGB)

  apple = (apple / 255.0).astype(np.float32)
  orange = (orange / 255.0).astype(np.float32)

  alpha = create_vert_alpha_matte(apple.shape[:2], 120, apple.shape[1] // 2)

  blended = blend_images_laplacian(apple, orange, alpha, 5, 2)

  plt.plot()
  plt.imshow(blended)
  plt.show()


  