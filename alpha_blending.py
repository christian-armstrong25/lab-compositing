import cv2
import numpy as np
import matplotlib.pyplot as plt


def create_vert_alpha_matte(dims, f_width, f_location) -> np.ndarray:
    """
    dims: (height, width)
    f_width: width
    f_location: (height, width)
    b_width: (height, width)
    b_location: (height, width)
    """
    mask = np.zeros(dims, dtype=np.float32)
    start_feather = f_location - f_width // 2
    end_feather = f_location + f_width // 2

    if f_width % 2 != 0:
      end_feather += 1

    mask[:, end_feather:] = 1.0
    mask[:, start_feather:end_feather] = np.linspace(0.0, 1.0, f_width) 
    return mask

def blend_images_alpha(im1, im2, alpha) -> np.ndarray:
    """
    im1: (height, width, 3)
    im2: (height, width, 3)
    alpha: (height, width)
    """
    # error checking <3
    assert im1.shape == im2.shape
    assert im1.shape[:2] == alpha.shape
    assert im2.shape[:2] == alpha.shape

    return im2 * alpha[..., np.newaxis] + im1 * (1 - alpha[..., np.newaxis])

if __name__ == '__main__':
  apple = cv2.cvtColor(cv2.imread('images/burt_apple.png'), cv2.COLOR_BGR2RGB)
  orange = cv2.cvtColor(cv2.imread('images/burt_orange.png'), cv2.COLOR_BGR2RGB)

  apple = (apple / 255.0).astype(np.float32)
  orange = (orange / 255.0).astype(np.float32)

  alpha = create_vert_alpha_matte(apple.shape[:2], 120, apple.shape[1] // 2)
  
  plt.plot()
  plt.imshow(alpha, cmap='gray')
  plt.show()

  blended = blend_images_alpha(apple, orange, alpha)

  plt.plot()
  plt.imshow(blended)
  plt.show()
  
