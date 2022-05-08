import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img = np.array(Image.open('1.jpg'))
fig = plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(img)
random_gamma = np.random.uniform(low = 0.9, high = 1.1)
img1 = 255.0 * ((img / 255.0) ** random_gamma)
img1 = np.clip(img1, 0.0, 255.0)
img1 = img1.astype(np.uint8)
diff = np.abs(img - img1)
print(diff.max(), diff.min(), np.mean(diff))
plt.subplot(2, 2, 2)
plt.imshow(img1)

random_brightness = np.random.uniform(low = 0.8, high = 1.2)
plt.subplot(2, 2, 3)
img2 = 255.0 * (img / 255.0)
img2 = img2 * random_brightness
img2 = np.clip(img2, 0.0, 255.0)
img2 = img2.astype(np.uint8)
plt.imshow(img2)

plt.subplot(2, 2, 4)
random_colors = np.random.uniform(low = 0.8, high = 1.2, size = [3])
img3 = 255.0 * (img / 255.0)
img3 = img * np.reshape(random_colors, [1, 1, 3])
img3 = np.clip(img3, 0.0, 255.0)
img3 = img3.astype(np.uint8)
print(random_colors)
plt.imshow(img3)
plt.show()
