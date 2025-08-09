from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Step 1: Load the image
img = Image.open("input.jpg").convert("L")  # Convert to grayscale
img_array = np.array(img)  # Convert the image to a NumPy array

# Step 2: Reshape the image for PCA
pixels_flattened = img_array.reshape(img_array.shape[0], -1)  # Reshape rows as samples

# Step 3: Standardize the pixel values
mean = np.mean(pixels_flattened, axis=0)
std_dev = np.std(pixels_flattened, axis=0)
pixels_standardized = (pixels_flattened - mean) / std_dev

# Step 4: Perform PCA
num_components = 20  # Number of principal components to retain
pca = PCA(n_components=num_components)
pixels_pca = pca.fit_transform(pixels_standardized)

# Step 5: Reconstruct the image from principal components
pixels_reconstructed = pca.inverse_transform(pixels_pca)
pixels_reconstructed = (pixels_reconstructed * std_dev) + mean  # De-standardize
pixels_reconstructed = pixels_reconstructed.reshape(img_array.shape)  # Reshape

# Step 6: Visualize the original and reconstructed images
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img_array, cmap="gray")
plt.axis("off")

# Reconstructed image
plt.subplot(1, 2, 2)
plt.title("Reconstructed Image (PCA)")
plt.imshow(pixels_reconstructed, cmap="gray")
plt.axis("off")

plt.show()