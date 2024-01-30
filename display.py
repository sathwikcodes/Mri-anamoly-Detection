import numpy as np
import matplotlib.pyplot as plt
import os


path_save = r"C:\Users\bnvsa\OneDrive\Desktop\data"


# Load the saved T1w and T2w arrays
T1w_array = np.load(os.path.join(path_save, "T1w_slices_4.npy"))
T2w_array = np.load(os.path.join(path_save, "T2w_slices_4.npy"))

# Display the shapes of the arrays
print("T1w Array Shape:", T1w_array.shape)
print("T2w Array Shape:", T2w_array.shape)

# Display a sample 2D slice from T1w array
plt.imshow(T1w_array[0], cmap="gray")
plt.title("Sample 2D Slice from T1w Array")
plt.show()

# Display a sample 2D slice from T2w array
plt.imshow(T2w_array[0], cmap="gray")
plt.title("Sample 2D Slice from T2w Array")
plt.show()
