import nibabel as nib
import os
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

def process_file(file_path, count):
    print(f"Reading file {count}: {file_path}")
    img = nib.load(file_path, mmap=True)  # Use memory mapping
    data = img.get_fdata()  # Get the data as a NumPy array
    img.uncache()  # Close the NIfTI image to release resources
    return data

# Define the root folder where subfolders for each person's MRI images are stored
root_folder = r'E:\Work\Data\HCP-YARe'

# Initialize dictionaries to store the data
nifti_data = {}
slice_data = {}
T1w = []
T2w = []

# Define the fixed x-coordinates
x_coordinates = [60, 120]

# Traverse the directory structure
for root, dirs, files in os.walk(root_folder):
    for dir_name in dirs:
        if dir_name.startswith('9'):
            mni_folder = os.path.join(root, dir_name, 'mni')
            if os.path.exists(mni_folder):
                nifti_files = [f for f in os.listdir(mni_folder) if f.endswith('reoriented.mni.nii')]
                
                with ThreadPoolExecutor() as executor:
                    futures = []
                    for count, nifti_file in enumerate(nifti_files, start=1):
                        file_path = os.path.join(mni_folder, nifti_file)
                        futures.append(executor.submit(process_file, file_path, count))
                    
                    for count, future in enumerate(futures, start=1):
                        data = future.result()
                        nifti_data[nifti_files[count - 1]] = data

                        # Extract the 2D slices at the specified x-coordinates
                        slices = [data[x_coord, :, :] for x_coord in x_coordinates]
                        slice_data[nifti_files[count - 1]] = slices

                        # Check the file name and categorize slices
                        if 'hrT1' in nifti_files[count - 1]:
                            T1w.extend(slices)
                        elif 'hrT2' in nifti_files[count - 1]:
                            T2w.extend(slices)

# Convert the lists of 2D slices to NumPy arrays
T1w_array = np.array(T1w)
T2w_array = np.array(T2w)

path_save = r'E:\Work\Data\numpy_arrays'

# Save the T1w and T2w arrays
np.save(os.path.join(path_save, 'T1w_slices_9.npy'), T1w_array)
np.save(os.path.join(path_save, 'T2w_slices_9.npy'), T2w_array)

# # Print the saved arrays
# loaded_T1w = np.load(os.path.join(path_save, 'T1w_slices_2.npy'))
# loaded_T2w = np.load(os.path.join(path_save, 'T2w_slices_2.npy'))
# print(loaded_T1w)
# print(loaded_T2w)

# # Display the 2D slices using Matplotlib
# for file_name, slice_arrays in slice_data.items():
#     for i, slice_array in enumerate(slice_arrays):
#         plt.figure()
#         plt.imshow(slice_array, cmap='gray')  # Display in grayscale
#         plt.title(f"2D Slice {i+1} from {file_name} at x={x_coordinates[i]}")

# plt.show()
# plt.close()