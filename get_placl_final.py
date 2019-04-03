
#====================================================================================== -

# AUTHOR: Jonas Anderegg

# Extract PLACL from leaf scans

#====================================================================================== -


#import required packages and functions

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import pandas as pd

path_folder_raw = "O:/FIP/2018/WW023/RefTraits/Macro/stb_senescence2018_fpww023/macro_outcomes/"

#function to list all images to analyze
def list_files(dir):
    file_list = []
    for (dirpath, dirnames, filenames) in os.walk(dir):
        for f in filenames:
            if f.endswith("leafOriginal.png"):
                if "_t3_" in f:
                    file_list.append(os.path.join(dirpath, f))
    return file_list

#list all images to analyze
files = list_files(path_folder_raw)

save_path = "O:/FIP/2018/WW023/RefTraits/Preprocessed/t3/"

#iterate over images

PLACL = []
ID = []

#iterate over all images

for k in files:

    try:

        # Load image, convert from BGR to RGB
        img = cv2.cvtColor(cv2.imread(k),
                           cv2.COLOR_BGR2RGB)

        ############################################
        # SEGMENT LEAF FROM BACKGROUND
        ############################################

        # crop to area of interest, removing black lines
        img = img[350:1900, 285:8000]

        # remove white background
        # blur image a bit
        blur = cv2.GaussianBlur(img, (15, 15), 2)

        # mask for paper background
        lower_white = np.array([200, 200, 200], dtype=np.uint8)  # threshold for white pixels
        upper_white = np.array([255, 255, 255], dtype=np.uint8)
        mask1 = cv2.inRange(blur, lower_white, upper_white)  # could also use threshold
        # mask needs to be inverted,
        # since we want to set the BACKGROUND to white
        mask1 = cv2.bitwise_not(mask1)

        # There are still spots not belonging to the leaf
        # remove small objects to get rid of them

        # find all connected components
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask1, connectivity=8)
        # connectedComponentswithStats yields every seperated component with information on each of them, such as size
        # take out the background which is also considered a component
        sizes = stats[1:, -1];
        nb_components = nb_components - 1

        # minimal size of particle
        # somewhere between largest unreal feature and leaf size
        min_size = 600000

        # cleaned mask
        mask_cleaned_seg = np.zeros((output.shape))
        # for every component in the image,
        # keep only those above min_size
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                mask_cleaned_seg[output == i + 1] = 255
        mask_cleaned_seg = np.uint8(mask_cleaned_seg)

        # apply cleaned mask to the image
        [indx, indy] = np.where(mask_cleaned_seg == 0)
        Color_Masked = img.copy()
        Color_Masked[indx, indy] = 255

        ############################################
        # GET LESIONS AS BROWN PIXELS AND FILTER
        ############################################

        # Transform to HSV
        img = cv2.cvtColor(Color_Masked, cv2.COLOR_RGB2HSV)

        # Create a mask for brown pixels
        mask = cv2.inRange(img, np.array([0, 95, 95]), np.array([19, 255, 255]))

        # Remove small areas (holes WITHIN LESIONS)
        ## Find all connected components
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(cv2.bitwise_not(mask), connectivity=8)
        ## ConnectedComponentswithStats yields every seperated component with information on each of them, such as size
        ## Take out the background which is also considered a component
        sizes = stats[1:, -1];
        nb_components = nb_components - 1
        ## Define minimal size of particle
        min_size = 3500
        ## Create cleaned mask
        mask_cleaned_1 = np.ones((output.shape))
        ## for every component in the image,
        ## keep only those above min_size
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                mask_cleaned_1[output == i + 1] = 255
        mask_cleaned_1 = cv2.bitwise_not(np.uint8(mask_cleaned_1))

        # Remove Noise
        ## Rectangular Kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
        ## Remove noise by morphological opening
        opening = cv2.morphologyEx(mask_cleaned_1, cv2.MORPH_OPEN, kernel)

        # Remove small areas (holes WITHIN HEALTHY)
        ## Find all connected components
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(opening, connectivity=8)
        ## ConnectedComponentswithStats yields every seperated component with information on each of them, such as size
        ## Take out the background which is also considered a component
        sizes = stats[1:, -1];
        nb_components = nb_components - 1
        ## Define minimal size of particle
        min_size = 2500
        ## Create cleaned mask
        mask_cleaned_2 = np.zeros((output.shape))
        ## For every component in the image,
        ## keep only those above min_size
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                mask_cleaned_2[output == i + 1] = 255
        mask_cleaned_2 = np.uint8(mask_cleaned_2)

        try:

            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask_cleaned_2, connectivity=8)
            nb_components = nb_components - 1

            #first centroid is from background, not of interest, drop
            centroids = centroids[1:,]
            #get the right-most component centroid x-coordinate
            max_r_comp = centroids[0:,0].max()
            #identify the ocomponent
            comp_id = list(centroids[0:,0]).index(max(list(centroids[0:,0])))

            # Get right hand edge of the leaf
            rows, cols = np.nonzero(mask_cleaned_seg)
            right_edge = cols.max()

            # Drop component if only leaf tip necrosis: If there are only few lesions,
            # and if centroid of the right-most lesion is close to leaf tip
            mask_cleaned_3 = mask_cleaned_2.copy()
            if nb_components <= 5:
                if max_r_comp >= 0.8*right_edge:
                    mask_cleaned_3[output == comp_id + 1] = 0

            # Find contours
            _, contours, _ = cv2.findContours(mask_cleaned_3, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

            # Draw contours onto original image
            cnt = cv2.drawContours(Color_Masked, contours, -1, (128,255,0), 2)

            # Calculate PLACL as the area of the lesion devided by the area of the segmented leaf
            result = np.count_nonzero(mask_cleaned_3)/np.count_nonzero(mask_cleaned_seg)

            # Add to
            PLACL.append(result)
            ID.append(os.path.basename(k))

        except:

            print("no lesions in:" + k)

            PLACL.append(0)
            ID.append(os.path.basename(k))

            # Find contours
            _, contours, _ = cv2.findContours(mask_cleaned_2, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

            # Draw contours onto original image
            cnt = cv2.drawContours(Color_Masked, contours, -1, (128,255,0), 2)

        ############################################
        # SAVE RESULT
        ############################################

        # Resize image to save space
        cnt = cv2.resize(cnt, (0, 0), fx=0.5, fy=0.5)

        # Save overlay
        filename = os.path.basename(k)
        cv2.imwrite(save_path + filename, cv2.cvtColor(cnt, cv2.COLOR_RGB2BGR))

    except:

        print("Error in: " + k)

#store output in dataframe
df = pd.DataFrame(
    {'id': ID,
     'placl': PLACL,
    })

#save to csv
df.to_csv("O:/Projects/KP0011/3/RefData/Result/placl.csv", index = False)

# plot results if required
fig, ax = plt.subplots(1, 2, figsize=(10, 10), sharex=True, sharey=True)
ax[0].imshow(mask_cleaned_3)
ax[0].set_title("seg")
ax[1].imshow(drawing)
ax[1].set_title("final")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(Color_Masked)
ax.set_title("MASKED")
plt.tight_layout()
plt.show()