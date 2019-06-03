
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
                if "_t2_" in f:
#               if "_t3_" in f or "_t2_" in f:
                    file_list.append(os.path.join(dirpath, f))
    return file_list

#list all images to analyze
files = list_files(path_folder_raw)

save_path = "O:/Projects/KP0011/3/RefData/Result/"

#iterate over images

PLACL = []
ID = []

k = "O:/FIP/2018/WW023/RefTraits/Macro/stb_senescence2018_fpww023/macro_outcomes/t3\\Overlay\\fpww023_t3_sn15_1_leafOriginal.png"

#iterate over all images

for k in files:

    #verbose iter
    print(os.path.basename(k))

    try:

        # Load image, convert from BGR to RGB
        img = cv2.cvtColor(cv2.imread(k),
                           cv2.COLOR_BGR2RGB)

        ############################################
        # SEGMENT LEAF FROM BACKGROUND
        ############################################

        # crop to area of interest
        img = img[350:1900, 285:8000]

        # remove white background
        # blur image
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

        # Transform image to HSV color space where brown can be efficiently segmented
        # this efficiently segments lesions; insect damage, powdery mildew symptoms and other damages on leaves are not
        # considered.
        img = cv2.cvtColor(Color_Masked, cv2.COLOR_RGB2HSV)

        # Create a mask (i.e. binary image) for brown pixels;
        # Ideal brown color has HSV (h, s, v) (30°, 100%, 59%). In openCV this corresponds to (15, 255, 150);
        # "Near to brown" pixels (i.e. more brown or more green) pixels are also included by defining a wider range
        mask = cv2.inRange(img, np.array([0, 95, 95]), np.array([25, 255, 255]))

        # Remove small objects from the mask, these arise from non-uniform color of lesions;
        ## Get all connected components
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(cv2.bitwise_not(mask), connectivity=8)

        ## Remove the background, which also counts as component
        sizes = stats[1:, -1];
        nb_components = nb_components - 1

        ## Define a minimal particle size
        min_size = 3500

        ## Remove components smaller than min_size
        mask_cleaned_1 = np.ones((output.shape))
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                mask_cleaned_1[output == i + 1] = 255
        mask_cleaned_1 = cv2.bitwise_not(np.uint8(mask_cleaned_1))

        # Remove noise and smooth mask borders
        ## Define an elliptic structuring element
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))
        ## Perform morphological opening (i.e. erosion followed by dilation)
        opening = cv2.morphologyEx(mask_cleaned_1, cv2.MORPH_OPEN, kernel)

        # Remove small detected areas; these are most likely not lesions
        ## Get all connected components
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(opening, connectivity=8)

        ## Remove the background, which also counts as component
        sizes = stats[1:, -1];
        nb_components = nb_components - 1

        ## Define minimal lesion size
        min_size = 2500

        ## Remove components smaller than min_size
        mask_cleaned_2 = np.zeros((output.shape))
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                mask_cleaned_2[output == i + 1] = 255
        mask_cleaned_2 = np.uint8(mask_cleaned_2)


        # If there are any lesions, filter them according to
        #  i) the position of their centroid and the number of lesions on the entire leaf;
        #  ii) convexity defects.
        # these criteria help to avoid detecting leaf tip necrosis as STB lesions.

        try:
            # Get all connected components
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask_cleaned_2, connectivity=8)
            # Remove background
            nb_components = nb_components - 1

            #first centroid is from background, hence not of interest
            centroids = centroids[1:,]
            #get the right-most component centroid x-coordinate
            max_r_comp = centroids[0:,0].max()
            #identify the ocomponent
            comp_id = list(centroids[0:,0]).index(max(list(centroids[0:,0])))
            comp_size = sizes[comp_id]

            # Get right hand edge of the leaf
            rows, cols = np.nonzero(mask_cleaned_seg)
            right_edge = cols.max()

            #get convexity defects
            _, contours, _ = cv2.findContours(mask_cleaned_2, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

            # Find right-most contour
            maxx = []
            for i in range(0, len(contours)):
                maxx.append(contours[i][0:, 0].max())

            cnt_right = contours[maxx.index(max(maxx))]

            hull = cv2.convexHull(cnt_right, returnPoints=False)
            defects = cv2.convexityDefects(cnt_right, hull)

            maj_defs = []
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(cnt_right[s][0])
                end = tuple(cnt_right[e][0])
                far = tuple(cnt_right[f][0])
                cv2.line(Color_Masked, start, end, [0, 255, 0], 2)
                if d > 75000:
                    cv2.circle(Color_Masked, far, 10, [0, 0, 255], -1)
                    maj_defs.append(1)

            # Drop component if only leaf tip necrosis: If there are only few lesions,
            # and if centroid of the right-most lesion is close to leaf tip
            mask_cleaned_3 = mask_cleaned_2.copy()
            if (nb_components <= 5 and comp_size > 50000) or len(maj_defs) > 0:
                if max_r_comp >= 0.8*right_edge:
                    mask_cleaned_3[output == comp_id + 1] = 0

            # Find contours
            _, contours, _ = cv2.findContours(mask_cleaned_3, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

            # Draw contours onto original image
            cnt = cv2.drawContours(Color_Masked, contours, -1, (255,0,0), 2)

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

        # Resize image
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