
#====================================================================================== -

# AUTHOR: Jonas Anderegg

# Extract PLACL from leaf scans via thresholding in the HSV color space
# and simple filtering operations using functions of openCV V3.0.0.

#====================================================================================== -


#import required packages and functions

import numpy as np
import cv2
import os
import pandas as pd

path_folder_raw = dirfrom

#function to list all images to analyze
#only t2 and t3 scans are amenable to the analysis
def list_files(dir):
    file_list = []
    for (dirpath, dirnames, filenames) in os.walk(dir):
        for f in filenames:
            if f.endswith("leafOriginal.png"):
                if "_t3_" in f or "_t2_" in f:
                    file_list.append(os.path.join(dirpath, f))
    return file_list

#list all images to analyze
files = list_files(path_folder_raw)

save_path = dirto

#iterate over images

PLACL = []
ID = []

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
        lower_white = np.array([190, 190, 190], dtype=np.uint8)  # threshold for white pixels
        upper_white = np.array([255, 255, 255], dtype=np.uint8)
        mask1 = cv2.inRange(blur, lower_white, upper_white)  # could also use threshold
        mask1 = cv2.bitwise_not(mask1)

        # Remove noise
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask1, connectivity=8)
        sizes = stats[1:, -1];
        nb_components = nb_components - 1

        # minimal size of particle
        # somewhere between largest unreal feature and leaf size
        min_size = 600000

        # cleaned mask
        mask_cleaned_seg = np.zeros((output.shape))
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                mask_cleaned_seg[output == i + 1] = 255
        mask_cleaned_seg = np.uint8(mask_cleaned_seg)

        # apply cleaned mask to the image
        [indx, indy] = np.where(mask_cleaned_seg == 0)
        Color_Masked = img.copy()
        Color_Masked[indx, indy] = 255

        # mask for black
        lower_black = np.array([40, 40, 40], dtype=np.uint8)  # threshold for white pixels
        upper_black = np.array([85, 85, 85], dtype=np.uint8)
        mask2 = cv2.inRange(Color_Masked, lower_black, upper_black)  # could also use threshold
        mask2 = cv2.bitwise_not(mask2)

        [indx, indy] = np.where(mask2 == 0)
        Color_Masked = Color_Masked.copy()
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
        # "Near to brown" pixels (i.e. more brown or more green) pixels are also included by defining a wider range;
        # for early scans, define "brown" with more tolerance;
        # many lesions are in development with still significant "green touch" to them;
        # for t3 more restrictive, to protect against effects of physiological senescence;
        # these values were optimised by visual validation on a subset of 30 leaf scans per date
        if "_t2_" in k:
            mask = cv2.inRange(img, np.array([0, 65, 65]), np.array([24, 255, 255]))
        elif "_t3_" in k:
            mask = cv2.inRange(img, np.array([0, 80, 80]), np.array([19, 255, 255]))

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
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(25,25))
        ## Perform morphological opening (i.e. erosion followed by dilation)
        opening = cv2.morphologyEx(mask_cleaned_1, cv2.MORPH_OPEN, kernel)

        # Remove small detected areas; these are most likely not lesions
        ## Get all connected components
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(opening, connectivity=8)

        ## Remove the background, which also counts as component
        sizes = stats[1:, -1];
        nb_components = nb_components - 1

        ## Define minimal lesion size
        min_size = 2000

        ## Remove components smaller than min_size
        mask_cleaned_2 = np.zeros((output.shape))
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                mask_cleaned_2[output == i + 1] = 255
        mask_cleaned_2 = np.uint8(mask_cleaned_2)

        # If there are any lesions, filter them according to
        #  i) the position of their centroid,
        #  ii) the number of detected lesions on the leaf
        # these criteria help to avoid detecting leaf tip necrosis as STB lesions.

        try:

            # Get all connected components
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask_cleaned_2, connectivity=8)

            # Remove background
            nb_components = nb_components - 1

            sizes = stats[1:, -1];

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

            # Drop components if only leaf tip necrosis: If there are only few lesions,
            # if there are no lesions in the left 80% of the leaf, then components are leaf-tip necrosis
            # These thresholds have been optimised in an iterative procedure by evaluating all leaf scans
            x_coords = centroids[0:,0]
            mask_cleaned_3 = mask_cleaned_2.copy()
            if not any(x_coord < 0.8*right_edge for x_coord in x_coords):
                for i in range(0,len(x_coords)+1):
                    mask_cleaned_3[output == i] = 0

            # if centroid of the right-most lesion is close to leaf tip
            # and this right-most lesion has a minimum size;
            # only for t3
            if "_t3_" in k and nb_components <= 5 and comp_size > 80000 and max_r_comp >= 0.8*right_edge:
                mask_cleaned_3[output == comp_id + 1] = 0

            # Find contours
            _, contours, _ = cv2.findContours(mask_cleaned_3, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

            # Draw red contours onto original image
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

            # Draw red contours onto original image
            cnt = cv2.drawContours(Color_Masked, contours, -1, (255,0,0), 2)

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
df.to_csv(dirto, index = False)