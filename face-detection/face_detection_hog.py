import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

def get_differential_filter():
    # Using sobel filter (Double Flipped version for direct element-wise multiplication)
    filter_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    filter_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return filter_x, filter_y

def filter_image(im, filter):
    # Adding padding to the edges
    padding_n = (np.floor(filter.shape[0]/2)).astype(int)
    padded_im = np.pad(im, pad_width=padding_n, mode='constant', constant_values=0)

    im_filtered = np.zeros(((im.shape[0]+(2*padding_n)-filter.shape[0]+1), (im.shape[1]+(2*padding_n)-filter.shape[1]+1)))

    for i in range(0, padded_im.shape[0]-filter.shape[0]+1):
        for j in range(0, padded_im.shape[1]-filter.shape[1]+1):
            filter_view_im = padded_im[i:i+filter.shape[0], j:j+filter.shape[1]]
            filtered_view = np.sum(np.multiply(filter_view_im, filter))
            im_filtered[i][j] = filtered_view

    return im_filtered

def get_gradient(im_dx, im_dy):
    if (im_dx.shape != im_dy.shape):
        raise ValueError("The shape of differentials should be same")

    grad_mag = np.zeros(shape=im_dx.shape)
    grad_angle = np.zeros(shape=im_dx.shape)

    for i in range(0, im_dx.shape[0]):
        for j in range(0, im_dx.shape[1]):
            # Calculating magnitude (sqrt(dx^2 + dy^2))
            grad_mag[i][j] = np.sqrt(im_dx[i][j] ** 2 + im_dy[i][j] ** 2)
            # Calculating tan-1(dy/dx)
            grad_angle[i][j] = math.degrees(np.arctan(np.divide(im_dy[i][j],im_dx[i][j])))
            # Converting the negative angles to be in range [0, pi)
            if grad_angle[i][j] < 0:
                grad_angle[i][j] = grad_angle[i][j] + 180
    
    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):
    
    if grad_mag.shape != grad_angle.shape:
        raise Exception("Both Grad Angles and Grad Magnitudes should have the same shape")

    no_of_bins = 6
    
    cells_x = math.floor(grad_mag.shape[0] / cell_size)
    cells_y = math.floor(grad_mag.shape[1] / cell_size)

    ori_histo = np.zeros(shape=(cells_x, cells_y, no_of_bins))

    for m in range(0, cells_x):
        for n in range(0, cells_y):
            mags_in_cell = grad_mag[(m * cell_size):((m+1) * cell_size), (n * cell_size):((n+1) * cell_size)]
            angles_in_cell = grad_angle[(m * cell_size):((m+1) * cell_size), (n * cell_size):((n+1) * cell_size)]

            for i in range(0, mags_in_cell.shape[0]):
                for j in range(0, mags_in_cell.shape[1]):
                    # Calculating the bins by deviating the angle by 15 degrees
                    # So, effectively any angle between 0 to 15 and any angle between 165 and 180 has a modulo of less than 30 making it bin 0
                    # Rest of the values will have a modulo greater than 30 and will fall into respective bins.
                    bin_val = math.floor(((np.nan_to_num(angles_in_cell[i][j]) + 15) % 180) / 30)
                    # adding magnitude to the respective bins
                    ori_histo[m][n][bin_val] = ori_histo[m][n][bin_val] + mags_in_cell[i][j]
    
    return ori_histo


def get_block_descriptor(ori_histo, block_size=8):

    # normalization constant 
    e = 0.001

    # number of bins are always 6
    no_of_bins = ori_histo.shape[2]

    ori_histo_normalized = np.zeros(shape=(ori_histo.shape[0]-block_size+1, ori_histo.shape[1]-block_size+1, (no_of_bins * (block_size ** 2))))

    for i in range(ori_histo.shape[0]-block_size+1):
        for j in range(ori_histo.shape[1]-block_size+1):
            histo_in_block = ori_histo[i:i+block_size, j:j+block_size, :]
            histo_descriptor = histo_in_block.flatten(order='C')
            denom = np.sqrt(np.sum(histo_descriptor ** 2) + (e ** 2))
            # block normalized descriptors 
            ori_histo_normalized[i][j] = histo_descriptor/np.array(denom)

    return ori_histo_normalized


def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0

    filter_x, filter_y = get_differential_filter()

    im_dx = filter_image(im, filter_x)
    im_dy = filter_image(im, filter_y)

    grad_mag, grad_angle = get_gradient(im_dx, im_dy)
    ori_histo = build_histogram(grad_mag, grad_angle, 8)

    hog = get_block_descriptor(ori_histo, 2)

    ## NOTE: Will display HOGs for each and every window
    # # visualize to verify
    # visualize_hog(im, hog, 8, 2)

    return hog


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()

def face_recognition(I_target, I_template):

    correlation_threshold = 0.42

    # Obtaining the hog of template image and calculating the normalized desriptor and magnitude
    template_hog = extract_hog(I_template).flatten()
    norm_template_descr = template_hog - np.mean(template_hog)
    norm_template_mag = np.sqrt(np.sum(norm_template_descr ** 2))

    all_bounding_boxes = []

    for y in range(0, I_target.shape[0]-I_template.shape[0]+1, 4):
        for x in range(0, I_target.shape[1]-I_template.shape[1]+1, 4):
            target_hog = extract_hog(I_target[y:y+I_template.shape[0], x:x+I_template.shape[1]]).flatten()
            norm_target_descr = target_hog - np.mean(target_hog)
            norm_target_mag = np.sqrt(np.sum(norm_target_descr ** 2))
            all_bounding_boxes.append([x, y, (np.dot(norm_target_descr, norm_template_descr)/(norm_template_mag * norm_target_mag))])

    all_bounding_boxes = np.array(all_bounding_boxes)

    # Only considering the bounding boxes which have correlation greater than the threshold    
    possible_bb = all_bounding_boxes[(all_bounding_boxes[:, 2] > correlation_threshold)]

    # Obtaining bounding boxes after non maximum supression
    template_size = I_template.shape[0]

    iou_threshold = 0.5

    area_of_boxes = template_size * template_size

    bounding_boxes = []
    indices = np.arange(len(possible_bb))

    # making a copy of the bounding boxes list
    bb_set = np.copy(possible_bb)

    while (len(bb_set) != 0):
        # finding the index of the bounding box whose correlation is the maximum
        max_bb_index = np.argmax(bb_set, axis=0)[2]
        max_bb_coords = bb_set[max_bb_index][0:2]
        # adding the bounding box with maximum value to the list of final bounding boxes
        bounding_boxes.append(bb_set[max_bb_index])

        # Finding the length of intersections on x and y coordinates
        x_diff = np.minimum(max_bb_coords[0]+template_size, bb_set[:, 0]+template_size) - np.maximum(max_bb_coords[0], bb_set[:, 0])
        y_diff = np.minimum(max_bb_coords[1]+template_size, bb_set[:, 1]+template_size) - np.maximum(max_bb_coords[1], bb_set[:, 1])

        # intersection area
        intersection_areas = x_diff * y_diff

        # union = box1 + box2 - intersection
        # Given that the area of bounding box 1 and 2 are the same we are multiplying with 2
        union_areas = ((2 * area_of_boxes) - intersection_areas)

        # iou = intersection / union
        iou = np.divide(intersection_areas, union_areas)
        
        # we need to discard the iou greater than the threshold, so we will only retain the ones which are less than or equal to threshold
        # getting the indices of the bounding boxes which will be retained
        bbs = iou <= iou_threshold
        # only keeping the indices of the eligible bounding boxes
        indices = indices[bbs]
        # updating the bounding boxes set with the remaining ones
        # Given that we are removing all the bounding boxes whose iou is greater than the threshold, the max. bounding box (box in picture) will also be removed
        # as we are supposed to do after adding it to the list of final bounding boxes
        bb_set = possible_bb[indices]

    bounding_boxes = np.array(bounding_boxes)

    return bounding_boxes


def visualize_face_detection(I_target,bounding_boxes,box_size):

    hh,ww,cc=I_target.shape

    fimg=I_target.copy()
    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii,0]
        x2 = bounding_boxes[ii, 0] + box_size 
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1<0:
            x1=0
        if x1>ww-1:
            x1=ww-1
        if x2<0:
            x2=0
        if x2>ww-1:
            x2=ww-1
        if y1<0:
            y1=0
        if y1>hh-1:
            y1=hh-1
        if y2<0:
            y2=0
        if y2>hh-1:
            y2=hh-1
        fimg = cv2.rectangle(fimg, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f"%bounding_boxes[ii,2], (int(x1)+1, int(y1)+2), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 0), 1, cv2.LINE_AA)


    plt.figure(3)
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.show()


if __name__=='__main__':

    im = cv2.imread('./face-detection/assets/cameraman.tif', 0)
    hog = extract_hog(im)

    I_target= cv2.imread('./face-detection/assets/target.png', 0)
    #MxN image

    I_template = cv2.imread('./face-detection/assets/template.png', 0)
    #mxn  face template

    bounding_boxes=face_recognition(I_target, I_template)

    I_target_c= cv2.imread('./face-detection/assets/target.png')
    # MxN image (just for visualization)
    visualize_face_detection(I_target_c, bounding_boxes, I_template.shape[0])
    #this is visualization code.
