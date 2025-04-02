import numpy as np
#==============No additional imports allowed ================================#

def get_ncc_descriptors(img, patchsize):
    '''
    Prepare normalized patch vectors for normalized cross
    correlation.

    Input:
        img -- height x width x channels image of type float32
        patchsize -- integer width and height of NCC patch region.
    Output:
        normalized -- height* width *(channels * patchsize**2) array

    For every pixel (i,j) in the image, your code should:
    (1) take a patchsize x patchsize window around the pixel,
    (2) compute and subtract the mean for every channel
    (3) flatten it into a single vector
    (4) normalize the vector by dividing by its L2 norm
    (5) store it in the (i,j)th location in the output

    If the window extends past the image boundary, zero out the descriptor
    
    If the norm of the vector is <1e-6 before normalizing, zero out the vector.

    '''
    x, y, c = img.shape
    offset = patchsize // 2

    normalized = np.zeros((x, y, c * patchsize**2), dtype=np.float32)

    for i in range(x):
        if (i - offset < 0) | (i + offset >= x):
            normalized[i, :, :] = np.zeros_like(normalized[i, :, :])
            continue
        for j in range(y):
            if (j - offset < 0) | (j + offset >= y):
                normalized[i, j, :] = np.zeros_like(normalized[i, j, :])
                continue
            patch = img[i - offset : i + offset + 1, j - offset : j + offset + 1, :].copy()
            mean = np.mean(patch, axis=(0, 1))
            patch -= mean
            patch_flat = patch.flatten()
            norm = np.linalg.norm(patch_flat)
            if norm > 1e-6:
                patch_flat /= norm
            else:
                patch_flat = np.zeros_like(patch_flat)
            normalized[i, j, :] = patch_flat
    return normalized
    


def compute_ncc_vol(img_right, img_left, patchsize, dmax):
    '''
    Compute the NCC-based cost volume
    Input:
        img_right: the right image, H x W x C
        img_left: the left image, H x W x C
        patchsize: the patchsize for NCC, integer
        dmax: maximum disparity
    Output:
        ncc_vol: A dmax x H x W tensor of scores.

    ncc_vol(d,i,j) should give a score for the (i,j)th pixel for disparity d. 
    This score should be obtained by computing the similarity (dot product)
    between the patch centered at (i,j) in the right image and the patch centered
    at (i, j+d) in the left image.

    Your code should call get_ncc_descriptors to compute the descriptors once.
    '''
    left_ncc = get_ncc_descriptors(img_left, patchsize)
    right_ncc = get_ncc_descriptors(img_right, patchsize)
    h, w, _ = img_left.shape
    ncc_vol = np.zeros((dmax, h, w), dtype=np.float32)
    for d in range(dmax):
        for i in range(h):
            for j in range(w):
                if (j + d) >= w:
                    ncc_vol[d, i, j] = 0
                    continue
                left_patch = left_ncc[i, j + d, :]
                right_patch = right_ncc[i, j, :]
                ncc_vol[d, i, j] = np.dot(left_patch, right_patch)
    return ncc_vol

def get_disparity(ncc_vol):
    '''
    Get disparity from the NCC-based cost volume
    Input: 
        ncc_vol: A dmax X H X W tensor of scores
    Output:
        disparity: A H x W array that gives the disparity for each pixel. 

    the chosen disparity for each pixel should be the one with the largest score for that pixel
    '''
    disparity = np.argmax(ncc_vol, axis=0)
    return disparity





    
