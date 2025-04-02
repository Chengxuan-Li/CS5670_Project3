import numpy as np
##======================== No additional imports allowed ====================================##






def photometric_stereo_singlechannel(I, L):
    #L is 3 x k
    #I is k x n (k lights, n pixels)
    G = np.linalg.inv(L @ L.T) @ L @ I
    # G is  3 x n 
    albedo = np.sqrt(np.sum(G*G, axis=0))

    normals = G/(albedo.reshape((1,-1)) + (albedo==0).astype(float).reshape((1,-1)))
    return albedo, normals


def photometric_stereo(images, lights):
    '''
        Use photometric stereo to compute albedos and normals
        Input:
            images: A list of N images, each a numpy float array of size H x W x 3
            lights: 3 x N array of lighting directions. 
        Output:
            albedo, normals
            albedo: H x W x 3 array of albedo for each pixel
            normals: H x W x 3 array of normal vectors for each pixel

        Assume light intensity is 1.
        Compute the albedo and normals for red, green and blue channels separately.
        The normals should be approximately the same for all channels, so average the three sets
        and renormalize so that they are unit norm

    '''
    h, w, c = images[0].shape
    _, n_lights = lights.shape

    albedo = np.zeros((h, w, c))
    normals = np.zeros((h, w, c))
    
    for channel in range(c):
        I = np.zeros((n_lights, h*w))
        for i in range(n_lights):
            I[i] = images[i][:, :, channel].flatten()
        L = lights
        channel_albedo, channel_normals = photometric_stereo_singlechannel(I, L)
        albedo[:, :, channel] = channel_albedo.reshape((h, w))
        normals[:, :, :] += channel_normals.T.reshape((h, w, 3)) / c
        
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)

    return albedo, normals



