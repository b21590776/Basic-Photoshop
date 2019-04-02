import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

# masks
mask1 = cv2.imread("images/mask1.png")
mask2 = cv2.imread("images/mask2.png")
mask3 = cv2.imread("images/mask3.png")
mask4 = cv2.imread("images/mask4.png")
mask5 = cv2.imread("images/mask5.png")

# original images
im1 = cv2.imread("images/im1.jpg")
im2 = cv2.imread("images/im2.jpg")
im3 = cv2.imread("images/im3.jpg")
im4 = cv2.imread("images/im4.jpg")
im5 = cv2.imread("images/im5.jpg")

# backgrounds
bg1 = cv2.imread("images/bg1.jpg")
bg2 = cv2.imread("images/bg2.jpg")
bg3 = cv2.imread("images/bg3.jpg")
bg4 = cv2.imread("images/bg4.jpeg")
bg5 = cv2.imread("images/bg5.jpeg")

# --------------------------------------------------------------------------------------

def bgr2rgb(im):

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # image normalization
    return (im - np.min(im)) / (np.max(im) - np.min(im))


im1 = bgr2rgb(im1)
im2 = bgr2rgb(im2)
im3 = bgr2rgb(im3)
im4 = bgr2rgb(im4)
im5 = bgr2rgb(im5)

bg1 = bgr2rgb(bg1)
bg2 = bgr2rgb(bg2)
bg3 = bgr2rgb(bg3)
bg4 = bgr2rgb(bg4)
bg5 = bgr2rgb(bg5)


# --------------------------------------------------------------------------------------


def Gaussian(size, sigma):
    # gaussian kernel implementation
    # sigma is full-width-half-maximum(effective radius).
    # size is the matrix size of gaussian kernel

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x1 = y1 = size // 2

    return np.exp(-4 * np.log(2) * ((x - x1) ** 2 + (y - y1) ** 2) / sigma ** 2)


# size 5x5 and sigma 5
gausskernel = Gaussian(5, 5)
#print(gauskernel)

# ---------------------------------------------------------------------------------------

# for getting portrait mode: foreground sharpening and  background blurring

def backgroundblurring(bg):
    # i take only background
    # convolution of each signal with gauss kernel
    gausIm = np.empty(bg.shape)
    for i in range(3):
        gausIm[:, :, i] = signal.convolve(bg[:, :, i], gausskernel, mode='same')

    # image normalization
    gausIm = (gausIm - np.min(gausIm)) / (np.max(gausIm) - np.min(gausIm))

    plt.imshow(gausIm), plt.title('Blur Background')
    plt.show()

    return gausIm


bg1 = backgroundblurring(bg1)
bg2 = backgroundblurring(bg2)
bg3 = backgroundblurring(bg3)
bg4 = backgroundblurring(bg4)
bg5 = backgroundblurring(bg5)

# ---------------------------------------------------------------------------------------

def foregroundsharpening(im,alpha):

    # original image sharpening

    # convolution of each signal with gauss kernel
    gausIm = np.empty(im.shape)
    for i in range(3):
        gausIm[:, :, i] = signal.convolve(im[:, :, i], gausskernel, mode='same')
    # image normalization
    gausIm = (gausIm - np.min(gausIm)) / (np.max(gausIm) - np.min(gausIm))

    # to get the sharp image firstly i create image with edges =  original image subtract blur image

    edg = im - gausIm

    # than multiply by alpha add original image creates sharped image
    sharp = im + edg * alpha
    plt.imshow(sharp), plt.title('sharp Image')
    plt.show()
    return sharp


im1 = foregroundsharpening(im1, 3)
im2 = foregroundsharpening(im2, 3)
im3 = foregroundsharpening(im3, 3)
im4 = foregroundsharpening(im4, 3)
im5 = foregroundsharpening(im5, 3)

# -----------------------------------------------------------------------------------------
# getting together foreground and background with blur background and sharp original image

def part1(im, mask, bg, rowshift, columnshift, picnumber):

    # with np.where i extract the black pixels(0) of the mask and puts the original image to non blacks pixels of mask
    foreground = np.where(mask, im, 0)

    # for shifting the foreground getting mask's rows and columns
    maskrow, maskcolumn , a = mask.shape
    for i in range(maskrow):
        for j in range(maskcolumn):
            if mask[i, j].all() == 1:  # getting the white(1) pixels
                bg[i + rowshift, j + columnshift] = foreground[i, j]   # shift foreground and put to background

    plt.imshow(bg), plt.title('portrait mode image')
    plt.show()
    plt.imshow(bg)
    plt.savefig('Part2/bg' + str(picnumber) + '.jpg')
    return bg

# adjusting the location of the each foregrounds with using rowshift, columnshift


PhotoshoppedImage1 = part1(im1, mask1, bg1, 10, 100, 1)
PhotoshoppedImage2 = part1(im2, mask2, bg2, 40, 130, 2)
PhotoshoppedImage3 = part1(im3, mask3, bg3, 237, 350, 3)
PhotoshoppedImage4 = part1(im4, mask4, bg4, 15, 180, 4)
PhotoshoppedImage5 = part1(im5, mask5, bg5, 65, 140, 5)
