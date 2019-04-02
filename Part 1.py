import cv2
import numpy as np

# original images
im1 = cv2.imread("images/im1.jpg")
im2 = cv2.imread("images/im2.jpg")
im3 = cv2.imread("images/im3.jpg")
im4 = cv2.imread("images/im4.jpg")
im5 = cv2.imread("images/im5.jpg")

# masks
mask1 = cv2.imread("images/mask1.png")
mask2 = cv2.imread("images/mask2.png")
mask3 = cv2.imread("images/mask3.png")
mask4 = cv2.imread("images/mask4.png")
mask5 = cv2.imread("images/mask5.png")

# backgrounds
bg1 = cv2.imread("images/bg1.jpg")
bg2 = cv2.imread("images/bg2.jpg")
bg3 = cv2.imread("images/bg3.jpg")
bg4 = cv2.imread("images/bg4.jpeg")
bg5 = cv2.imread("images/bg5.jpeg")


def part1(im, mask, bg, rowshift, columnshift, picnumber):

    # with np.where i extract the black pixels(0) of the mask and puts the original image to non blacks pixels of mask
    foreground = np.where(mask, im, 0)

    # for shifting the foreground getting mask's rows and columns
    maskrow, maskcolumn , a = mask.shape
    for i in range(maskrow):
        for j in range(maskcolumn):
            if mask[i,j].all() == 1:  # getting the white(1) pixels
                bg[i + rowshift, j + columnshift] = foreground[i, j]  # shift foreground and put to background
    cv2.imshow("original image", im)
    cv2.imshow("mask", mask)
    cv2.imshow("foregroundimage.jpg", foreground)
    cv2.imshow("Photoshopped Image", bg)
    cv2.waitKey(0)
    cv2.imwrite("Part1/PhotoshoppedImage"+str(picnumber)+".jpg", bg)
    return bg

# adjusting the location of the each foregrounds with using rowshift, columnshift


PhotoshoppedImage1 = part1(im1, mask1, bg1, 10, 100, 1)
PhotoshoppedImage2 = part1(im2, mask2, bg2, 40, 130, 2)
PhotoshoppedImage3 = part1(im3, mask3, bg3, 237, 350, 3)
PhotoshoppedImage4 = part1(im4, mask4, bg4, 15, 180, 4)
PhotoshoppedImage5 = part1(im5, mask5, bg5, 65, 140, 5)


