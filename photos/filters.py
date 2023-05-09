import numpy as np
import cv2

def get_face(box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    return x1, y1, x2, y2


def blur(img, x1y1, x2y2):
    x1, y1 = x1y1
    x2, y2 = x2y2
    half_width = (x2 - x1) // 2
    half_height = (y2 - y1) // 2
    h, w, c = img.shape

    image_mask = np.zeros((h, w), np.uint8)
    image_mask = cv2.ellipse(image_mask, (x2 - half_width, y2 - half_height),
                             (half_width + 10, half_height + 5),
                             0, 0, 360, color=(255, 255, 255), thickness=-1)

    image_mask2 = cv2.ellipse(img.copy(), (x2 - half_width, y2 - half_height),
                             (half_width + 10, half_height + 5),
                             0, 0, 360, color=(255, 255, 255), thickness=-1)


    kernel = np.ones((35, 35), np.float32) / (35 * 35)
    blurred_image = cv2.filter2D(img, -1, kernel)

    mask2 = cv2.bitwise_and(blurred_image, blurred_image, mask=image_mask)

    final_img = image_mask2 + mask2
    return final_img


def pixelate(img, x1y1, x2y2):
    x1, y1 = x1y1
    x2, y2 = x2y2
    half_width = (x2 - x1) // 2
    half_height = (y2 - y1) // 2
    h, w, c = img.shape

    image_mask = np.zeros((h, w), np.uint8)
    image_mask = cv2.ellipse(image_mask, (x2 - half_width, y2 - half_height),
                             (half_width + 5, half_height + 5),
                             0, 0, 360, color=(255, 255, 255), thickness=-1)
    image_mask2 = cv2.ellipse(img.copy(), (x2 - half_width, y2 - half_height),
                              (half_width + 5, half_height + 5),
                              0, 0, 360, color=(255, 255, 255), thickness=-1)


    pixelated_image = np.zeros((h, w, c), np.uint8)
    for x in range(0, h, 10):
        for y in range(0, w, 10):
            pixelated_image[x:x + 10, y:y + 10] = img[x:x + 10, y:y + 10].mean(axis=(0, 1))


    mask2 = cv2.bitwise_and(pixelated_image, pixelated_image, mask=image_mask)

    final_img = mask2 + image_mask2
    return final_img


def blacked_eyes(img, x1y1, x2y2, eyes_xy):
    # if eyes_xy == None:
    #     return img
    kernel = np.ones((5, 5), np.float32)/25

    x1, y1, x2, y2 = eyes_xy
    ara = (x2y2[1] - x1y1[1]) // 10

    if y1 > y2:
        x1 -= ara
        x2 += ara
        y1 += ara
        y2 -= ara
    else:
        x1 -= ara
        x2 += ara
        y1 -= ara
        y2 += ara
    for i in range(5):
        if y1 > y2:
            # img[y2: y1, x1: x2] = cv2.GaussianBlur(img[y2: y1, x1: x2], (5, 5), 0)
            # img[y2: y1, x1: x2] = cv2.filter2D(img[y2: y1, x1: x2], -1, kernel)
            img[y2: y1, x1: x2] = 0
        else:
            # img[y1: y2, x1: x2] = cv2.GaussianBlur(img[y1: y2, x1: x2], (5, 5), 0)
            # img[y1: y2, x1: x2] = cv2.filter2D(img[y1: y2, x1: x2], -1, kernel)
            img[y1: y2, x1: x2] = 0

    return img


def emoji_face(img, x1y1, x2y2, emojiSelect):
    x1, y1 = x1y1
    x2, y2 = x2y2
    emoji_path = "static/photoshare/images/emojis/" + emojiSelect[0] + ".png"

    emoji = cv2.imread(emoji_path)
    #emoji_rgb = cv2.cvtColor(emoji, cv2.COLOR_BGR2RGB)

    width = x2 - x1
    height = y2 - y1

    emoji = cv2.resize(emoji, (width, height))
    img[y1:y2, x1:x2] = blend_non_transparent(img[y1:y2, x1:x2], emoji)
    return img


def blend_non_transparent(face_img, overlay_img):
    # Let's find a mask covering all the non-black (foreground) pixels
    # NB: We need to do this on grayscale version of the image
    gray_overlay = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
    overlay_mask = cv2.threshold(gray_overlay, 254, 255, cv2.THRESH_BINARY_INV)[1]

    # Let's shrink and blur it a little to make the transitions smoother...
    overlay_mask = cv2.erode(overlay_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    overlay_mask = cv2.blur(overlay_mask, (3, 3))

    # And the inverse mask, that covers all the black (background) pixels
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))

# face_detector = mtcnn.MTCNN()
#
# image = cv2.imread("test/chris.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# faces = face_detector.detect_faces(image)
# face = faces[0]
#
# x1, y1, x2, y2 = get_face(face['box'])
# x1 = (x2 - x1) // 2
# y1 = (y2 - y1) // 2
#
# h, w, c = image.shape
# image_mask = np.zeros((h, w), np.uint8)
# image_mask = cv2.ellipse(image_mask, (x2 - x1, y2 - y1), (face["box"][2]//2 + 5, face["box"][3]//2 + 10),
#                          0, 0, 360, color=(255, 255, 255), thickness=-1)
# image_mask2 = cv2.ellipse(image.copy(), (x2 - x1, y2 - y1), (face["box"][2]//2 + 5, face["box"][3]//2 + 10),
#                          0, 0, 360, color=(255, 255, 255), thickness=-1)
#
# cv2.imshow("elips", cv2.cvtColor(image_mask, cv2.COLOR_BGR2RGB))
# cv2.waitKey(0)
#
# kernel = np.ones((35, 35), np.float32) / (35 * 35)
# blurred_image = cv2.filter2D(image, -1, kernel)
# cv2.imshow("blur", cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
# cv2.waitKey(0)
#
# mask2 = cv2.bitwise_and(blurred_image, blurred_image, mask=image_mask)
# cv2.imshow("mask2", cv2.cvtColor(mask2, cv2.COLOR_BGR2RGB))
# cv2.waitKey(0)
#
# final = image_mask2 + mask2
# cv2.imshow("final", cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
# cv2.waitKey(0)


# face_detector = mtcnn.MTCNN()
#
# image = cv2.imread("test/chris.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# faces = face_detector.detect_faces(image)
# face = faces[0]
#
# x1, y1, x2, y2 = get_face(face['box'])
# x1 = (x2 - x1) // 2
# y1 = (y2 - y1) // 2
#
# h, w, c = image.shape
# image_mask = np.zeros((h, w), np.uint8)
# image_mask = cv2.ellipse(image_mask, (x2 - x1, y2 - y1), (face["box"][2]//2 + 5, face["box"][3]//2 + 10),
#                          0, 0, 360, color=(255, 255, 255), thickness=-1)
# image_mask2 = cv2.ellipse(image.copy(), (x2 - x1, y2 - y1), (face["box"][2]//2 + 5, face["box"][3]//2 + 10),
#                          0, 0, 360, color=(255, 255, 255), thickness=-1)
#
# cv2.imshow("elips", cv2.cvtColor(image_mask2, cv2.COLOR_BGR2RGB))
# cv2.waitKey(0)
#
# pixelated_image = np.zeros((h, w, c), np.uint8)
# for x in range(0, h, 10):
#     for y in range(0, w, 10):
#         pixelated_image[x:x+10, y:y+10] = image[x:x+10, y:y+10].mean(axis=(0, 1))
#
# cv2.imshow("pixelate", cv2.cvtColor(pixelated_image, cv2.COLOR_BGR2RGB))
# cv2.waitKey()
#
# mask2 = cv2.bitwise_and(pixelated_image, pixelated_image, mask=image_mask)
# cv2.imshow("mask2", cv2.cvtColor(mask2, cv2.COLOR_BGR2RGB))
# cv2.waitKey(0)
#
# final_img = mask2 + image_mask2
# cv2.imshow("final", cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
# cv2.waitKey(0)

