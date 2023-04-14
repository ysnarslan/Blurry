import numpy as np
import mtcnn
# from cv2 import filter2D
import cv2

def blur(img, x1y1, x2y2):
    x1, y1 = x1y1
    x2, y2 = x2y2
    kernel = np.ones((35, 35), np.float32) / (35 * 35)
    # img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (21, 21), 0)
    img[y1:y2, x1:x2] = cv2.filter2D(img[y1:y2, x1: x2], -1, kernel)
    return img


def pixelate(img, x1y1, x2y2):
    x1, y1 = x1y1
    x2, y2 = x2y2
    # n, m = n - n % 30, m - m % 30
    for x in range(y1, y2 + 1, 10):
        for y in range(x1, x2, 10):
            img[x:x+10, y:y+10] = img[x:x+10, y:y+10].mean(axis=(0, 1))
    return img.astype(np.uint8)


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
    #emoji_path = "static/photoshare/images/angelFace.png"
    emoji = cv2.imread(emoji_path)
    #emoji_rgb = cv2.cvtColor(emoji, cv2.COLOR_BGR2RGB)
    #cv2.imshow("dnm", emoji)
    #cv2.waitKey()
    print(x1, y1, x2, y2)
    width = x2 - x1
    height = y2 - y1

    emoji = cv2.resize(emoji, (width, height))
    # cv2.imshow("dnm", img[x1:x2, y1:y2])
    # cv2.waitKey()
    # cv2.imshow("dnm", emoji[0:100, 0:500])
    # cv2.waitKey()

    print(emoji.shape, img[x1:x2, y1:y2].shape)

    # img = cv2.imread("static/images/rdj1.jpg")

    img[y1:y2, x1:x2] = blend_non_transparent(img[y1:y2, x1:x2], emoji)
    # cv2.imshow("dnm", img)
    # cv2.waitKey(0)

    return img

# overlay_img = cv2.imread("static/photoshare/images/coolFace.png")
#
# cv2.imshow("dnm", overlay_mask)
# cv2.waitKey()

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

# image = cv2.imread("static/images/rdj1.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# face_detector = mtcnn.MTCNN()
# faces = face_detector.detect_faces(image)
# eyes_xy = []
# type(faces[0]["keypoints"]["left_eye"])
# type(eyes_xy)
# type(faces[0])
# faces[0]["keypoints"]

# for face in faces:
#     print(i, ".", [face["keypoints"]["left_eye"], face["keypoints"]["right_eye"]])
#     # eyes_xy.append([face["keypoints"]["left_eye"], face["keypoints"]["right_eye"]])
#     i += 1
#     cv2.rectangle(image, face["keypoints"]["left_eye"], face["keypoints"]["right_eye"], (0, 0, 255), 2)
#
# cv2.imshow("eye", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)
#
# for face in faces:
#     print(face)


# if (face["keypoints"]["left_eye"] & face["keypoints"]["right_eye"]) else None
