from django.core.files.storage import FileSystemStorage
import cv2
import os 
import json
from django.shortcuts import render, redirect
import numpy as np
from architecture import *
from train_v2 import face_encoding
from scipy.spatial.distance import cosine
from PIL import Image
from sklearn.preprocessing import Normalizer
from photos.filters import blur, pixelate, blacked_eyes, emoji_face


def detect_faces(face_detector, image):
    faces = face_detector.detect(image)
    return faces

def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def get_encode(face_encoder, face):
    face = normalize(face)
    if (face.shape[0] > face.shape[1]):
        scale = face.shape[0] / 160
        width = int(face.shape[1] / scale)
        face = cv2.resize(face, (width, 160))
        padding = int((160 - face.shape[1]) / 2)

        if (160 - face.shape[1]) % 2 == 0:
            face = cv2.copyMakeBorder(face, 0, 0, padding, padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:
            face = cv2.copyMakeBorder(face, 0, 0, padding, padding + 1, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    else:
        scale = face.shape[1] / 160
        height = int(face.shape[0] / scale)
        face = cv2.resize(face, (160, height))
        padding = int((160 - face.shape[0]) / 2)

        if (160 - face.shape[0]) % 2 == 0:
            face = cv2.copyMakeBorder(face, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:
            face = cv2.copyMakeBorder(face, padding, padding + 1, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode

def hide_face(img, x1y1, x2y2, eyes_xy, blur_mod, emojiSelect):
    if blur_mod == "blurFace": #pixelFace, blackFace, emojiFace
        return blur(img, x1y1, x2y2)

    elif blur_mod == "pixelFace":
        return pixelate(img, x1y1, x2y2)

    elif blur_mod == "blackFace":
        return blacked_eyes(img, x1y1, x2y2, eyes_xy)

    elif blur_mod == "emojiFace":
        return emoji_face(img, x1y1, x2y2, emojiSelect)

def load_pickle(path="encodings/rdj.json"):
    with open(path, 'r') as f:
        pyresponse = json.loads(f.read())

    return pyresponse

def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

def l2_normalizer():
    l2_normalizer = Normalizer('l2')
    return l2_normalizer

def recognize_faces(image, save_path, encode_name, blur_mod, emojiSelect, process):

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_detector = cv2.FaceDetectorYN.create(
        model="weights/face_detection_yunet_2022mar.onnx",  # yunet.onnx face_detection_yunet_2022mar
        config='',
        input_size=(480, 640),
        score_threshold=0.6,
        nms_threshold=0.35,
        top_k=5000,
        backend_id=3,
        target_id=0
    )
    face_detector.setInputSize((image.shape[1], image.shape[0]))
    _, faces = detect_faces(face_detector, img_rgb)

    recognition_t = 0.65
    confidence_t = 0.6
    #required_shape = (160, 160)

    face_encoder = InceptionResNetV2()
    path_m = "weights/facenet_keras_weights.h5"
    face_encoder.load_weights(path_m)
    encodings_path = f'encodings/{encode_name}.json'
    encoding_dict = eval(load_pickle(encodings_path), {"array": np.array, "float32": np.float32})


    for face in faces:
        # if face[-1] < confidence_t:
        #     continue
        face = face[:-1].astype(np.int32)
        img_face, pt_1, pt_2 = get_face(img_rgb, face[:4])
        # cv2.imshow("faces", img_face)
        # cv2.waitKey(0)

        encode = get_encode(face_encoder, img_face)
        l2 = l2_normalizer()
        encode = l2.transform(encode.reshape(1, -1))[0]
        name = 'unknown'

        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            print(db_name, dist)
            if dist <= recognition_t:
                name = db_name

        if process == "Me":
            if name == 'unknown':
                eyes_xy = (face[4], face[5]) + (face[6], face[7])
                image = hide_face(image, pt_1, pt_2, eyes_xy, blur_mod, emojiSelect)

        elif process == "Other":
            if name != 'unknown':
                eyes_xy = (face[4], face[5]) + (face[6], face[7])
                image = hide_face(image, pt_1, pt_2, eyes_xy, blur_mod, emojiSelect)


    cv2.imwrite(save_path, image)

    PIL_image = Image.open(save_path)

    return PIL_image

def blurPhoto(request):
    pathList = []
    select = ""
    if request.method == 'POST':
        data = request.POST
        fs = FileSystemStorage()

        procress = request.POST.get('procress')
        images = request.FILES.getlist('image')
        groupPhoto = request.FILES.getlist('groupPhoto')
        select = request.POST.getlist('filterSelect')
        emojiSelect = request.POST.getlist('emojiSelect')

        image_names = []
        encode_name = ""
        for image in images:
            file = fs.save(str(image), image)
            image_names.append(file)
            encode_name += file + "_"


        face_encoding(image_names)


        for image in groupPhoto:
            file = fs.save(str(image), image)
            img = fs.url(file)

            name = "processed_" + img.split("/")[-1]
            save_path = os.path.join('static/images/', name)
            test_image2 = cv2.imread("static" + img)

            try:
                if not test_image2: continue
            except: pass
            pathList.append(name)
            recognize_faces(test_image2, save_path, encode_name, select[0], emojiSelect, procress)


        if pathList:
            return render(request, "photos/blur.html", {"photos": pathList, "select":select})
    return render(request, 'photos/blur.html', {"photos": pathList,"select":select})
