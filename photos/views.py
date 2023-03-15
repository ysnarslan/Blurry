from django.core.files.storage import FileSystemStorage
import cv2
import os 
import io
import mtcnn
import json
from django.shortcuts import render, redirect
import numpy as np
from architecture import *
from train_v2 import face_encoding
from scipy.spatial.distance import cosine
from PIL import Image
from django.core.files.uploadedfile import InMemoryUploadedFile
from sklearn.preprocessing import Normalizer


def detect_faces(face_detector, image):
    faces = face_detector.detect_faces(image)
    return faces

def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def get_encode(face_encoder, face, size):
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

def blur_face(img, x1y1, x2y2):
    x1, y1 = x1y1
    x2, y2 = x2y2
    kernel = np.ones((35, 35), np.float32) / (35*35)
    #img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (21, 21), 0)
    img[y1:y2, x1:x2] = cv2.filter2D(img[y1:y2, x1: x2], -1, kernel)
    return img

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

def recognize_faces(image, save_path):

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_detector = mtcnn.MTCNN()
    faces = detect_faces(face_detector, img_rgb)

    recognition_t = 0.55
    confidence_t = 0.95
    required_shape = (160, 160)

    face_encoder = InceptionResNetV2()
    path_m = "facenet_keras_weights.h5"
    face_encoder.load_weights(path_m)
    encodings_path = 'encodings/ben_rdj.json'
    encoding_dict = encoding_dict = eval(load_pickle(encodings_path), {"array": np.array, "float32": np.float32})

    for face in faces:
        if face['confidence'] < confidence_t:
            continue
        face, pt_1, pt_2 = get_face(img_rgb, face['box'])
        encode = get_encode(face_encoder, face, required_shape)
        l2 = l2_normalizer()
        encode = l2.transform(encode.reshape(1, -1))[0]
        name = 'unknown'

        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            print(dist)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist

        if name == 'unknown':
            cv2.rectangle(image, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(image, name + f'__{distance:.2f}', pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # eyes_xy = get_eyes(res)
            # blur_eyes(img, eyes_xy)
            print(pt_1)
            blur_face(image, pt_1, pt_2)

        else:
            cv2.rectangle(image, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(image, name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 200), 2)
            print(name + f'__{distance:.2f}')

    cv2.imwrite(save_path, image)

    PIL_image = Image.open(save_path)

    im_io = io.BytesIO()
    PIL_image.save(im_io, 'JPEG')
    im_io.seek(0)
    image_memory = InMemoryUploadedFile(
        im_io, None, save_path, 'image/jpeg', len(im_io.getvalue()), None
    )

    return image

def blurPhoto(request):
    pathList = []
    select = ""
    if request.method == 'POST':
        data = request.POST
        fs = FileSystemStorage()

        images = request.FILES.getlist('image')
        groupPhoto = request.FILES.getlist('groupPhoto')
        select = request.POST.getlist('filterSelect')

        file = fs.save(str(images[0]), images[0])
        images = fs.url(file)
        encode_photo = cv2.imread("static" + images)
        face_encoding(encode_photo)


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
            recognize_faces(test_image2, save_path)


        if pathList:
            return render(request, "photos/blur.html", {"photos": pathList, "select":select})
    return render(request, 'photos/blur.html', {"photos": pathList,"select":select})
