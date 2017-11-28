#coding: utf-8

import cv2

# Haar-like特徴分類器読み込み
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

# イメージファイル読み込み
face_img = cv2.imread('face3.jpg')

# グレースケールに変換する
gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

# 顔検出
faces = face_cascade.detectMultiScale(gray)
for (x,y,w,h) in faces:
    # 検知した顔を矩形で囲む
    cv2.rectangle(face_img, (x,y),(x+w,y+h),(255,0,0),2)

    # 顔画像（グレースケール）
    roi_gray = gray[y:y+h, x:x+w]
    # 顔画像（カラースケール）
    roi_color = face_img[y:y+h, x:x+w]

    # 顔の中から目を検出
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        # 検知した顔を矩形で囲む
        cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)

# 画像表示
cv2.imshow('OpenCV face recognition', face_img)

# 何かキーを押したら終了
cv2.waitKey(0)
cv2.destroyAllWindows()