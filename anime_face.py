#coding: utf-8

import os
import cv2

# Haar-like特徴分類器読み込み
aniface_cascade = cv2.CascadeClassifier('haarcascades/lbpcascade_animeface.xml')

# イメージファイル読み込み
face_img = cv2.imread('anime_face/anime_img3.bmp')

# グレースケールに変換する
gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

# 顔検出
faces = aniface_cascade.detectMultiScale(gray)

# print(faces)

# 画像出力ディレクトリを作成
out_dir = 'output_faces'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

#for i, (x, y, w, h) in enumerate(faces):
    # 一人ずつ顔を切り抜く
#    out_img = face_img[y:y+h, x:x+w]
#    out_path = os.path.join(out_dir, '{0}.png'.format(i))
#    cv2.imwrite(out_path, out_img)

# cv2.imwrite('face.jpg', face_img)


for (x,y,w,h) in faces:
    # 検知した顔を矩形で囲む
    cv2.rectangle(face_img, (x,y),(x+w,y+h),(0,0,255),3)

# 画像表示
cv2.imshow('OpenCV face recognition', face_img)

# 何かキーを押したら終了
cv2.waitKey(0)
cv2.destroyAllWindows()