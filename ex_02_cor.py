import cv2
import numpy as np

cv2.namedWindow('imagem', cv2.WINDOW_NORMAL)
cv2.resizeWindow('imagem', (600, 400))
vid = cv2.VideoCapture(0)

while 1:
    _, imagem = vid.read()
    imagem_hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(imagem_hsv, 
                       np.array([0, 0, 0]), 
                       np.array([179, 255, 100]))
    imagem = cv2.bitwise_and(imagem, imagem, mask=mask)

    cv2.imshow('imagem', imagem)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break