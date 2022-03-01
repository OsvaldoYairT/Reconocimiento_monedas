# monedas

import cv2
import numpy as np

varlorGaus=3
valorKernel=3
original=cv2.imread('mon.jpg')
gris=cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
gaus=cv2.GaussianBlur(gris,(varlorGaus, varlorGaus), 0)
canny= cv2.Canny(gaus, 60,100)
kernel= np.ones((valorKernel, valorKernel), np.uint8)
cierre=cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

contornos, jerarquia=cv2.findContours(cierre.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("monedas encontradas: {}".format(len(contornos)))
cv2.drawContours(original, contornos, -1, (0,0,255),2)

# mostrar resultados
cv2.imshow('imagen de grises', gris)
cv2.imshow('imagen de gaus', gaus)
cv2.imshow('imagen de canny', canny)
cv2.imshow('cierre', cierre)
cv2.imshow('resultado', original)
cv2.waitKey(0)
#cv2.destroyAllWindows()

