import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

WINDOWSIZEX = 640
WINDOWSIZEY = 480
image_cnt = 1
BOUNDARY = 5 # pixels
WHITE = (255, 255, 255)
BLACK = (0,0,0)
RED = (255,0,0)

PREDICT = True

IMAGE_SAVE = False

MODEL = load_model("bestmodel.h5")

LABELS = {
    0: "Zero",
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine"
}

pygame.init()
FONT = pygame.font.Font(pygame.font.get_default_font(), 18)
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption('Digit Board')
#WHILE_INIT = 
iswriting = False

number_xcord = []
number_ycord = []

while True:
    for event in pygame.event.get(): # get any event from mouse, keyboard etc...
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0) # 4 thickness, 0 brightness
            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            # for rectange coordinates
            rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDARY, 0), min(WINDOWSIZEX, number_xcord[-1]+BOUNDARY)
            rect_min_y, rect_max_y = max(number_ycord[0] - BOUNDARY, 0), min(WINDOWSIZEY, number_ycord[-1]+BOUNDARY)

            number_xcord = []
            number_ycord = []

            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)
            if img_arr.any():
                cv2.imwrite('image.png', img_arr)
                image_cnt +=1

            if PREDICT:
                image = cv2.resize(img_arr, (28,28))
                image = np.pad(image, (10,10), 'constant', constant_values=0)
                image = cv2.resize(image, (28,28))/255

                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1,28,28,1)))])

                text_surf = FONT.render(label, True, RED, WHITE)
                text_rect_obj = text_surf.get_rect()
                text_rect_obj.left, text_rect_obj.bottom = rect_min_x, rect_max_y

                DISPLAYSURF.blit(text_surf, text_rect_obj)


        if event.type == KEYDOWN:
            DISPLAYSURF.fill(BLACK)

    pygame.display.update()
            


