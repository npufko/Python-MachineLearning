import cv2 as cv
import numpy as np
import os
from time import time,sleep
from windowcapture import WindowCapture
from tkinter import *
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from numpy import array, arange, sin, pi
import cv2
import win32api, win32con, win32gui

#training from scratch

import uuid             #unique identifier
import pyautogui
import pydirectinput
from threading import Thread
from vision import Vision as vision

from commons import Commons
#TRAINING
"""
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

IMAGES_PATH = os.path.join('data', 'images') #/data/images
labels = ['enemy', 'friendly', 'bomb', 'molotov', 'smoke']
number_imgs = 20

for label in labels:
    print ('collecting images for {}'.format(labels))
    time.sleep(5)


"""
#LOAD MODEL
"""
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
img = 'https://www.motortrend.com/uploads/sites/25/2019/06/Honda-Super-Meet-LA-Petersen-Automotive-Museum-01.jpg'

results = model(img)
results.print()
"""


#LOAD CUSTOM MODEL
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("starting script... ##############################################################################################################################################################\n")

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp8/weights/last.pt', force_reload=True)


#WORKING LIVE IDENTIFIER
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# initialize the WindowCapture class
wincap = WindowCapture('Counter-Strike: Global Offensive - Direct3D 9')

#initialize the common random gestures
gesture = Commons()

prevCircle = None
dista = lambda x1,y1,x2,y2: (x1-x2)**2+(y1-y2)**2

#initialize Capturing of Minimap

#resets to the bot action state
is_bot_in_action = False
isAiming = False

#sets Target location
targetLocation = [0,0]
enemyVisible = False

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
sqrt3 = np.sqrt(3)
sqrt5 = np.sqrt(5)

mouse_startx = 0
mouse_starty = 0

mouse_devx = 0
mouse_devy = 0

def wind_mouse(start_x, start_y, dest_x, dest_y, G_0=9, W_0=3, M_0=1, D_0=12, move_mouse=lambda x,y: None):
    '''
    WindMouse algorithm. Calls the move_mouse kwarg with each new step.
    Released under the terms of the GPLv3 license.
    G_0 - magnitude of the gravitational fornce
    W_0 - magnitude of the wind force fluctuations
    M_0 - maximum step size (velocity clip threshold)
    D_0 - distance where wind behavior changes from random to damped
    '''
    global enemyVisible
    global targetLocation

    global mouse_startx, mouse_starty, mouse_devx, mouse_devy

    current_x,current_y = start_x,start_y
    v_x = v_y = W_x = W_y = 0
    #while (dist:=np.hypot(dest_x-start_x,dest_y-start_y)) >= 1:
    print("\ndist=",np.hypot(dest_x-start_x,dest_y-start_y))
    dist = np.hypot(dest_x-start_x,dest_y-start_y)
    iteration = 0
    print("iteration",iteration,"destx",dest_x)
    while ((enemyVisible == True) and (iteration <= abs(0.25*(dest_x)))):
        iteration+=1
        target = wincap.get_screen_position(targetLocation)
        print("\nTARGET COORDS ----------", target)
        dest_x=int(target[0])
        dest_y=int(target[1])
        print("\ndist on run",iteration,"---",np.hypot(dest_x-start_x,dest_y-start_y))


        W_mag = min(W_0, dist)
        if dist >= D_0:
            W_x = W_x/sqrt3 + (2*np.random.random()-1)*W_mag/sqrt5
            W_y = W_y/sqrt3 + (2*np.random.random()-1)*W_mag/sqrt5
        else:
            W_x /= sqrt3
            W_y /= sqrt3
            if M_0 < 3:
                M_0 = np.random.random()*3 + 3
            else:
                M_0 /= sqrt5
        v_x += W_x + G_0*(dest_x-start_x)/dist
        v_y += W_y + G_0*(dest_y-start_y)/dist
        v_mag = np.hypot(v_x, v_y)
        if v_mag > M_0:
            v_clip = M_0/2 + np.random.random()*M_0/2
            v_x = (v_x/v_mag) * v_clip
            v_y = (v_y/v_mag) * v_clip
        start_x += v_x
        start_y += v_y
        if np.round(start_x) != 0.0:
            move_x = int(np.round(start_x))
        else:
            move_x = 0
        if np.round(start_y) != 0.0:
            move_y = int(np.round(start_y))
        else: 
            move_y = 0
        if current_x != move_x or current_y != move_y:
            #This should wait for the mouse polling interval
            #move_mouse(current_x:=move_x,current_y:=move_y)
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, move_x,move_y,0,0)
            mouse_devx += move_x
            mouse_devy += move_y
            print("\nmove_x", move_x, "move_y", move_y)
            sleep(0.001)
    return current_x,current_y

def aim_enemy():
    
    # Dynamic Variables
    global targetLocation
    global enemyVisible

    global mouse_devx, mouse_devy
    # targetLocation
    # enemyVisibl

    #mouse move. 0,0 is cuz center of screen is 0
    #second two args are DOWN and RIGHT if positive.

    if (enemyVisible == True):

        target = wincap.get_screen_position(targetLocation)
        #pydirectinput.moveTo(x=target[0], y=target[1])
        #print("target1  =", targets[1])
        x=int(target[0])
        y=int(target[1])

        print("\nAIMING!!! x",x,"y",y,"----------------------------------------------------------------------------------------------------------------------------------------------")

        wind_mouse(0,0, x, y)
    else:
        if (mouse_devx != 0) or (mouse_devy != 0):
            print("Returning mouse to original position.")
            newx = (-1) * mouse_devx
            newy = (-1) * mouse_devy
            wind_mouse(0,0, newx, newy)
        else:
            print("Enemy not visible. Not aiming.")

    global isAiming
    print("\nNo longer aiming.")
    isAiming = False


#class that the thread activates
def bot_actions(rectangles):
    if len(rectangles) > 0:

        #while target in focus is on screen, move mouse to enemy smoothly. then shoot
        # when target x is less than 5 pixels from MOUSE LOCATION

        # if x value is close to 0, move y value up or down depending on situation.
        global enemyVisible
        global isAiming
        global targetLocation
        enemyVisible = True

        #get coords
        targets = vision.get_click_points(rectangles)

        #find target with biggest hitbox
        assignTarget = 0
        newRun = True
        for h in range(len(targets)):
            print("\nTARGET! ", targets[h])   
            if newRun == True:
                newRun = False
                assignTarget = h
                print("\nNew run! assigning target", h)
            if abs((wincap.w/2) - targets[h][0]) < abs((wincap.w/2) - targets[assignTarget][0]):
                print("\nTarget selected! Value of difference:",abs((wincap.w/2) - targets[h][0]))
                assignTarget = h
            else:
                print("\nTarget",h," not selected. Value of difference:", abs((wincap.w/2) - targets[h][0]))
        newRun = True

        #find target closest to crosshair
        # tbd

        #set global target location for Thread2
        targetLocation = targets[assignTarget]
        print("\nTarget assigned to AIM, targetLocation =", targetLocation)

        # check if crosshair is on target, shoot if so!

        #set target for below code
        target = wincap.get_screen_position(targetLocation)
        #pydirectinput.moveTo(x=target[0], y=target[1])
        #print("target1  =", targets[1])
        x=int(target[0])
        y=int(target[1])

        if enemyVisible == True:
            if (abs(x) < 30) and (abs(y) < 30):
                pydirectinput.click()
                print("\nSHOTS FIRED, SLEEPING 1.5")
                
        """
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x,y,0,0)
        """

    global is_bot_in_action
    is_bot_in_action = False

loop_time = time()
while(True):

    # get an updated image of the game
    screenshot = wincap.get_screenshot()

    #isolate for MiniMap
    screen_array = np.array(screenshot)

    #crop region for minimap
    #cropped_region = screen_array[80:300, 110:330, :]
    cropped_region = screen_array[120:290, 160:330, :]
    #cropped_region_target = (cropped_region, cv.COLOR_BGR2RGB)


    #results for main window
    results = model(screenshot) #results

    #prints results just in case
    print(results.pandas().xywh[0])

    #Build [ if results = valid, then do computations ]

    #only add targets that are above 80% confidence
    numValidTargets = 0
    notValid = []
    
    for p in range(len(results.pandas().xywh[0])):
        if (results.pandas().xywh[0]['confidence'][p]) > (0.80):
            numValidTargets+=1
            notValid.append(p)
            #print("Valid Targets =", numValidTargets)

    rectangles = [[0 for i in range(4)] for j in range(numValidTargets)]

    for i in range(len(results.pandas().xywh[0])):
        if(results.pandas().xywh[0]['confidence'][i]) < (0.80): #ADD IF ITS ENEMY
            print("\nNOT A TARGET! Ignoring row",i)
        else:
            #rectangles[i] = results.pandas().xywh[0].iloc[:,[0,1,2,3]]
            rectangles[i][0] = results.pandas().xywh[0]['xcenter'][i]
            #print("rectangles[i][0]= ",rectangles[i][0])
            rectangles[i][0] = int(rectangles[i][0])
            rectangles[i][1] = results.pandas().xywh[0]['ycenter'][i]
            rectangles[i][1] = int(rectangles[i][1])
            rectangles[i][2] = results.pandas().xywh[0]['width'][i]
            rectangles[i][2] = int(rectangles[i][2])
            rectangles[i][3] = results.pandas().xywh[0]['height'][i]
            rectangles[i][3] = int(rectangles[i][3])


    #RENDER MODEL
    cv.imshow('Computer Vision', np.squeeze(results.render()))

    #render minimap
    cv.imshow('MiniMap', cropped_region)

    cropped_region = cv.resize(cropped_region, (0,0), fx = 3, fy = 3)

    grayFrame = cv.cvtColor(cropped_region, cv.COLOR_BGR2GRAY)
    blurFrame = cv.GaussianBlur(grayFrame, (17,17), 0)

    circles = cv.HoughCircles(blurFrame, cv.HOUGH_GRADIENT, 1.2, 100, param1=100,
        param2=10, minRadius=0, maxRadius=50)
    #param1 = higher equals less sensitive
    #param2 = accuracy of circle, edgepoints

    if circles is not None:
        circles = np.uint16(np.around(circles))
        chosen = None
        for i in circles[0,:]:
            if chosen is None: chosen = i
            if prevCircle is not None:
                if dista(chosen[0], chosen[1], prevCircle[0], prevCircle[1]) <= dista(i[0],i[1],prevCircle[0],
                    prevCircle[1]):
                    chosen = i

        cv.circle(cropped_region, (chosen[0], chosen[1]), 1, (0,100,100), 3)
        cv.circle(cropped_region, (chosen[0], chosen[1]), chosen[2], (255,0,255), 3)
        prevCircle = chosen

    cv.imshow("circles", cropped_region)



    cv.imshow('outpuit', cropped_region)



    if numValidTargets > 0:
        if not is_bot_in_action:
            print("\nStarting targeting thread.")
            is_bot_in_action = True
            t = Thread(target=bot_actions, args=(rectangles,))
            t.start()
        if not isAiming:
            print("\nStarting aim thread.")
            isAiming = True
            s = Thread(target=aim_enemy)
            s.start()
    else:
        enemyVisible = False
        #gesture.knife_Inspect_1()
        #gesture.shittalk()


    # debug the loop rate
    print('\nFPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        enemyVisible = False
        is_bot_in_action = False
        break

print('\nDone.')
#-----------------------------------------------------------------------------------

