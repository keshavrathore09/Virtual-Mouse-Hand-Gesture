import cv2 #engine
import numpy as np
import time
import pyautogui
import math
import os
import mediapipe as mp #brain for hand pattern
last_click_time = 0 # To track last click time for debouncing
scroll_smooth = 0 # Adjust this value to make scrolling faster or slower
palm_time = 0  # To track when the palm was last seen
pyautogui.FAILSAFE = False #screen will not stop even if mouse at corner
pyautogui.PAUSE = 0 #to remove auto 0.1sec delay of pyautogui
is_dragging = False #for drag initial false
pScrollY = 0 # Previous Y specifically for scrolling
cap=cv2.VideoCapture(0) #camera obj

#setting fix camera resolution as original camera res may be too high
cap.set(3,640) #3 is width id
cap.set(4,480) #4 is height id

mpHands=mp.solutions.hands #calling hand section of mediapipe #alias
#above the solution is subfolder in mp and hands is the hands detection logic inside that

#creating an obj hands by calling the mphands 
#Hands is a class

hands = mpHands.Hands(static_image_mode=False,  #treat input as video stream
                      max_num_hands=1,  #to track only one hand
                      min_detection_confidence=0.7, #AI must be at least 50% sure it sees a hand before it starts tracking.|| 
                                                    #if 0.9 very strict || 0.1 very less strict can detect shadow
                      min_tracking_confidence=0.7) #tracking the detected hand confidence

mpDraw = mp.solutions.drawing_utils #to visualize 21 landmark and draw circles on landmark
wScr, hScr = pyautogui.size() # get Mac's screen width and height
# print(wScr, hScr)

# Smoothing setup
frameR = 60        # Frame Reduction: This is our margin for the "Active Area"
smoothening = 4     # Higher value = more smooth but slower cursor
plocX, plocY = 0, 0 # Previous locations
clocX, clocY = 0, 0 # Current locations

last_click_time = 0 #tracking last click time

while True:

    success, img=cap.read() #succees is bool frame grabbed or not || img is numpy array of pixels

    if not success: break # Security check

    #flip image
    img=cv2.flip(img,1) #1nis horizontal flip
    #bgr to rgb for mediapipe
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 
    results=hands.process(imgRGB) #contain all mathematical data for 21 coordinates

    # Drawing the Active Area box on the screen
    cv2.rectangle(img, (frameR, frameR), (640 - frameR, 480 - frameR), (255, 0, 255), 2)

    if results.multi_hand_landmarks: #if we hide hand it will not work
        for handLms in results.multi_hand_landmarks: #although one hand but mediapipe return a list
            #img - we draw the original bgr image
            #handLms - 21 dots coordinates
            #mpHands.HAND_CONNECTIONS - to coonect dots with lines
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS) 

            lmList = []

            #mediapipe coordinate 0-1 to pixel multiply
            for id, lm in enumerate(handLms.landmark): #This loops through all 21 points. id is the number (0-20) and lm contains the $(x, y, z)$ data
                h, w, c = img.shape #height width channel for current webcam(3 for bgr)
                cx, cy = int(lm.x * w), int(lm.y * h) #converting 0 1 to pixel coordinates
                lmList.append([id, cx, cy])

            if len(lmList) != 0:
                #Check which fingers are up
                fingers = []

                # Thumb logic
                # if the tip (4) is to the left of the joint (3), it’s considered "open"
                if lmList[4][1] < lmList[3][1]: fingers.append(1) # 4 , 3 is id
                else: fingers.append(0)

                # 4 Fingers logic
                for id in [8, 12, 16, 20]:
                    if lmList[id][2] < lmList[id - 2][2]: fingers.append(1) #y axis start top left
                    else: fingers.append(0)
                
                # ---------------------------------------------------------
                # MASTER QUIT GESTURE (Palm -> Fist)
                # ---------------------------------------------------------
                if all(f == 1 for f in fingers): # Palm detected
                    palm_time = time.time()
                    cv2.putText(img, "QUIT READY: MAKE FIST", (200, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                elif all(f == 0 for f in fingers): # Fist detected
                    if time.time() - palm_time < 1.0:
                        print("Force Quitting...")
                        cap.release()
                        cv2.destroyAllWindows()
                        os._exit(0) # This kills the program instantly

                # VOLUME MODE (Trigger: ONLY Pinky and Thumb open - 'Call Me' gesture)
                elif fingers[0] == 1 and fingers[4] == 1 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0:
                    current_y = lmList[20][2] # Using Pinky tip
                    cv2.circle(img, (lmList[20][1], lmList[20][2]), 15, (0, 255, 255), cv2.FILLED)
                    
                    current_time = time.time()
                    if current_time - last_click_time > 0.1:
                        # Map the Y position of your hand to Volume 0-100
                        # frameR at top = 100% volume, 480-frameR at bottom = 0% volume
                        if current_y < pScrollY - 5:
                            pyautogui.press("volumeup")
                        elif current_y > pScrollY + 5:
                            pyautogui.press("volumedown")

                        pScrollY = current_y

                # Movement Mode
                # We check: Is Index Up (1) AND Middle Down (0)? so cursor move
                elif fingers[1] == 1 and fingers[2] == 0 and fingers[0] == 0: 
                    
                    # Get the current Index Tip coordinates from our list
                    cx, cy = lmList[8][1], lmList[8][2]
                    
                    #Map Coordinates with the Active Area (frameR)
                    #We subtracted frameR from the boundaries (0+100 to 640-100)

                    # Convert finger x (0-640) to screen x (0-wScr) || Linear Interpolation.
                    #without frameR | x3 = np.interp(cx, (0, 640), (0, wScr)) #numpy fn
                    x3 = np.interp(cx, (frameR, 640 - frameR), (0, wScr))

                    # Convert finger y (0-480) to screen y (0-hScr)
                    #without frameR | y3 = np.interp(cy, (0, 480), (0, hScr)) 
                    y3 = np.interp(cy, (frameR, 480 - frameR), (0, hScr))

                    # Limit the values to screen boundaries
                    # This ensures x3 is never less than 0 or more than screen width
                    x3 = np.clip(x3, 0, wScr)
                    y3 = np.clip(y3, 0, hScr)

                    #Smoothen Values
                    #Logic: New Location = Old Location + (Current Finger Position - Old Location) / Smoothing_Factor
                    clocX = plocX + (x3 - plocX) / smoothening #(LERP) formula
                    clocY = plocY + (y3 - plocY) / smoothening

                    #Move Mouse
                    #before smoothing -> pyautogui.moveTo(x3, y3)
                    pyautogui.moveTo(clocX, clocY) #after smoothing

                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED) #purple color code bgr , 15 is rad of circle | filled for filling circle

                # LEFT CLICK : Thumb + Index Pinch
                elif fingers[1] == 1 and fingers[2] == 0 and fingers[0]==1: # thumb index up
                    # Find distance between Thumb (4) and Index (8)
                    x1, y1 = lmList[4][1], lmList[4][2] # Thumb Tip
                    x2, y2 = lmList[8][1], lmList[8][2] # Index Tip
                    
                    # Calculate middle point for visual feedback
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    #movement
                    x3 = np.interp(x2, (frameR, 640 - frameR), (0, wScr))
                    y3 = np.interp(y2, (frameR, 480 - frameR), (0, hScr))
                    clocX = plocX + (x3 - plocX) / smoothening
                    clocY = plocY + (y3 - plocY) / smoothening
                    pyautogui.moveTo(clocX, clocY)
                    
                    # Math: Length of the line between points

                    length = math.hypot(x2 - x1, y2 - y1) #euclidian distance under root x2-x1 sq - y2-y1 sq
                    
                    # B. If distance is short, perform click
                    if length < 25: #25 pixel

                        cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED) # Green for click
                        current_time = time.time()
                        if current_time - last_click_time > 0.4:
                            pyautogui.click() # Instant click 
                            last_click_time = current_time

                    else:
                        cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED) # Blue circle when preparing to click
                
                # DEDICATED DRAG (Thumb + Middle)
                elif fingers[2] == 1 and fingers[0] == 1 and fingers[1] == 1 and fingers[3] == 0 and fingers[4] == 0:
                    x1, y1 = lmList[4][1], lmList[4][2] # Thumb
                    x2, y2 = lmList[12][1], lmList[12][2] # Middle
                    
                    x3 = np.interp(x2, (frameR, 640 - frameR), (0, wScr))
                    y3 = np.interp(y2, (frameR, 480 - frameR), (0, hScr))
                    clocX = plocX + (x3 - plocX) / smoothening
                    clocY = plocY + (y3 - plocY) / smoothening
                    pyautogui.moveTo(clocX, clocY)

                    length = math.hypot(x2-x1,y2-y1)
                    if length < 18:
                        cv2.circle(img, (x2, y2), 15, (255, 165, 0), cv2.FILLED) # Orange
                        if not is_dragging:
                            pyautogui.mouseDown()
                            is_dragging = True
                            time.sleep(0.15)
                    
                    else:
                        if is_dragging:
                            pyautogui.mouseUp()
                            is_dragging = False
                        # Regular movement feedback
                        cv2.circle(img, (x2, y2), 15, (0, 255, 255), cv2.FILLED) # Yellow

                # RIGHT CLICK- Index + Middle Parallel/Touching
                elif fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 0 and fingers[3] == 0:
                    x1, y1 = lmList[8][1], lmList[8][2]
                    x2, y2 = lmList[12][1], lmList[12][2]
                    
                    length = math.hypot(x2 - x1, y2 - y1)
                    
                    # Move cursor first
                    x3 = np.interp(x1, (frameR, 640 - frameR), (0, wScr))
                    y3 = np.interp(y1, (frameR, 480 - frameR), (0, hScr))
                    pyautogui.moveTo(x3, y3)

                    if length < 35:  # increased threshold
                        cv2.circle(img, (x1, y1), 15, (0, 0, 255), cv2.FILLED)
                        
                        current_time = time.time()
                        if current_time - last_click_time > 0.7:
                            pyautogui.click(button='right')
                            last_click_time = current_time
                    else:
                        cv2.circle(img, (x1, y1), 15, (0, 255, 255), cv2.FILLED)
                    
                # SCROLLING MODE: Index, Middle, and Ring fingers UP 
                elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 0 and fingers[0] == 0:
                    # Getting the Y-coordinate of the Index tip (ID 8)
                    current_y = lmList[8][2]
                    
                    # Visual feedback: Drawing yellow circles on the three active fingers
                    for id in [8, 12, 16]:
                        cv2.circle(img, (lmList[id][1], lmList[id][2]), 15, (0, 255, 255), cv2.FILLED)

                    # LOGIC: Check the vertical movement of the hand
                    # We use a 20-pixel buffer to avoid "jittery" scrolling
                    """if pScrollY > current_y + 15: #The + 20 and - 20(pixel buffer) ensure the scroll only triggers when we make a clear up or down
                        pyautogui.scroll(5)  # Scroll Up (Positive)
                        pScrollY = current_y     # Reset plocY so it doesn't scroll infinitely
                    elif pScrollY < current_y - 15:
                        pyautogui.scroll(-5) # Scroll Down (Negative)
                        pScrollY = current_y"""
                    # Calculate movement distance
                    delta_y = pScrollY - current_y

                    if abs(delta_y) > 6:
                        scroll_amount = int(np.clip(delta_y * 4, -150, 150))
                        pyautogui.scroll(scroll_amount)

                    pScrollY = current_y
                else:
                     # to keep the scroll reference updated when you aren't scrolling
                    pScrollY = lmList[8][2]
                # Updating previous location for the next frame
                plocX, plocY = clocX, clocY

                #print(f"Index Finger is at: {cx}, {cy}") # to print x, y coordinates of finger to check if any error
    #show frame window
    cv2.imshow("Hand Tracking Feed", img) #not updated img to show as monitor accepts bgr

    #to exit
    if cv2.waitKey(1) & 0xFF==ord('q'): #0xFF is bitwise mask to filter mac command noise and cut it to 8 bit
        break

#clean up
cap.release()
cv2.destroyAllWindows()