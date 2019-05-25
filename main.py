'''
    Written by @Adel bennaceur
    21/12/2018
'''

import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)

while(cap.isOpened()):

    ret, img = cap.read()
    cv2.rectangle(img,(300,300),(0,0),(0,255,0),0)
    roi = img[0:300, 0:300]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    value = (35, 35)
    # applying gaussian blur
    blurred = cv2.GaussianBlur(gray, value, 0)

    # thresholdin: Otsu's  method

    _, thresh1 = cv2.threshold(blurred, 127, 255,
                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imshow('Thresholded', thresh1)
    contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, \
            cv2.CHAIN_APPROX_NONE)
    max_area = -1

    # find contour with max area
    for i in range(len(contours)):
        cn = contours[i]
        area = cv2.contourArea(cn)
        if(area>max_area):
            max_area = area
            ci=i
    cn = contours[ci]

    # create bounding rectangle around the contour (can skip below two lines)
    x,y,w,h = cv2.boundingRect(cn)
    cv2.rectangle(roi,(x,y),(x+w,y+h),(0,0,255),0)

    # finding convex hull
    hull = cv2.convexHull(cn)

    # draw contours
    draw = np.zeros(roi.shape,np.uint8)
    cv2.drawContours(draw,[cn],0,(0,255,0),0)
    cv2.drawContours(draw,[hull],0,(0,0,255),0)

    # finding convex hull
    hull = cv2.convexHull(cn,returnPoints = False)

    # finding convexity defects
    defects = cv2.convexityDefects(cn,hull)
    count_defects = 0
    cv2.drawContours(thresh1, contours, -1, (0,255,0), 3)

    # applying cos rule to find angle for all defects (between fingers)
    # with angle > 90 degrees and ignore defects
    for i in range(defects.shape[0]):

        s,e,f,d = defects[i,0]
        start = tuple(cn[s][0])
        end = tuple(cn[e][0])
        far = tuple(cn[f][0])

        # find length of all sides of triangle
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

        # apply cosine rule here
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

        # ignore angles > 90 and highlight rest with red dots
        if angle <= 90:
            count_defects += 1
            cv2.circle(roi,far,2,[0,0,255],-1)

        cv2.line(roi,start,end,[0,255,0],2)
        #cv2.circle(roi,far,3,[0,0,255],-1)

    # define actions required
    if count_defects == 0:
        cv2.putText(img,"1", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    if count_defects == 1:
        cv2.putText(img,"2", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 2:
        cv2.putText(img, "3", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 3:
        cv2.putText(img,"4", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 4:
        cv2.putText(img,"5", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

    #cv2.imshow('draw', draw)
    #cv2.imshow('end', roi)

    # show appropriate images in windows
    cv2.imshow('Final result', img)
    all_img = np.hstack((draw, roi))
    cv2.imshow('Contours', all_img)
    k = cv2.waitKey(10)
    if k == 27:
        break
