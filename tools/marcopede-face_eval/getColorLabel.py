#!/usr/bin/env python
import pylab
import numpy as np
import os
cl_count = 5
#color_list = pylab.cm.Set1(np.linspace(0, 1, 26))
color_list = [
    '#FFB300', # Vivid Yellow
    '#803E75', # Strong Purple
    '#FF6800', # Vivid Orange
    '#A6BDD7', # Very Light Blue
    '#C10020', # Vivid Red
    '#CEA262', # Grayish Yellow
    '#817066', # Medium Gray

    # The following don't work well for people with defective color vision
    '#007D34', # Vivid Green
    '#F6768E', # Strong Purplish Pink
    '#00538A', # Strong Blue
    '#FF7A5C', # Strong Yellowish Pink
    '#53377A', # Strong Violet
    '#FF8E00', # Vivid Orange Yellow
    '#B32851', # Strong Purplish Red
    '#F4C800', # Vivid Greenish Yellow
    '#7F180D', # Strong Reddish Brown
    '#93AA00', # Vivid Yellowish Green
    '#593315', # Deep Yellowish Brown
    '#F13A13', # Vivid Reddish Orange
    '#232C16', # Dark Olive Green
    ]

def getColorLabel(name):


    #print (name)
    global cl_count, color_list

    if name.find("SquaresChnFtrs") != -1 or name.find("Baseline") != -1:
        color = color_list[cl_count]
        
        label = "SquaresChnFtrs-5"
    if os.path.splitext(os.path.basename(name))[0]=="FM CNN":
        color = color_list[4]
        label = "FM-CNN"
    elif name.find("Fast FM CNN")!=-1:
        color = color_list[2]
        label = "Fast FM-CCNN"
    elif name.find("Faster R-CNN") != -1:
        color = color_list[1]
        label = "Faster R-CNN"
    elif name.find("Faceness") !=-1:
        color = color_list[0]
        label = "Faceness"
    elif name.find("HyperFace") !=-1:
        color = color_list[3]
        label = "HyperFace"

    elif name.find("Headhunter") != -1 or name.find("HeadHunter") != -1:
        color = color_list[7]
        label = "HeadHunter"
    elif name.find("Face++") != -1:
        color = color_list[cl_count]
        cl_count+= 1
        label = "Face++"
    elif name.find("Picasa") != -1:
        color = color_list[cl_count]
        cl_count+= 1
        label = "Picasa"
    elif name.find("Structured") != -1:
        color = color_list[cl_count]
        cl_count+= 1
        label = "Structured Models [33]"
    elif name.find("WS_Boosting") != -1:
        color = color_list[cl_count]
        cl_count+= 1
        label = "W.S. Boosting [14]"
    elif name.find("Sky") != -1:
        color = color_list[cl_count]
        cl_count+= 1
        label = "Sky Biometry [28]"
    elif name.find("OpenCV") != -1:
        color = color_list[cl_count]
        cl_count+= 1
        label = "OpenCV"
    elif name.find("TSM") != -1:
        color = color_list[cl_count]
        cl_count+= 1
        label = "TSM [36]"
    elif name.find("DPM") != -1 or name.find("<0.3") != -1:
        color = color_list[cl_count]
        cl_count+= 1
        #color = 'b'
        label = "DPM [9]"
    elif name.find("Shen") != -1:
        color = color_list[cl_count]
        cl_count+= 1
        #color = 'b'
        label = "Shen et al. [27]"
    elif name.find("Viola") != -1:
        color = color_list[cl_count]
        cl_count+= 1
        #color = 'b'
        label = "Viola Jones [30]"
    else:
        color = color_list[cl_count]
        cl_count+= 1
        label = os.path.splitext(os.path.basename(name))[0]
        label = label.replace("_", " ")
    return [color, label]
