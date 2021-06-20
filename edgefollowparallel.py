import sys
import os
import wave
import math
import struct
import random
import argparse
from itertools import *
import cv2

# write_wavefile and write_pcm functions from https://zach.se/generate-audio-with-python/ 

def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

def write_wavefile(filename, samples, nframes=None, nchannels=2, sampwidth=2, framerate=44100, bufsize=2048):
    # Write samples to a wavefile.
    if nframes is None:
        nframes = -1

    w = wave.open(filename, 'w')
    w.setparams((nchannels, sampwidth, framerate, nframes, 'NONE', 'not compressed'))

    max_amplitude = float(int((2 ** (sampwidth * 8)) / 2) - 1)

    # split the samples into chunks (to reduce memory consumption and improve performance)
    for chunk in grouper(bufsize, samples):
        frames = b"".join(b"".join(struct.pack('h', int(max_amplitude * sample)) for sample in channels) for channels in chunk if channels is not None)
        w.writeframesraw(frames)

    w.close()

    return filename

def write_pcm(f, samples, sampwidth=2, framerate=44100, bufsize=2048):
    # Write samples as raw PCM data."
    max_amplitude = float(int((2 ** (sampwidth * 8)) / 2) - 1)

    # split the samples into chunks (to reduce memory consumption and improve performance)i
    for chunk in grouper(bufsize, samples):
        frames = b"".join(b"".join(struct.pack('h', int(max_amplitude * sample)) for sample in channels) for channels in chunk if channels is not None)
        f.write(frames)

    f.close()

    return filename

def selectnextpoint(cv_img, center_point, pathstepradius, threshold):
    pointToReturn = None
    pickedPoint = False
    # search in a spiral converging on the center point
    focuspoint = [center_point[0]-pathstepradius,center_point[1]-pathstepradius]
    xboundupper = center_point[0]+pathstepradius
    xboundlower = center_point[0]-pathstepradius
    yboundupper = center_point[1]+pathstepradius
    yboundlower = center_point[1]-pathstepradius
    direction = 'xincrease'
    nopoints = (2*pathstepradius)**2-1
    while focuspoint != center_point and nopoints > 0:
        try:
            if cv_img[focuspoint[0],focuspoint[1]] <= threshold and (not pickedPoint) and (focuspoint[0] >=0 and focuspoint[1] >= 0):
                pointToReturn = focuspoint
                pickedPoint = True
                break
        except:
            pass
        if direction == 'xincrease':
            if focuspoint[0]+1 > xboundupper:
                direction = 'yincrease'
                yboundlower+=1
            else:
                focuspoint[0]+=1
                nopoints-=1
        if direction == 'yincrease':
            if focuspoint[1]+1 > yboundupper:
                direction = 'xdecrease'
                xboundupper-=1
            else:
                focuspoint[1]+=1
                nopoints-=1
        if direction == 'xdecrease':
            if focuspoint[0]-1 < xboundlower:
                direction = 'ydecrease'
                yboundupper-=1
            else:
                focuspoint[0]-=1
                nopoints-=1
        if direction == 'ydecrease':
            if focuspoint[1]-1 < yboundlower:
                direction = 'xincrease'
                xboundlower+=1
            else:
                focuspoint[1]-=1
                nopoints-=1
    if not pickedPoint:
        return None
    return pointToReturn

def follow_path(cv_img, pathstepradius,pathlengthmax,threshold,patheraseradius):
    emptyImage = True
    for i in range(cv_img.shape[0]):
        for j in range(cv_img.shape[1]):
            if cv_img[i,j] <= threshold:
                emptyImage = False
                startPoint = [i,j]
                cv_img[startPoint[0],startPoint[1]]=255
                break
        if not emptyImage:
            break
    if emptyImage:
        return []
    values = [startPoint]
    for i in range(pathlengthmax):
        erasesquare(cv_img, values[len(values)-1],patheraseradius)
        nextPoint = selectnextpoint(cv_img,values[len(values)-1],pathstepradius,threshold)
        if nextPoint is None:
            break
        values.append(nextPoint)
    return values

def erasesquare(cv_img, centerpoint, eraseradius):
    for i in range(centerpoint[0]-eraseradius,centerpoint[0]+eraseradius):
        for j in range(centerpoint[1]-eraseradius,centerpoint[1]+eraseradius):
            if i>=0 and j>= 0:
                try:
                    cv_img[i,j] = 255
                except:
                    pass

filename = sys.argv[1]
threshold = 50
try:
    pathstepradius = sys.argv[2]
except:
    pathstepradius = 5
try:
    patheraseradius = sys.argv[3]
except:
    patheraseradius = 5
try:
    pathlengthmax = sys.argv[4]
except:
    pathlengthmax = 256
try:
    maxside = sys.argv[5]
except:
    maxside = 400
try:
    fps = sys.argv[6]
except:
    fps = 60
try:
    lengthseconds=sys.argv[7]
except:
    lengthseconds=10

img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
trans_mask = img[:,:,3] == 0
img[trans_mask] = [255, 255, 255, 255]
largest_dim = max(img.shape[0],img.shape[1])
ratio = maxside/largest_dim
tgt_width = int(img.shape[1]*ratio)
tgt_height = int(img.shape[0]*ratio)
tgt_dim = (tgt_width,tgt_height)

resized = cv2.resize(img, tgt_dim ,interpolation = cv2.INTER_CUBIC)
img_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

paths = []
cv2.namedWindow("image",0)
while True:
    pathToAdd = follow_path(img_gray, pathstepradius, pathlengthmax, threshold,patheraseradius)
    cv2.imshow("image", img_gray)
    cv2.waitKey(1)
    if pathToAdd == []:
        break
    else:
        paths.append(pathToAdd)

cv2.destroyAllWindows()

print(len(paths))
#paths.sort(key = lambda x: len(x),reverse=True)
print(len(paths[0]))
print(len(paths[-1]))
print(paths[0])
print(paths[-1])

divisor = float(max(img_gray.shape[0],img_gray.shape[1]))/2
xzero = img_gray.shape[0]/2
yzero = img_gray.shape[1]/2

masterwaveform = []

# generate waveform from paths here
## interleave paths
pointsremaining = True
while pointsremaining:
    pointsremaining = False
    for path in paths:
        if len(path) > 0:
            pointsremaining = True
            point = path.pop(0)
            normpoint = (float(point[1]-yzero)/divisor,-float(point[0]-xzero)/divisor)
            masterwaveform.append(normpoint)

# end waveform generation

sampleimage = cycle(masterwaveform)

nchannels=2
sampwidth=2
framerate = min(44100,len(masterwaveform)*fps)
nframes = len(masterwaveform)*fps*lengthseconds

samples = islice(sampleimage,nframes)

write_wavefile("".join([sys.argv[1],".pathparallel.wav"]), samples, nframes,nchannels,sampwidth,framerate)
