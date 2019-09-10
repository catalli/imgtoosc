import sys
import os
import wave
import math
import struct
import random
import argparse
from itertools import *
from PIL import Image

if len(sys.argv) < 4:
    fps = 20
else:
    fps = int(sys.argv[3])

def downsample_to_proportion(rows, proportion=1):
    return list(islice(rows, 0, len(rows), int(1/proportion)))

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


# main code
im = Image.open(sys.argv[1])
pix = im.load()
divisor = float(max(im.size[0]/2, im.size[1]/2))
xzero = im.size[0]/2
yzero = im.size[1]/2
values=[]
for x in range(0, im.size[0]):
    if x % 2 == 0:
        yrange = range(0, im.size[1])
    else:
        yrange = range(im.size[1]-1,-1,-1)
    for y in yrange:
        if pix[x,y][3] == 255 and pix[x,y][0]+pix[x,y][1]+pix[x,y][2] <= int(sys.argv[2]):
            values.append((float((x-xzero))/divisor,float(-(y-yzero))/divisor))

downvalues = downsample_to_proportion(values, float(384000)/float(len(values)*fps))
sampleimage = cycle(downvalues)

nchannels=2
sampwidth=2
framerate=len(downvalues)*fps
nframes=framerate

samples = islice(sampleimage, nframes)

write_wavefile("".join([sys.argv[1],".scan.wav"]), samples, nframes,nchannels,sampwidth,framerate)
