import numpy as np
import cv2, argparse, parser
from math import *

WRITETOFILE='file'
RETURNVIDEO='object'



def process_video_name(FileName: str, Function):
    print(f'Attempting to read {FileName}')
    Video = cv2.VideoCapture(FileName)
    print('File read')

    FrameCount = int(Video.get(cv2.CAP_PROP_FRAME_COUNT))
    FrameWidth = int(Video.get(cv2.CAP_PROP_FRAME_WIDTH))
    FrameHeight = int(Video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(FrameCount, FrameWidth, FrameHeight)
    #VideoBuffer = np.empty((FrameCount, FrameHeight, FrameWidth, 3), np.dtype('uint8'))
    #print(VideoBuffer.shape)
    ret = True
    #cv2.namedWindow('Video frames')
    #print('Iterating through frames and adding to 3d matrix')
    # for Frame in FrameCount:
    #     for x in FrameWidth:
    #         for y in FrameHeight:
    #             Video.set()
    #             yield
    # Function: x+y
    cv2.namedWindow('In progress video frame')
    #cv2.namedWindow('Source Video Frame')
    #TODO: Find a way to make this faster. Current efficiency is 1 however it can now be multi-processed or threaded
    for FramePos in range(FrameCount):
        OutFrame = np.empty((int(FrameHeight), int(FrameWidth), 3), np.dtype('uint8'))
        for y in range(FrameHeight):
            # IDEA: Cache a line of time frames to speed up processing (Have a call for every x not every y)
            print('Calculating TemporalLine')
            TemporalLine = np.empty((int(FrameCount), int(FrameWidth), 3), np.dtype('uint8'))
            Video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for Pos in range(FrameCount):
                ret, Frame = Video.read()
                if ret:
                    TemporalLine[Pos] = Frame[y]
            for x in range(FrameWidth):
                # TODO: Test if eval can be abused with Function
                FResult = eval(Function)
                print(f'Reading frame {int((FramePos+FResult)%(FrameCount))}')
                #Video.set(cv2.CAP_PROP_POS_FRAMES, ((FramePos+FResult)%(FrameCount)))
                #ret, Frame = Video.read()

                OutFrame[y][x] = TemporalLine[int((FramePos+FResult)%(FrameCount))][x]
                #cv2.imshow('Source Video Frame', Frame)
                print(f'{FramePos}({x},{y}) = {FResult}')
                print(f'{TemporalLine[int((FramePos+FResult)%(FrameCount))][x]}')
                cv2.imshow('In progress video frame', OutFrame)
                # if(ret):
                #     print(f'{FramePos}({x},{y}) = {FResult}')
                #     print(f'{Frame[y][x]}')
                #     OutFrame[y][x] = Frame[y][x]
                #     cv2.imshow('In progress video frame', OutFrame)
                cv2.waitKey(1)
        yield OutFrame
    Video.release()



# def process_video_object(VideoObject: cv2.VideoCapture):
#     return
#
#
# def generate_video(VideoMatrix, RETURNFORMAT=WRITETOFILE):
#     return


def parseArgs():
    parser = argparse.ArgumentParser(description='Process video files by a f(x,y) to alter playback')
    parser.add_argument('inputfile', type=str, help='The file you would like to submit for processing')
    parser.add_argument('-f', '--function', type=str, default='0', help='A function expression that maps x, y into a value to distort t. f(x, y) = x+y would be written as x+y')
    parser.add_argument('-o', '--output', type=str, default=None,help='The output file location')
    return parser.parse_args()


def main(inputfile, function, output):
    # Parse the function: str into a mathematical statement
    function = parser.expr(function).compile()
    for frame in process_video_name(inputfile, function):
        cv2.namedWindow('Finished Video frames')
        cv2.imshow('Finished Video Frames', frame)


if __name__ == '__main__':
    args = parseArgs()
    main(args.inputfile, args.function, args.output)
