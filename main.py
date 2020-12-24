import numpy as np
import cv2, argparse, parser, multiprocessing
from UMatFileVideoStream import UMatFileVideoStream
from math import *

WRITETOFILE='file'
RETURNVIDEO='object'

# TODO: Refactor threading into saying processing

def process_video_name(FileName: str, Function, ThreadCount: int):
    print(f'Attempting to read {FileName}')
    Video = cv2.VideoCapture(FileName)
    print('File read')

    FrameCount = int(Video.get(cv2.CAP_PROP_FRAME_COUNT))
    FrameWidth = int(Video.get(cv2.CAP_PROP_FRAME_WIDTH))
    FrameHeight = int(Video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Video.release()
    print(FrameCount, FrameWidth, FrameHeight)

    #TODO: Find a way to make this faster. Current efficiency is 2 It has multi processing but is still fairly cpu intensive
    # The next possible move could be to have this run on a gpu
    for FramePos in range(FrameCount):
        OutFrame = np.empty((int(FrameHeight), int(FrameWidth), 3), np.dtype('uint8'))


        with multiprocessing.Pool(ThreadCount) as p:
            p.map(process_frame, [((int(FrameHeight/ThreadCount) * i, int(FrameHeight/ThreadCount) * (i+1)), FrameHeight, FrameWidth, Function, FramePos, FrameCount, FileName, OutFrame, f'temp{i}.npy') for i in range(0, ThreadCount)])
        for i in range(ThreadCount):
            OutFrame = np.bitwise_xor(OutFrame, np.load(f'temp{i}.npy'))
            #TODO: Clean up after process
        yield OutFrame

def test_proc(FrameHeight, FrameWidth, FrameCount, FramePos, Function, TemporalLine, x, y):
    FResult = eval(Function)
    #print(x)
    # print(f'Reading frame {int((FramePos+FResult)%(FrameCount))}')
    # print(f'({x}, {y})')
    # print(TemporalLine[int((FramePos+FResult)%(FrameCount))][x])
    return TemporalLine[int((FramePos+FResult)%(FrameCount))][x]


def process_frame(specialTuple):
    yBounds, FrameHeight, FrameWidth, Function, FramePos, FrameCount, VideoName, OutFrame, TempOut = specialTuple
    yStart, yEnd = yBounds
    print(yBounds)
    # Video = cv2.VideoCapture(VideoName)
    # Video = UMatFileVideoStream(VideoName, 128).start()
    for y in range(yStart, yEnd):
            # IDEA: Cache a line of time frames to speed up processing (Have a call for every x not every y)
            print('Calculating TemporalLine')
            TemporalLine = np.empty((int(FrameCount), int(FrameWidth), 3), np.dtype('uint8'))
            # Video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            Video = UMatFileVideoStream(VideoName, 128).start()
            rgb= cv2.UMat(FrameHeight, FrameWidth, cv2.CV_8UC3)
            for Pos in range(FrameCount):
                Frame = Video.read()
                # cv2.cvtColor(Video.read(), cv2.COLOR_BGR2RGB, Frame, 0)
                Frame = cv2.UMat.get(Frame)
                if Frame.any():
                    TemporalLine[Pos] = Frame[y]
            # for x in range(FrameWidth):
                # TODO: Test if eval can be abused with Function
            # FResult = eval(Function)
            # print(f'Reading frame {int((FramePos+FResult)%(FrameCount))}')

            # Extract from temporalline based on time function
            print(OutFrame.shape)
            testWrapper = lambda x: test_proc(FrameHeight, FrameWidth, FrameCount, FramePos, Function, TemporalLine, x, y)
            test = np.vectorize(testWrapper, signature='()->(n)')
            testFrame  = test(np.arange(FrameWidth))
            # testFrame = np.fromiter(map(list, testFrame), dtype=np.dtype('uint8'))
            # testFrame = list(map(list, testFrame))
            # testFrame = np.array(testFrame)
            # print(testFrame)
            OutFrame[y] = testFrame

            # print(f'{FramePos}({x},{y}) = {FResult}')
            # print(f'{TemporalLine[int((FramePos+FResult)%(FrameCount))][x]}')
            cv2.imshow('In progress video frame', OutFrame)
            cv2.waitKey(1)
    # Video.release()
    np.save(TempOut, OutFrame)

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
    parser.add_argument('-o', '--output', type=str, default=None, help='The output file location')
    parser.add_argument('-p', '--process', type=int, default=1, help='The amount of processes to use when processing each frame')
    return parser.parse_args()


def main(inputfile, function, output, threads):
    # Parse the function: str into a mathematical statement
    # function = parser.expr(function).compile()
    for frame in process_video_name(inputfile, function, threads):
        cv2.namedWindow('Finished Video frames')
        cv2.imshow('Finished Video Frames', frame)
        cv2.waitKey(1)
        #TODO: have the frames be used
        np.save('test', frame)

if __name__ == '__main__':
    args = parseArgs()
    main(args.inputfile, args.function, args.output, args.process)
