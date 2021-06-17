import  numpy as np
import csv
import cv2 as cv

name = "UBFC2"
signalGT = 'BVP'
numlevel = 2         # depth of the filesystem collecting video and BVP files
numSubjects = 26     #number of subjects
video_EXT = 'avi'    #extension of the video files
frameRate = 30       #video frame rate
VIDEO_SUBSTRING = '' #substring contained in the filename
SIG_EXT = 'txt'      #extension of the ground truth file
SIG_SUBSTRING = ''   #substring contained in the filename
SIG_SampleRate = 30  #sample rate of the BVP files
skinThresh = [40,60]      #thresholds for skin detection

def readSigfile(filename):
    """
    load BVP signal file.
    :param self:
    :param filename:
    :return:
    """
    gtTrace = []
    gtTime = []
    gtHR = []
    with open(filename,'r') as f :
        x = f.readlines()
    s = x[0].split(' ')
    s = list(filter(lambda a:a != '',s))
    gtTrace = np.array(s).astype(np.float64)

    t = x[2].split(' ')
    t = list(filter(lambda  a: a != '' ,t))
    gtTime = np.array(t).astype(np.float64)

    hr = x[1].split(' ')
    hr = list(filter(lambda  a: a != '' ,hr))
    gtHR = np.array(hr).astype(np.float64)

    data = np.array(gtTrace)
    time = np.array(gtTime)
    hr = np.array(gtHR)

    return data,hr
MIN_FRAME = 50
def read_video(filename,data):
    cap = cv.VideoCapture(filename)
    success = 1
    graph_height = 200
    graph_width = 0
    count = 0
    value = []
    fourcc = cv.VideoWriter_fourcc('X','V','I','D')
    out = cv.VideoWriter('BPV_show.avi',fourcc, 30.0, (640,680))
    while success :
        success ,image = cap.read()
        if success is False:
            break
        view = np.array(image)
        print(view.shape)
        value.append(data[count])
        graph_width = int(view.shape[1])
        if len(value) > MAX_VALUES_TO_GRAPH:
            value.pop(0)
        graph = draw_graph(value, graph_width, graph_height)
        new_image  = np.vstack((view, graph))
        cv.imwrite('./frame/UBFC_{}.jpg'.format(count),new_image)
        print(new_image.shape)
        out.write(new_image)
        count = count +1
    cap.release()
    out.release()
MAX_VALUES_TO_GRAPH = 50

def draw_graph(signal_values,graph_width, graph_height):
    graph = np.zeros((graph_height, graph_width, 3), np.uint8)
    scale_factor_x = float(graph_width) / MAX_VALUES_TO_GRAPH
    #Automatically rescale vertically based on the value with largest absolute value
    max_abs = get_max_abs(signal_values)
    scale_factor_y = (float(graph_height) / 2.0) / max_abs
    midpoint_y = graph_height / 2
    for i in range(0,len(signal_values) - 1):
        curr_x = int(i * scale_factor_x)
        curr_y = int(midpoint_y + signal_values[i] * scale_factor_y)
        next_x = int((i + 1) * scale_factor_x)
        next_y = int(midpoint_y + signal_values[i + 1] * scale_factor_y)
        #print('curr_x: {},curr_y: {};next_x: {},next_y: {}'.format(curr_x, curr_y,next_x,next_y))
        cv.line(graph,(curr_x, curr_y), (next_x, next_y), color=(0,255,0),thickness=2)
    return graph

def get_max_abs(lst):
    return (max(max(lst),-min(lst)))
BVP_filename = "/home/zq/video_process/dataset/UBFC/dataset_2/subject1/ground_truth.txt"
data,hr = readSigfile(BVP_filename)
print(data,hr)
Video_filename = '/home/zq/video_process/dataset/UBFC/dataset_2/subject1/vid.avi'



def channel_augment(file):
    cap = cv.VideoCapture(file)
    success = 1
    fourcc = cv.VideoWriter_fourcc('X','V','I','D')
    out = cv.VideoWriter('BPV_show.avi',fourcc, 30.0, (640,480))
    while success:
        success, image = cap.read()
        if success is False:
            break
        image = np.array(image)
        image[:,:,1] = image[:,:,1]*1.4
        image[:, :, 0] = image[:, :, 0] * 1
        image[:, :, 2] = image[:, :, 2] * 1
        out.write(image)
    cap.release()
    out.release()

#channel_augment(Video_filename)

