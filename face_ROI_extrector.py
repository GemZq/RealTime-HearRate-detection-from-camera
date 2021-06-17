import cv2
import dlib
import numpy as np
from scipy import signal
from  PIL import Image
import matplotlib.pyplot as plt
import json
from UBFC2 import *
VIDEO = '/home/zq/video_process/dataset/UBFC/dataset_2/subject1/vid.avi'
GT_file = '/home/zq/video_process/dataset/UBFC/dataset_2/subject1/ground_truth.txt'
Window_Title = 'Pulse Observer'
BUFFER_MAX_SIZE = 500 #Number of recent ROI average values to store
MAX_VALUES_TO_GRAPH = 50 #Number of recent ROI average values to show in the pulse graph
MIN_HZ = 0.83 #50 BPM -minimum allowed heart rate
MAX_HZ = 3.33 #200 BPM - maximum allowd heart rate
MIN_FRAMES = 100 #Minnimum number of frames required befor heart rate is computed . Highter values are slower ,but more accurate.
DEBUG_MODE = False
VideoCapture = cv2.VideoCapture(VIDEO) # the video of will process
fps = VideoCapture.get(cv2.CAP_PROP_FPS)  # frames per secound
size = (int(VideoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(VideoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))#获取视频的大小
def frameCapture(path,data,hr):
        cap = cv2.VideoCapture(path)
        count = 0
        success = 1
        roi_avg_values = []
        times = []
        graph_values = []
        graph_height = 100
        graph_width = 0
        last_time = []
        last_bpm = 0
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 30, (640, 680))
        Ground_Truth = []
        while success:
                success , image = cap.read()
                times.append(cap.get(cv2.CAP_PROP_POS_MSEC))
                if success is False:
                    cap.release()
                    out.release()
                    break
                view = np.array(image)
                Ground_Truth.append(data[count])
                graph_width = int(view.shape[1]*0.75)
                bpm_es_graph_width = int(view.shape[1]*0.25)
                avg ,face_retangle = face_landmarks(image)
                roi_avg_values.append(avg)
                if len(times) > BUFFER_MAX_SIZE:
                        roi_avg_values.pop(0)
                        times.pop(0)
                        Ground_Truth.pop(0)
                curr_buffer_size = len(times)
                #Don't try to compute pulse until we have at least the min number of frames
                if curr_buffer_size > MIN_FRAMES:
                        print("processing and saving....")
                        time_elapsed = (times[-1] - times[0])/1000
                        fps = curr_buffer_size / time_elapsed
                        values = np.array(roi_avg_values)
                        np.nan_to_num(values, copy=False)
                        detrended = signal.detrend(values, type='linear')
                        demeaned = sliding_window_demean(detrended, 15)
                        filtered = butterworth_filter(demeaned, MIN_HZ, MAX_HZ, fps,order = 5)
                        bpm = int(compute_bpm(filtered, fps, curr_buffer_size, last_bpm))
                        bpm_es_graph = draw_bpm_graph(str(bpm),bpm_es_graph_width,graph_height)
                        view[face_retangle[1]:face_retangle[3],face_retangle[0]:face_retangle[2],:] = view[face_retangle[1]:face_retangle[3],face_retangle[0]:face_retangle[2],:] + view[face_retangle[1]:face_retangle[3],face_retangle[0]:face_retangle[2],:] *filtered[-1]
                        graph_values.append(filtered[-1])
                        last_time.append(times[-1])
                        if len(graph_values) > MAX_VALUES_TO_GRAPH:
                            graph_values.pop(0)
                        graph = draw_graph(graph_values, graph_width, graph_height)
                        GT_graph = draw_graph(Ground_Truth, graph_width, graph_height, color=0)
                        bpm_gt_graph = draw_bpm_graph(str(int(hr[count])), bpm_es_graph_width, graph_height)
                        bpmgraph = np.hstack((graph,bpm_es_graph))
                        gtbpmgraph = np.hstack((GT_graph,bpm_gt_graph))
                        view = np.vstack((view, bpmgraph, gtbpmgraph))
                        out.write(view)
                        cv2.imwrite('colorchange.jpg',view)
                count += 1
        return graph_values,last_time

def face_landmarks(img):
        detector = dlib.get_frontal_face_detector()
        predictor_path = './shape_predictor_68_face_landmarks.dat'
        predictor = dlib.shape_predictor(predictor_path)
        view = np.array(img,np.uint8)
#        print('the image shape is {}'.format(img.shape))
        dets = detector(img,1)
#        print("检测到了%d个人脸"%len(dets))
        for inx, face  in enumerate(dets):
#                print('- %d: Left %d Top %d Right %d Bottom %d' % (inx, face.left(), face.top(), face.right(), face.bottom()))
                face_retangle = [face.left(),face.top(),face.right(),face.bottom()]
                face_points = predictor(img,face)
                points = np.zeros((len(face_points.parts()),2)) # 68*2
#                print(face_points.parts())
                for i,part in enumerate(face_points.parts()):
                        points[i] = (part.x, part.y)
                forehead_left = int(points[21,0])
                forehead_right = int(points[22, 0])
                forehead_top = int(min(points[21, 1], points[22, 1])) - (int(points[22, 0]) - int(points[21,0]))
                forehead_bottom = int(int(max(points[21, 1], points[22, 1])) * 0.98)
                cv2.rectangle(view,  (forehead_left, forehead_top), (forehead_right, forehead_bottom), color=(0, 255, 0), thickness=2)
                #leftface_ROI
                leftface_left = int(points[17, 0])
                leftface_right = int(points[38, 0])
                leftface_top = int(points[38, 1] * 1.05)
                leftface_bottom = int(points[50, 1] * 0.95 )
                cv2.rectangle(view, (leftface_left, leftface_top), (leftface_right,leftface_bottom), color=(0, 255, 0), thickness=2)
                #rightface_ROI
                rightface_left = int(points[47, 0])
                rightface_right = int(points[26, 0])
                rightface_top = int(points[38, 1] * 1.05)
                rightface_bottom = int(points[50, 1] * 0.95)
                cv2.circle(view, (rightface_left, rightface_top), 1, (0, 0, 255), -1)
                cv2.circle(view, (rightface_right, rightface_bottom), 1, (0, 0, 255), -1)
                cv2.rectangle(view, (rightface_left, rightface_top), (rightface_right, rightface_bottom), color=(0, 255, 0),thickness=2)
                forehead_ROI = img[forehead_top:leftface_bottom,forehead_left:forehead_right]
                leftface_ROI = img[leftface_top:leftface_bottom,leftface_left:leftface_right]
                rightface_ROI = img[rightface_top:rightface_bottom,rightface_left:rightface_right]
                forehead_green = forehead_ROI[:, :, 1]
                leftface_green = leftface_ROI[:, :, 1]
                rightface_green = forehead_ROI[:, :, 1]
                avg = (np.mean(forehead_green) + np.mean(leftface_green) + np.mean(rightface_green)) / 3.0
                return avg,face_retangle
                # (x, y) = points[50]
                # cv2.circle(view,(int(x), int(y)), 1, (0, 0, 255), -1)
                #cv2.imwrite('facelandmark.jpg', np.uint8(view))

def sliding_window_demean(signal_values, num_windows):
        windows_size = int(round(len(signal_values) / num_windows))
        demeaned = np.zeros(signal_values.shape)
        for i in range(0,len(signal_values),windows_size):
                if i + windows_size > len(signal_values):
                        windows_size = len(signal_values) - 1
                curr_slice = signal_values[i: i + windows_size]
                #print("Inx is {},len of signal_val is {},len of curr_slice is {}".format(i,len(signal_values),len(curr_slice)))
                if DEBUG_MODE and curr_slice.size == 0 :
                        print('Empty Slice: size={0}, i={1}, window_size={2}'.format(signal_values.size, i,windows_size))
                        print(curr_slice)
                demeaned[i:i + windows_size] = curr_slice - np.mean(curr_slice)
        return demeaned

def butterworth_filter(data, low, high, sample_rate , order = 5):
        nyquist_rate = sample_rate * 0.5
        low /= nyquist_rate
        high /= nyquist_rate
        b, a = signal.butter(order, [low,high], btype='bandpass')
        return signal.lfilter(b, a, data)

#Draw the HR graph in video
def draw_graph(signal_values,graph_width, graph_height, color=1):
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
        if color == 1:
            cv2.line(graph,(curr_x, curr_y), (next_x, next_y), color=(0,255,0),thickness=2)
        elif color == 0 :
            cv2.line(graph, (curr_x, curr_y), (next_x, next_y), color=(0, 0, 255), thickness=2)
    return graph
#Draw the heart rate number in video
def draw_bpm_graph(bpm, bpm_display_width, graph_height):
    bpm_display = np.zeros((graph_height,bpm_display_width,3),np.uint8)
    bpm_text_size , BaseLine = cv2.getTextSize(str(bpm), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale = 2.7,thickness = 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bpm_display,bpm,(int(bpm_display_width - bpm_text_size[0]),int(graph_height/2 + BaseLine)),fontFace=font, fontScale=2, color=(0,255,0), thickness=2)
    return bpm_display

#Return maximum absolute value from a list
def get_max_abs(lst):
    return (max(max(lst),-min(lst)))

#Calculate the pulse in beats per minute (BPM)
def compute_bpm(filtered_values, fps, buffer_size, last_bpm):
    # Compute FFt
    fft = np.abs(np.fft.rfft(filtered_values))
    # Generate list of frequncies that correspond to the FFT values
    freqs = fps / buffer_size * np.arange(buffer_size / 2 + 1)
    # Filter out any peaks in the FFT that are not within our range of [MIN_HZ, MAX_HZ]
    while True :
        max_idx = fft.argmax()
        bps = freqs[max_idx]
        if bps < MIN_HZ or bps  > MAX_HZ:
            fft[max_idx] = 0
        else :
            bpm = bps * 60
        break
    #It's impossible for the heart rate to change more than 10% between samples，
    if last_bpm > 0 :
        bpm = (last_bpm * 0.9) + (bpm * 0.1)
    return bpm


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

data,hr = readSigfile(GT_file)
graph_values,times=frameCapture(VIDEO,data,hr)
# image = './test.jpg'
# image = cv2.imread(image)
# face_landmarks(image)


