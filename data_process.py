#code: utf-8
import cv2
import os
import datetime
import time
import numpy
source_ROOT = '/mnt/原始视频/'
path_list=['downward_view','side_view','upward_view']
goal_ROOT = '/home/zq/video_process/dataset/mouse/'
dir_list =['female1','female2','female3','female4','male1','male2','male3','male4']
video_dict = {}
view_dict = {}
time_dict = {}


def calculate_timestamp():
    for idx, name in enumerate(dir_list):
        view_dict = {}
        for i, dir in enumerate(path_list):
            file = os.path.join(source_ROOT,dir,name,'a1.MP4')
            time1 = str(os.popen('ffmpeg -i {}   2>&1  | grep -I "creation_time" | head -n 1 |  cut -d " " -f 9| cut -d "T" -f 2 | cut -d "." -f 1'.format(file)).read()).replace('\n','')
            time1 = '2021/01/30 ' + time1
            d = time.strptime(time1,'%Y/%m/%d %H:%M:%S')
            timestamp = time.mktime(d)
            view_dict[dir] = timestamp
            time_local = time.localtime(timestamp)
            dt_new = time.strftime("%Y%m%d-%H:%M:%S",time_local)
            time_dict[dir] = dt_new
        video_dict[name] = view_dict
        max_value = max(video_dict[name].values())
        min_value = min(video_dict[name].values())
    return video_dict

create_time_dict = calculate_timestamp()
print(create_time_dict)

def calculate_duration():
    video_duration_dict = {}
    for idx, name in enumerate(dir_list):
        view_duration_dict = {}
        for i, dir in enumerate(path_list):
            file = os.path.join(goal_ROOT, dir, '{}.MP4'.format(name))
            duration = str(os.popen('ffmpeg -i {}   2>&1  | grep Duration  |  cut -d " " -f 4 | cut -d "." -f 1'.format(file)).read()).replace('\n','')
            minute = duration.split(':',2)
            duration = int(minute[1]) * 60 + int(minute[2])
            view_duration_dict[dir] = duration
        video_duration_dict[name] = view_duration_dict
    return video_duration_dict

duration_dict = calculate_duration()
print(duration_dict)

def calculate_starttime_diff(start_dict):
    time_diff = {}
    for sex,views in start_dict.items():
        max_value = max(views.values())
        max_value = max(views.items(), key=lambda x: x[1])
        min_value = min(views.items(), key=lambda x: x[1])
        min_name = min_value[0]
        max_name = max_value[0]
        time_diff[min_name + '/' + sex] = int(max_value[1]) - int(min_value[1])
        del views[min_name]
        min_value = min(views.items(), key=lambda x: x[1])
        min_name = min_value[0]
        time_diff[min_name + '/' + sex] = int(max_value[1]) - int(min_value[1])
        if time_diff[min_name + '/' + sex] < 10 :
            time_diff[min_name + '/' + sex] = '0' +  str(time_diff[min_name + '/' + sex])
        time_diff[max_name + '/' + sex] = '00'
    return time_diff
time_diff = calculate_starttime_diff(create_time_dict)
print(time_diff)

def calculate_duration_diff(time_diff,duration_dict,dir_list):
     duration_diff = {}
     view = []
     duration_diff_dict = {}
     for idx, name in enumerate(dir_list):
         diff_ = []
         for i, dir in enumerate(path_list):
             try:
                duration_diff_seound = int(duration_dict[name][dir]) - int(time_diff[dir + '/' + name])
             except :
                duration_diff_seound = 9999
                pass
             diff_.append(duration_diff_seound)
         duration_diff_dict[name] = diff_

     duration = {}
     for i in dir_list:
         duration_secound = min(duration_diff_dict[i])
         secound = str(duration_secound % 60)
         if duration_secound % 60 < 10:
             secound = '0' + str(duration_secound%60)
         duration_secound = '00:' + str(int(duration_secound/60)) + ':' +  secound
         duration_dict[i] = duration_secound
     return duration_dict



duration_dict = calculate_duration_diff(time_diff, duration_dict,dir_list)
print(duration_dict)
for idx, sex in enumerate(dir_list):
    crop_duration = duration_dict[sex]
    for i, view in enumerate(path_list):
        start_time = '00:00:{}'.format(time_diff[view+'/'+sex])
        cmd = "ffmpeg -ss {} -t {} -i /home/zq/video_process/dataset/mouse/{}/{}.MP4 -vcodec copy -acodec copy /home/zq/video_process/dataset/crop_mouse/{}/{}_crop.MP4".format(start_time,crop_duration,view,sex,view,sex)
        os.system(cmd)


# 最先开始的减-最后开始的female1
# ffmpeg -ss 00:00:10 -t 00:15:00 -i side_view/female1/output.MP4 -vcodec copy -acodec copy side_view/female1/crop.MP4
# ffmpeg -ss 00:00:14 -t 00:15:00 -i upward_view/female1/output.MP4 -vcodec copy -acodec copy upward_view/female1/crop.MP4
# ffmpeg -ss 00:00:00 -t 00:15:00 -i downward_view/female1/output.MP4 -vcodec copy -acodec copy downward_view/female1/crop.MP4
# rm -f downward_view/female1/output.MP4
# rm -f side_view/female1/output.MP4
# rm -f upward_view/female1/output.MP4

#female2
# ffmpeg -ss 00:00:09 -t 00:15:00 -i side_view/female2/output.MP4 -vcodec copy -acodec copy side_view/female2/crop.MP4
# ffmpeg -ss 00:00:11 -t 00:15:00 -i upward_view/female2/output.MP4 -vcodec copy -acodec copy upward_view/female2/crop.MP4
# ffmpeg -ss 00:00:00 -t 00:15:00 -i downward_view/female2/output.MP4 -vcodec copy -acodec copy downward_view/female2/crop.MP4
# rm -f downward_view/female2/output.MP4
# rm -f side_view/female2/output.MP4
# rm -f upward_view/female2/output.MP4
#female3
# ffmpeg -ss 00:00:10 -t 00:15:00 -i side_view/female3/output.MP4 -vcodec copy -acodec copy side_view/female3/crop.MP4
# ffmpeg -ss 00:00:13 -t 00:15:00 -i upward_view/female3/output.MP4 -vcodec copy -acodec copy upward_view/female3/crop.MP4
# ffmpeg -ss 00:00:00 -t 00:15:00 -i downward_view/female3/output.MP4 -vcodec copy -acodec copy downward_view/female3/crop.MP4
# rm -f downward_view/female3/output.MP4
# rm -f side_view/female3/output.MP4
# rm -f upward_view/female3/output.MP4
#female4
# ffmpeg -ss 00:00:09 -t 00:15:00 -i side_view/female4/output.MP4 -vcodec copy -acodec copy side_view/female4/crop.MP4
# ffmpeg -ss 00:00:13 -t 00:15:00 -i upward_view/female4/output.MP4 -vcodec copy -acodec copy upward_view/female4/crop.MP4
# ffmpeg -ss 00:00:00 -t 00:15:00 -i downward_view/female4/output.MP4 -vcodec copy -acodec copy downward_view/female4/crop.MP4
# rm -f downward_view/female4/output.MP4
# rm -f side_view/female4/output.MP4
# rm -f upward_view/female4/output.MP4
#male1
# ffmpeg -ss 00:00:00 -t 00:15:00 -i side_view/male1/output.MP4 -vcodec copy -acodec copy side_view/male1/crop.MP4
# ffmpeg -ss 00:00:14 -t 00:15:00 -i upward_view/male1/output.MP4 -vcodec copy -acodec copy upward_view/male1/crop.MP4
# ffmpeg -ss 00:00:10 -t 00:15:00 -i downward_view/male1/output.MP4 -vcodec copy -acodec copy downward_view/male1/crop.MP4
# rm -f downward_view/male1/output.MP4
# rm -f side_view/male1/output.MP4
# rm -f upward_view/male1/output.MP4
#male2
# ffmpeg -ss 00:00:09 -t 00:15:00 -i side_view/male2/output.MP4 -vcodec copy -acodec copy side_view/male2/crop.MP4
# ffmpeg -ss 00:00:13 -t 00:15:00 -i upward_view/male2/output.MP4 -vcodec copy -acodec copy upward_view/male2/crop.MP4
# ffmpeg -ss 00:00:00 -t 00:15:00 -i downward_view/male2/output.MP4 -vcodec copy -acodec copy downward_view/male2/crop.MP4
# rm -f downward_view/male2/output.MP4
# rm -f side_view/male2/output.MP4
# rm -f upward_view/male2/output.MP4

#male3
# ffmpeg -ss 00:00:09 -t 00:15:00 -i side_view/male3/output.MP4 -vcodec copy -acodec copy side_view/male3/crop.MP4
# ffmpeg -ss 00:00:13 -t 00:15:00 -i upward_view/male3/output.MP4 -vcodec copy -acodec copy upward_view/male3/crop.MP4
# ffmpeg -ss 00:00:00 -t 00:15:00 -i downward_view/male3/output.MP4 -vcodec copy -acodec copy downward_view/male3/crop.MP4
# rm -f downward_view/male3/output.MP4
# rm -f side_view/male3/output.MP4
# rm -f upward_view/male3/output.MP4

#male4
# ffmpeg -ss 00:00:08 -t 00:15:00 -i side_view/male4/output.MP4 -vcodec copy -acodec copy side_view/male4/crop.MP4
# ffmpeg -ss 00:00:12 -t 00:15:00 -i upward_view/male4/output.MP4 -vcodec copy -acodec copy upward_view/male4/crop.MP4
# ffmpeg -ss 00:00:00 -t 00:15:00 -i downward_view/male4/output.MP4 -vcodec copy -acodec copy downward_view/male4/crop.MP4
# rm -f downward_view/male4/output.MP4
# rm -f side_view/male4/output.MP4
# rm -f upward_view/male4/output.MP4
