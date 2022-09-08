import multiprocessing
import os, sys
import shutil
import time
import numpy as np
import pandas as pd
import PIL.Image as Img
# from IPython.display import Image
from sklearn.metrics import confusion_matrix
from collections import deque
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, datasets
import cv2
from IPython import display
import copy
import torch.quantization.quantize_fx as quantize_fx
from multiprocessing import Process
# from multiprocessing import Pool
# from ray.util.multiprocessing import Pool
from torch.multiprocessing import Pool
import concurrent.futures as futures
import cupy as cp


# sys.path.append('./src')
###########################################################
#IMPORT ICMS FEATURES
# from pose_estimation import *
from src.driver_distraction.driver_distraction import Driver_distraction
from src.face_detector.face_detector import FaceDetector
from src.driver_distraction.models.HOD_model import Model
from src.seat_occupancy.seat_allocation import Seat_Allocation
from src.seatbelt_detection.seatbelt_detection import BeltDetector
from src.height_estimation.main import PassengerHeight
from src.window_detection.window_detection import Window_detector,Net
###########################################################
from src.utils.utils import print_text, draw_result, gamma_correction
from src.utils.quantizzer_and_loader import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using GPU:', torch.cuda.is_available())

# LOAD DRIVER DISTRACTION MODEL
# model = models.resnet50()
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 7)
# model_path = 'models/checkpoints/Distraction/ckpt_150'

model_path = 'models/checkpoints/Distraction/ckpt_4'



print('File is there') if os.path.isfile(model_path) else print('Model not found')
try:
    quant_name = model_path.split('/')[-1]
    output_path = f"models/checkpoints/Distraction/{quant_name}_quantized"
    if os.path.isfile(output_path):
        print(f'model {model_path} is already quantized')
    else:
        resnet_quantization(output_path)
except Exception as e:
    print(f'Could not load driver distraction model for the following reason:\n ',e)
# model_static_quantized = load_model(output_path)
model_static_quantized = load_quantized_model(output_path, device)
# model_static_quantized.eval()
# model_static_quantized.cuda()
# model_static_quantized.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# ----------------------------------------------------------------------------------------
# LOAD HAND ON MODEL1
hod = Model()
# hod.to(device)

window_detector = Window_detector()
model2 = Net()
state_dict = torch.load(window_detector.model_path, map_location=device)
model2.load_state_dict(state_dict)
model2.to(window_detector.device)
model2 = model2.eval()

driver_distraction = Driver_distraction(frame = None, fps=None, device=device)

# Load weights from the specified path
hod.load_state_dict(torch.load('./models/checkpoints/HOD/ckpt25', map_location=device))
# hod.load_state_dict(torch.jit.load("./models/checkpoints/HOD/quantized.pt", map_location=device))

def distraction(model_static_quantized, frame):
    # output = output
    first_pred, second_pred, result1, result2, diff = driver_distraction.detect_driver_distraction(
        model_static_quantized, frame)
    return first_pred, second_pred, result1, result2, diff
    # output = [first_pred, second_pred, result1, result2, diff]


# print('Distraction model takes :', time.time() - t2)

# HAND ON DETECTION
# t2 = time.time()
def hand_on(hod, frame):
    output2 = driver_distraction.detect_hand_on(hod, frame)
    return output2


# print('HOD inference :', time.time()-t2)
# print('Output2 is ', output2)

# WINDOW OPEN/CLOSE
# t3 =time.time()
def window_detection(unresized_frame):
    window_detector.frame = unresized_frame
    window_detector.HEIGHT, window_detector.WIDTH = unresized_frame.shape[:2]
    result_right, result_left, fps = window_detector.predict_results(unresized_frame, model2,
                                                                     window_detector.get_transform())
    return result_right, result_left


def height_estimation_and_seat_occupancy(frame, occupancy, current_state):
    t_3 = time.time()
    passengerHeight = PassengerHeight()
    # Resize and restore the fisheye effect
    frame = passengerHeight.restore_distorted_frame(frame)

    # Get pose landmarks
    frame, bbox = passengerHeight.detect_pose_landmarks(frame)

    # print(len(bbox))
    # =======================================================================================================
    upper_body_flag = True
    for box in bbox:
        # Skip unstable passenger pose
        if passengerHeight.skip_unstable_pose(frame, box):
            continue
        # Main stage to calculate height
        passengerHeight.measure_height_passengers(frame, box)

        # Main stage to calculate upper body
        passengerHeight.measure_upper_body_height_passengers(frame, box)

    driver_height, front_passenger_height, back_passenger_height = passengerHeight.calculate_average_result()
    upper_body = passengerHeight.calculate_upper_body_average_result()
    driver_upper_body_height, front_passenger_upper_body_height, back_passenger_upper_body_height = upper_body
    # =======================================================================================================
    print('Height and YOLO POSE take ', time.time() - t_3)
    # POSE AND PASSENGER DETECTION

    # total_fps += FPS
    # Increment frame count.
    # frame_count += 1

    t_4 = time.time()
    # Face detection
    try:
        # FD = FaceDetector(frame=frame)
        # image, faces = FD.face_detector()
        faces = None
    except Exception as e:
        print(f'Face detector is generating an error:{e}')
        faces = None
    print('Face detector takes ', time.time() - t_4)
    # End Face detection

    # SEAT OCCUPANCY
    t_5 = time.time()
    # To adapt automatically later
    extension = 0 if VIDEOS.index(VIDEO) > 4 else 40
    seat_alloc = Seat_Allocation(frame, faces, bbox, occupancy, current_state, last_state, extension)
    current_state, seat_occupancy, occupancy = seat_alloc.find_seat_occupancy_probability()
    # UPDATE OCCUPANCY
    seat_alloc.update_occupancy(current_state)
    print('Seat Occupancy takes ', time.time() - t_5)
    # END SEAT OCCUPANCY
    return current_state, seat_occupancy, occupancy, driver_height, front_passenger_height, back_passenger_height, driver_upper_body_height, front_passenger_upper_body_height, back_passenger_upper_body_height


if __name__ =='__main__':
    # multiprocessing.set_start_method('spawn')
    multiprocessing.set_start_method('forkserver', force=True) # For GPU SUPPORT
    pool = Pool(processes=1)  # processes = number of workers
    #Frame size
    FHD = (1920,1080)
    # hd_size = (1280,720)
    HD = (1280, 768)
    SD = (640, 480)

    VIDEOS = [2,
              'data/test_videos/2.mp4',
              'data/test_videos/7.mp4',
              'data/test_videos/15.mp4',
              'data/test_videos/17.mp4',
              'data/test_videos/01.mp4',
              'data/test_videos/065.mp4',
              'data/test_videos/036.mp4',
              'data/test_videos/054.mp4',
              'data/test_videos/057.mp4',
              'data/test_videos/058.mp4',
              'data/test_videos/059.mp4',
              'data/test_videos/067.mp4',]

    VIDEO = VIDEOS[1]
    cap = cv2.VideoCapture(VIDEO)
    FPS = 20.0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # for avi format
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    OUTPUT_NAME  = VIDEO.split("/")[-1].split(".")[0] if VIDEOS.index(VIDEO) != 0 else 'webcam'
    OUT = cv2.VideoWriter(f'data/output_videos/{OUTPUT_NAME}_output.avi', fourcc, 20.0, HD)

    #Seat Allocation
    # frame_count = 0  # To count total frames.
    # total_fps = 0  # To get the final frames per second.
    occupancy = [['empty', 'empty', 'empty', 'empty', 'empty']]
    last_state = occupancy[0]
    current_state = [0, 0, 0, 0, 0]

    #Seat Belt
    YOLOFI_WEIGHTS= "src/seatbelt_detection/YOLOFI2.weights"
    YOLOFI_CONF = "src/seatbelt_detection/YOLOFI.cfg"
    belt_detected = BeltDetector()
    # cap = cv2.VideoCapture(VIDEO[0])
    predictions = deque([])
    predictions1 = deque([])
    net = cv2.dnn.readNet(YOLOFI_WEIGHTS, YOLOFI_CONF)
    frame_id = -1
    counter = 0

    ###WINDOW OPEN CLOSE MODEL AND OBJECT INITIALIZATION

    all_processes =[]
    while True:
        t_1 = time.time()
        flag, frame = cap.read()

        if flag:
            fps = cap.get(cv2.CAP_PROP_FPS)
            h, w = frame.shape[:2]
            t0 = time.time()
            gamma = 0.5 if VIDEOS.index(VIDEO)>4 else 1
            frame = gamma_correction(frame, gamma)

            # cropped = Img.fromarray(frame)
            # cropped = cropped.crop((350, 0, w - 200, h))
            # frame = np.asarray(cropped)
            unresized_frame = np.copy(frame)

            frame = cv2.resize(frame, HD, interpolation=cv2.INTER_AREA)
            driver_distraction.frame = frame


            # print(frame.shape)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img2 = Img.fromarray(img)
            print('Frame preprocessing takes : ', time.time() - t0)
            # DRIVER'S DISTRACTION
            # first_pred, second_pred, result1, result2, diff = driver_distprint('Preprocessing takes :',time.time()-t0)raction.detect_driver_distraction(model)
            # t1 = time.time()

            # print('Window open/close takes: ',time.time()-t3)

            #
            ##SEAT BELT DETECTION
            # t_2 =time.time()
            # frame_id += 1
            # belt_detected.img = frame
            # _, result_passenger, cnt_on, cnt_off, FPS = belt_detected.batch_result(predictions, net, belt_detected,
            #                                                                        frame_id,
            #                                                                        flip=True)
            # _, result_driver, cnt_on1, cnt_off1, FPS = belt_detected.batch_result(predictions1, net, belt_detected,
            #                                                                       frame_id,
            #                                                                       flip=False)
            # print('Seatbelt detection takes ',time.time()-t_2)

            #HEIGHT ESTIMATION

            #MULTIPROCESSING###############################################################################################

            # P1 = Process(target=distraction, args=(model_static_quantized,frame))
            #
            # all_processes.append(P1)
            #
            # P2 = Process(target=hand_on, args=(hod,frame))
            # # P2.start()
            # all_processes.append(P2)
            #
            # P3 = Process(target=window_detection, args=(unresized_frame,))
            # # P3.start()
            # all_processes.append(P3)
            #
            # P4 = Process(target=height_estimation_and_seat_occupancy, args=(frame, occupancy, current_state))
            # # P4.start()
            # all_processes.append(P4)
            # print('SHSHS')
            # for p in all_processes:
            #     p.start()
            # for i,p in enumerate(all_processes):
            #     # if i>0:
            #     p.join()
            ##POOL#########
            try:
                print(frame.shape)
                # args1 = [(model_static_quantized,frame),]
                # distraction_result = pool.starmap_async(distraction, args1)
                # args2 = [(hod,frame),]
                # hod_result = pool.starmap_async(hand_on, args2)
                # args3 =[(unresized_frame,),]
                # window_result = pool.starmap_async(window_detection, args3)


                args4 = [(frame, occupancy, current_state)]
                height_result = pool.starmap_async(height_estimation_and_seat_occupancy, args4)
            except Exception as e:
                print('Error: ', e)
            # if distraction_result.successful():
            #     print('One is done')
            try:
                # for res in distraction_result.get():
                #     print(res, flush=True)
                # for res1 in hod_result.get():
                #     print(res1, flush=True)
                # for res2 in window_result.get():
                #     print(res2, flush=True)
                # for res3 in height_result.get(0.1):
                #     print(res3)
                # if height_result.successful():
                print(type(height_result.AsyncResult.get(0.2)))
            except Exception as e:
                print('No process complete: ',e)
            # pool.close()
            # pool.join()

            #########CONCURRENT################################################
            # with futures.ProcessPoolExecutor(max_workers=4) as executor:
            #     distraction_result = executor.submit(distraction, (model_static_quantized, )).result()
            #     hod_result = executor.submit(hand_on, (hod,)).result()
            #     window_result = executor.submit(window_detection, (unresized_frame,)).result()
            #     height_result = executor.submit(height_estimation_and_seat_occupancy, (frame, occupancy, current_state)).result()

            ###############################################################################################################
            ##WITHOUT MULTIPROCESSING######################################################################################

            # first_pred, second_pred, result1, result2, diff = distraction(model_static_quantized,frame)
            # t01 = time.time()
            # output2 = hand_on(hod,frame)
            # print('Hand on detection takes: ', time.time()-t01)
            # result_right, result_left = window_detection(unresized_frame)
            # current_state, seat_occupancy, occupancy, driver_height, front_passenger_height, back_passenger_height, driver_upper_body_height, front_passenger_upper_body_height, back_passenger_upper_body_height = height_estimation_and_seat_occupancy(frame, occupancy, current_state)
            ###############################################################################################################

            #PRINTING OUTPUT
            # print(first_pred, result1, result2, diff, output2, faces, sep='--')
            # draw_result(first_pred, result1, result2, diff, output2, boxes=faces, frame=frame)
            # belt_detected.img = frame

            # print_text(frame,f'Driver belt: {result_driver}, passenger belt: {result_passenger}', org=(200, 720))
            # print_text(frame,f'FPS: {int(FPS)}', org=(50, 50))

            #Window
            # print_text(frame,str(result_right), org=(1050, 300), fontScale=.7, thickness=1)
            # print_text(frame,str(result_left), org=(180, 300), fontScale=.7, thickness=1)
            # print_text(f"FPS = {fps:.2f}", org=(150, 1050), color=(0, 0, 255))
            # print('I am here')

            OUT.write(frame)    #save output
            cv2.imshow('video', frame)
            # cv2.imshow('pose', nimg)
            total_time = time.time() - t_1
            print('Total time =', total_time)
            print('FPS', 1/total_time)
            print('=' * 50)
            torch.cuda.empty_cache()
            counter += 1
            print({counter})
        if cv2.waitKey(10) == 27:
            break