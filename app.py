import imutils

import os 
import time
import argparse
import warnings
import operator

from PIL import Image

import cv2
import numpy as np
from numpy.core.numeric import Inf
import tensorflow

import torch
from torchreid.utils import FeatureExtractor
from torchreid.metrics import compute_distance_matrix

from yolo_v4 import YOLO4

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

from tools import generate_detections as gdet

tensorflow.keras.backend.clear_session()
warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help="path to input video", default = "./videos/input/4P-C2.mp4")
ap.add_argument("-o", "--output", help="path to output folder", default = "./videos/output/")
args = vars(ap.parse_args())



def main(yolo):

    start = time.time()

    # Define metric distance  
    threshold = 0.25
    distance_metric = 'cosine'
        
    # Define feature extractor. It can be used different models. See documentation from torchreid.utils.
    extractor = FeatureExtractor(
                                model_name='resnet50', 
                                # model_name='osnet_x0_25', 
                                # model_name='osnet_x1_0', 
                                model_path='/model_data/models/model.pth',
                                device='cuda')


    # DeepSORT
    max_cosine_distance = 0.2
    nn_budget = 30
    model_filename = 'model_data/models/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1) # use to get feature

    # tracking's metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_age=30)


    # Define output folder (if it wasn't)
    out_dir = args['output'] + os.path.basename(args['input']).split('.')[0]
    print('The output folder is: ', out_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Initialize files
    filename = out_dir + '/tracking.txt'
    filename_reid = out_dir + '/reid_tracking.txt'
    out_features = out_dir + './features.npy'

    # If files exist, delete them
    try:
        os.remove(filename)
    except:
        FileNotFoundError
    try:
        os.remove(filename_reid)
    except:
        FileNotFoundError
    try:
        os.remove(out_features)
    except:
        FileNotFoundError


    # Read video
    video_capture = cv2.VideoCapture(args["input"])  
    w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(round(video_capture.get(cv2.CAP_PROP_FPS)))
    
    # Initialize the video writer
    write_video = True
    write_video_path = out_dir + '/video_tracking' + '.avi'
    if write_video == True:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        output_video = cv2.VideoWriter(write_video_path, fourcc, frame_rate, (w, h), True)


    # Defining some needed variables
    fps = 0
    frame_cnt = 0
    track_cnt = dict()
    images_by_id = dict()
    ids_per_frame = []
    feats = dict()    
    final_feats = dict()
    exist_ids = set()
    final_fuse_id = dict()
    detection_time = []
    tracking_time = []
    BATCH_SIZE = 32

    # Start video
    while True:

        t1 = time.time()
        
        # Read frame (frame shape 640*480*3)
        ret, frame = video_capture.read()

        # End of video
        if ret != True: 
            break
        
        image = Image.fromarray(frame[...,::-1]) 

        # YOLO detection for each 5 frames
        if frame_cnt%5 == 0:
            t = time.time()

            # Use YOLO for object detection
            boxs = yolo.detect_image(image)
            features = encoder(frame, boxs)
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)] # score to 1.0 here
            detection_time.append(time.time()-t)


        t = time.time()
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
       
        tmp_ids = []
        population = 0
        # Tracking loop
        for track in tracker.tracks:

            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            
            # Get detection bounding box 
            bbox = track.to_tlbr()

            # If bbox is inside the image:
            if bbox[0] >= 0 and bbox[1] >= 0 and bbox[3] < h and bbox[2] < w:               

                tmp_ids.append(track.track_id)

                # Save the object's position and its image
                if track.track_id not in track_cnt:

                    track_cnt[track.track_id] = [[frame_cnt, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]]
                    images_by_id[track.track_id] = [frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]]

                else:

                    track_cnt[track.track_id].append([frame_cnt, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
                    images_by_id[track.track_id].append(frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])

                cv2_addBox(track.track_id, frame, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), 3, 2, 2)
                write_results(filename, 'mot', frame_cnt+1, str(track.track_id), int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), w, h)
                population += 1
        
        ids_per_frame.append(set(tmp_ids))
        tracking_time.append(time.time()-t)

        # Show tracking result
        cv2.putText(frame, "FPS: %.3f"%(fps),(int(20), int(40)),0, 5e-3 * 300, (0,255,0),3)
        # cv2.putText(frame, "Frame: {}".format(frame_cnt),(int(20), int(80)),0, 5e-3 * 600, (0,255,0), 6)
        # cv2.putText(frame, "Population: {}".format(population), (int(20), int(80)),0, 5e-3 * 600, (0,255,255),6)
        cv2.namedWindow("YOLOv4_with_DeepSORT", 0)
        cv2.resizeWindow('YOLOv4_with_DeepSORT', 1024, 768)
        cv2.imshow('YOLOv4_with_DeepSORT', frame)

        # Check fps
        try:
            fps  = ( fps + (1./(time.time()-t1)) ) * 0.5
        except:
            ZeroDivisionError
        
        # Check to see if we should write the frame to disk
        if write_video == True:
            output_video.write(frame)

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_cnt += 1
        
    #######################################
    #### End of video
    print('\n\nDetection average time consumption: ', round(sum(detection_time)/len(detection_time), 4), ' seconds')
    print('Tracking average time consumption: ', round(sum(tracking_time)/len(tracking_time), 4), ' seconds')


    # A bit of cleaning!
    video_capture.release()
    if write_video == True:
        output_video.release()
    cv2.destroyAllWindows()
    
    
    # Generate features from detections
    # BATCH_SIZE to avoid out of memory error
    t = time.time()
    print('\nTotal IDs = ', len(images_by_id))
    for i in images_by_id:
        aux = torch.Tensor()
        print('ID number {} -> Number of frames {}'.format(i, len(images_by_id[i])))

        if BATCH_SIZE > len(images_by_id[i]):
            feats[i] = extractor(images_by_id[i]).data.cpu()

        else:
            for j in range(0, len(images_by_id[i])//BATCH_SIZE):
                aux2 = extractor(images_by_id[i][j*BATCH_SIZE:(j+1)*BATCH_SIZE]).data.cpu()
                aux = torch.cat(tensors=[aux, aux2])
            try:
                aux2 = extractor(images_by_id[i][(j+1)*BATCH_SIZE:]).data.cpu()
                feats[i] = torch.cat(tensors=[aux, aux2])
            except:
                feats[i] = torch.cat(tensors=[aux, aux2])
                continue
    print('Features generation time consumption: ', round(time.time()-t,3), ' seconds')


    # Rewrite IDs
    # Reidentification on the same video
    t = time.time()
    print('\nReWriting IDs...')
    for f in ids_per_frame:

        if f:   # If there are IDs on this frame:

            if len(exist_ids) == 0:
                for i in f:   # Loop over each ID of a frame
                    final_fuse_id[i] = [i]   # final_fuse_id save on ID position, the ID value for this frame
                exist_ids = exist_ids or f      # Here it will be always the second value because exist_ids is null

            else:
                new_ids = f-exist_ids       # New Ids: known IDs minus IDs from this frame

                for nid in new_ids:     # Loop over new IDs
                    dis = []
                    if len(images_by_id[nid]) < 10:       # If there are not enough images for that ID, discard it.
                        exist_ids.add(nid)
                        continue
                    unpickable = []
                    
                    for i in f:    # Loop over IDs of a frame
                        for key,item in final_fuse_id.items():      # Grouping up IDs that can not be chosen in this frame (people can not be in two places at the same time!!)
                            if i in item:
                                unpickable += final_fuse_id[key]

                    list_oid = []
                    for oid in (exist_ids-set(unpickable))&set(final_fuse_id.keys()):
                        tmp = np.mean(compute_distance_matrix(feats[nid],feats[oid], metric=distance_metric).numpy())
                        dis.append([oid, tmp])
                        list_oid.append(oid)
                    exist_ids.add(nid)

                    if not dis:
                        final_fuse_id[nid] = [nid]
                        continue
            
                    dis.sort(key=operator.itemgetter(1))        # Sort the list
                    if dis[0][1] < threshold:
                        combined_id = dis[0][0]
                        images_by_id[combined_id] += images_by_id[nid]      # Mix images from IDX with IDY (X and Y are the same people)
                        final_fuse_id[combined_id].append(nid)

                    else:
                        final_fuse_id[nid] = [nid]
    print(final_fuse_id)
    print('[ReID done]\n')
    print('ReID time consumption: ', round(time.time()-t, 4), ' seconds')

    
    # Combine features after ReID
    # This is one of our corrections
    t = time.time()
    print('\nTotal diferent IDs = ', len(final_fuse_id))
    print('Combining IDs...')
    for key, subkey in final_fuse_id.items():
        final_feats[key] = feats[key]
        for k in subkey:    
            if k != key:
                final_feats[key] = torch.cat((final_feats[key], feats[k]), 0)
    print('Combining features time consumption: ', round(time.time()-t, 4), ' seconds')


    # Saving feature vectors on a NumPy file
    t = time.time()   
    print('\nWriting features...')
    np.save(file=out_features, arr=final_feats)
    print('[Features saved]')
    print('Saving features time consumption: ', round(time.time()-t, 3))


    # Generate videos             
    write_video_path_final = out_dir + '/video_final_results' + '.avi'
    write_video_path_reid = out_dir + '/video_reid_results' + '.avi'
    video_capture = cv2.VideoCapture(args["input"])
    out_reid = cv2.VideoWriter(write_video_path_reid, fourcc, frame_rate, (w, h), True)
    cond = False
    
    try:
        file_positions = open(filename, 'r')
        positions = file_positions.readline()
    except:
        FileExistsError

    # Reproduce video again
    for j in range(0, frame_cnt):

        t1 = time.time()
        ret, reid_frame = video_capture.read()
        frame = reid_frame.copy()

        for idx in final_fuse_id:
            for i in final_fuse_id[idx]:
                for f in track_cnt[i]:                    
                    if j == f[0]:                
                        cv2_addBox(idx, reid_frame, f[1], f[2], f[3], f[4], 3, 2, 2)
                        write_results(filename_reid, 'mot', j+1, str(idx), f[1], f[2], f[3], f[4], w, h)

        
        while True:

            if (positions.strip().split(',')[0] == '') or (int(positions.strip().split(',')[0]) > j):
                break
            position = positions.strip().split(',')
            index = position[1]
            bbox = position[2:6]
            cv2_addBox(int(index), frame, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), 3, 2, 2)
            positions = file_positions.readline()
                
        # Check fps
        try:
            fps  = ( fps + (1./(time.time()-t1)) ) * 0.5
        except:
            ZeroDivisionError

        # Show final results together
        final_frame = np.hstack((frame, reid_frame))
        final_frame = imutils.resize(image=final_frame, width=1920)

        cv2.putText(reid_frame, 'ReID', (int(20), int(40)), 0, 5e-3 * 200, (0,255,0), 3)
        # cv2.putText(reid_frame, "FPS: %.3f"%(fps),(int(60), int(40)), 0, 5e-3 * 200, (0,255,0), 3)

        cv2.putText(final_frame, 'DeepSORT', (int(20), int(40)), 0, 5e-3 * 200, (0,255,0), 3)
        cv2.putText(final_frame, 'ReID', (int(final_frame.shape[1]/2 + 20), int(40)), 0, 5e-3 * 200, (0,255,0), 3)
        cv2.line(final_frame, pt1=(int(final_frame.shape[1]/2), 0), pt2=(int(final_frame.shape[1]/2), h), color=(0,0,0), thickness=2)
        cv2.namedWindow("Final Result", 1)
        cv2.imshow('Final Result', final_frame)

        if cond == False:
            cond = True
            w = final_frame.shape[1]
            h = final_frame.shape[0]
            out_final = cv2.VideoWriter(write_video_path_final, fourcc, frame_rate, (w, h), True)

        # Write final video
        out_reid.write(reid_frame)
        out_final.write(final_frame)

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



    # Cleaning!
    out_reid.release()
    out_final.release()
    video_capture.release()
    cv2.destroyAllWindows()
    end = time.time()    
    print("\n[Finish]")
    print('\nTotal execution time: ',round(end - start, 2), ' seconds')
    



############################################################################################
############################################################################################
def cv2_addBox(track_id, frame, x1, y1, x2, y2, line_thickness, text_thickness,text_scale):

    color = get_color(abs(track_id))

    cv2.rectangle(img =frame,
                  pt1 = (x1, y1),
                  pt2 = (x2, y2),
                  color = color,
                  thickness = line_thickness)

    cv2.putText(img = frame,
                text = str(track_id),
                org = (x1, y1+30),
                fontFace = cv2.FONT_HERSHEY_PLAIN,
                fontScale = text_scale,
                color = (0,0,255),
                thickness = text_thickness)
    

def write_results(filename, data_type, w_frame_id, w_track_id, w_x1, w_y1, w_x2, w_y2, w_wid, w_hgt):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{x2},{y2},{w},{h}\n'
    else:
        raise ValueError(data_type)
    with open(filename, 'a') as f:
        line = save_format.format(frame=w_frame_id, id=w_track_id, x1=w_x1, y1=w_y1, x2=w_x2, y2=w_y2, w=w_wid, h=w_hgt)
        f.write(line)


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color




if __name__ == '__main__':

    gpu_options = tensorflow.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tensorflow.compat.v1.Session(config=tensorflow.compat.v1.ConfigProto(gpu_options=gpu_options))

    main(YOLO4())