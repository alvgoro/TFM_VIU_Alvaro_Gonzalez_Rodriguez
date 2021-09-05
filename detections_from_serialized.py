import os
import time
import argparse
import warnings
import operator

from PIL import Image

import cv2
import numpy as np
import tensorflow

import torch
from torchreid.utils import FeatureExtractor
from torchreid.metrics import compute_distance_matrix

import numpy as np

from yolo_v4 import YOLO4

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

from tools import generate_detections as gdet


torch.cuda.empty_cache()
tensorflow.keras.backend.clear_session()

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help="path to input video", default = "./videos/input/Double1.mp4")
ap.add_argument("-f", "--features", help="path to features filename", default = "./videos/output/Single1/features.npy")
ap.add_argument("-o", "--output", help="path to output folder", default = "./videos/output/reidentification_")
args = vars(ap.parse_args())

warnings.filterwarnings('ignore')

def main(yolo):

    # Loading saved features directory
    known_features = np.load(args['features'], allow_pickle=True).item()

    
    extractor = FeatureExtractor(model_name='resnet50', 
                                model_path='/model_data/models/model.pth',
                                device='cuda')

    # Define metric distance  
    threshold = 0.30            # 70% similitud
    distance_metric = 'cosine'

    start = time.time()   

    # deep_sort 
    max_cosine_distance = 0.2
    nn_budget = 30
    nms_max_overlap = 0.3
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

    # Initialize tracking file
    filename = out_dir + '/tracking.txt'
    filename_reid = out_dir + '/reid_tracking.txt'
    # If files exist, delete them
    try:
        os.remove(filename)
    except:
        FileNotFoundError
    try:
        os.remove(filename_reid)
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


    fps = 0
    frame_cnt = 0   
    detection_time = []
    tracking_time = []
    reid_time = []
    final_index = dict()

    while True:

        t1 = time.time()
        reid = False
        # Read frame (frame shape 640*480*3)
        ret, frame = video_capture.read()
       
        if ret != True:
            break

        image = Image.fromarray(frame[...,::-1]) 

        # YOLO detection for each K frames
        if frame_cnt % 5 == 0:
            t = time.time()

            # Use YOLO for object detection
            boxs = yolo.detect_image(image)
            features = encoder(frame, boxs)
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]    # score to 1.0 here


            # # Run non-maxima suppression.
            # # Keep the best detections
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            detection_time.append(time.time()-t)

        # Re-identification: (for each 10 frames)
        if frame_cnt % 10 == 0:

            reid = True
            # If there are detections on the frame
            if len(detections)>0:

                t = time.time()
                # Call the tracker
                tracker.predict()
                tracker.update(detections)
                tracking_time.append(time.time()-t)

                t = time.time()
                # Extract features from actual frame          
                img_features = dict()
                bbox = dict()
                for track in tracker.tracks:

                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue 
                
                    bbox[track.track_id] = track.to_tlbr()
                    image = [frame[int(bbox[track.track_id][1]):int(bbox[track.track_id][3]), int(bbox[track.track_id][0]):int(bbox[track.track_id][2])]]
                    img_features[track.track_id] = extractor(image).data.cpu()

                # Calculate matrix of distances: from actual detection to known features vectors
                # dist ----> [rows = img_features, cols = known_features]
                dist = np.zeros(shape=(len(img_features), len(known_features)))
                for k,i in enumerate(img_features):
                    for j, feats in enumerate(known_features.values()):
                        
                        tmp = np.mean(compute_distance_matrix(img_features[i], feats, metric=distance_metric).numpy())
                        dist[k][j] = tmp

                
                # Check if a detection is known or unknown
                i = 0
                for track in tracker.tracks:

                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue 
                    
                    # Indexes where distance is less than threshold
                    jdx = np.where(dist[i]< threshold)[0]
                    i += 1
                    if len(jdx) >= 1:
                        pickable = []
                        for j in jdx:

                            # Check if the minimun value is on that detection or other
                            kdx = np.where(dist[:,j] == dist[:,j].min())[0][0]
                            pickable.append([kdx, j, dist[kdx, j]])
                        
                        pickable.sort(key=operator.itemgetter(2))
                        final_index[track.track_id] = int(pickable[0][1])+1
                        cv2_addBox(int(pickable[0][1])+1, frame, int(bbox[track.track_id][0]), int(bbox[track.track_id][1]), int(bbox[track.track_id][2]), int(bbox[track.track_id][3]), 3, 2, 2)

                        # Drop column and row
                        dist = np.delete(arr=dist, obj=int(pickable[0][1]), axis=1)

                    else:
                        cv2_addBox(track.track_id+1, frame, int(bbox[track.track_id][0]), int(bbox[track.track_id][1]), int(bbox[track.track_id][2]), int(bbox[track.track_id][3]), 3, 2, 2)

                cv2.putText(frame, "ReID", (int(300), int(40)),0, 5e-3 * 200, (0,255,0),3)
                cv2.putText(frame, "Tracking", (int(300), int(70)),0, 5e-3 * 200, (0,0,255),3)
                reid_time.append(time.time() - t)
                

        # If Re-Identification has not been done, to do tracking
        if reid == False:

            t = time.time()
            # Call the tracker
            tracker.predict()
            tracker.update(detections)
            tracking_time.append(time.time()-t)

            for track in tracker.tracks:

                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 

                # Get detection bounding box 
                bbox = track.to_tlbr()

                # Condicion de si la bbox esta dentro de la imagen
                if bbox[0] >= 0 and bbox[1] >= 0 and bbox[3] < h and bbox[2] < w:
                    
                    try:
                        cv2_addBox(final_index[track.track_id], frame, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), 3, 2, 2)
                        
                    except:
                        pass
                        # cv2_addBox(track.track_id+1, frame, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), 3, 2, 2)

            cv2.putText(frame, "ReID", (int(300), int(40)),0, 5e-3 * 200, (0,0,255),3)
            cv2.putText(frame, "Tracking", (int(300), int(70)),0, 5e-3 * 200, (0,255,0),3)


        # Show tracking result
        # cv2.putText(frame, "FPS: %.3f"%(fps),(int(20), int(40)),0, 5e-3 * 200, (0,255,0),3)
        # cv2.putText(frame, str(frame_cnt),(int(20), int(80)),0, 5e-3 * 200, (0,0,200),3)
        cv2.namedWindow("YOLOv4_Deep_SORT+ReID", 0)
        cv2.resizeWindow('YOLOv4_Deep_SORT+ReID', 1024, 768)
        cv2.imshow('YOLOv4_Deep_SORT+ReID', frame)
        
        frame_cnt += 1

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
        # cv2.waitKey(0)
        
    # End of video
    print('\n\nDetection average time consumption: ', round(sum(detection_time)/len(detection_time), 4))
    print('Tracking average time consumption: ', round(sum(tracking_time)/len(tracking_time), 4))
    print('ReID average time consumption: ', round(sum(reid_time)/len(reid_time), 4))

    # A bit of cleaning!
    video_capture.release()
    if write_video == True:
        output_video.release()
    cv2.destroyAllWindows()

    # Bye :)
    end = time.time()    
    print("\n[Finish]")
    print('\nTotal execution time: ',round(end - start, 3), ' seconds')
    



############################################################################################
############################################################################################
def cv2_addBox(track_id, frame, x1, y1, x2, y2, line_thickness, text_thickness, text_scale):

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