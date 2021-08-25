from sort import *
from collections import defaultdict
import os
import pickle





detection_dir = '../../data/external_detections/FairMOT/new_short_segments_12fps'
output_dir = 'new_short_segments_12fps'
detection_subdirs = next(os.walk(detection_dir))[1]
detection_filenames = [os.path.join(detection_dir,detection_subdir,'saved_detections.pickle') for detection_subdir in detection_subdirs]

for detection_filename in detection_filenames:
    with open(detection_filename,'rb') as f:
        detections = pickle.load(f)

    total_time = 0.0
    total_frames = 0
    if not os.path.exists('output'):
        os.makedirs('output')
    mot_tracker = Sort(max_age=20, 
                        min_hits=3,
                        iou_threshold=0.3) #create instance of the SORT tracker    
    seq_dets = []
    for frame_nb, detections_for_frame in enumerate(detections):
        for detection in detections_for_frame:
            seq_dets.append([frame_nb+1, -1.0, detection[0], detection[1], detection[2]-detection[0], detection[3]-detection[1], detection[4], -1, -1, -1])
    seq_dets = np.array(seq_dets)

    tracklets = defaultdict(list)

    if len(seq_dets):

        for frame in range(int(seq_dets[:,0].max())):
            frame += 1 #detection and frame numbers begin at 1
            dets = seq_dets[seq_dets[:, 0]==frame, 2:7]
            dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
            total_frames += 1


            start_time = time.time()
            trackers = mot_tracker.update(dets)
            cycle_time = time.time() - start_time
            total_time += cycle_time
            
            for d in trackers:
                track_id = int(d[4])
                left, top, width, height = d[0],d[1],d[2]-d[0],d[3]-d[1]
                center_x, center_y = left+width/2, top+height/2
                tracklets[track_id].append((frame, center_x, center_y))

        print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

    else: print('No detections in file')
    tracklets = [tracklet for tracklet in list(tracklets.values())]
    results = []

    for tracker_nb, associated_detections in enumerate(tracklets):
        for associated_detection in associated_detections:
            results.append(
                (associated_detection[0], tracker_nb, associated_detection[1], associated_detection[2]))

    results = sorted(results, key=lambda x: x[0])

    with open(os.path.join(output_dir,detection_filename.split('/')[-2])+'.txt','w') as out_file:

        for result in results:
            out_file.write('{},{},{},{},{},{},{},{},{},{}\n'.format(result[0],
                                                                    result[1]+1,
                                                                    result[2],
                                                                    result[3],
                                                                    -1,
                                                                    -1,
                                                                    1,
                                                                    -1,
                                                                    -1,
                                                                    -1))


            # print('%d,%d,%.2f,%.2f,-1,-1,1,-1,-1,-1'%(frame, tracker_id_to_real_id[track_id], center_x, center_y), file=out_file)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=1)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args

