import os
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
import logging
import argparse
import warnings
import traceback
from timeit import time

import base64
import zmq
import time
import numpy as np

import cv2
import mxnet as mx
import numpy as np
import gluoncv as gcv
from gluoncv import model_zoo

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

from get_color_name import get_color_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('deepsort')
warnings.filterwarnings('ignore')

BASE_FPS = 30


def parse_args():
    parser = argparse.ArgumentParser(description='DeepSORT using mxnet YOLO3.')
    parser.add_argument('--network', type=str, default='yolo3_darknet53_voc',
                        help='Base network name which serves as feature extraction base.')
    parser.add_argument('--short', type=int, default=416,
                        help='Input data shape for short-side, use 320, 416, 608...')
    parser.add_argument('--gpu', type=int, default=1,
                        help='Is GPU accelration available')
    parser.add_argument('--fps', type=int, default=0,
                        help=f'Frame per sec to process based {BASE_FPS} fps,' +
                        ' 0 to consume with no skipped frame')
    parser.add_argument('--src', type=int, default=0,
                        help='Video filename to tracking person')
    parser.add_argument('--out-dir', type=str, default='.',
                        help='Location for missed frame image will be stored.')

    args = parser.parse_args()
    return args


def main(args):
    logger.info('Start Tracking...')

    ctx = mx.gpu(0) if args.gpu else mx.cpu()
    fps = max(0, min(BASE_FPS, args.fps))
    net = model_zoo.get_model(args.network, pretrained=True, ctx=ctx)
    net.reset_class(classes=['person'], reuse_weights=['person'])

    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # feature extractor for deepsort re-id
    encoder = gdet.BoxEncoder()

    metric = nn_matching.NearestNeighborDistanceMetric(
        'cosine', max_cosine_distance, nn_budget
    )
    tracker = Tracker(metric)

    #Socket define
    inbount_port = 13462

    print('make video receiving socket')
    context = zmq.Context()
    inbound_socket = context.socket(zmq.SUB)
    inbound_socket.bind('tcp://*:'+str(inbount_port))
    inbound_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))
    print('waiting video frame')

    #outbound_socket = context.socket(zmq.PUB)
    #outbound_socket.bind('tcp://192.168.0.9:13462')
    #outbound_socket.connect('tcp://192.168.0.14:13462')

    frame_interval = BASE_FPS // fps if fps > 0 else 0
    frame_index = 0
    while True:

        frame = inbound_socket.recv_string()
        frame = base64.b64decode(frame)
        npimg = np.fromstring(frame, dtype=np.uint8)
        source = cv2.imdecode(npimg, 1)

        if 0 < fps and frame_index % frame_interval != 0:
            frame_index += 1
            continue

        x, img = gcv.data.transforms.presets.yolo.transform_test(
            mx.nd.array(source).astype('uint8'),
            short=args.short,
        )

        class_IDs, det_scores, det_boxes = net(x.as_in_context(ctx))

        boxs = []
        person = mx.nd.array([0])
        score_threshold = mx.nd.array([0.5])
        for i, class_ID in enumerate(class_IDs[0]):
            if class_ID.copyto(mx.gpu(0)) == person.copyto(mx.gpu(0)) and det_scores[0][i].copyto(mx.gpu(0)) >= score_threshold.copyto(mx.gpu(0)):
                boxs.append(det_boxes[0][i].asnumpy())

        if boxs:
            features = encoder(img, boxs)
        else:
            features = np.array([])

        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature)
                      for bbox, feature in zip(boxs, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores
        )
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        frame_index += 1

        # store original scene
        cv2.imwrite(os.path.join(args.out_dir, f'{frame_index}.jpg'), img)

        show_img = img.copy()
        # check missed
        for track in tracker.tracks:
            bbox = [max(0, int(x)) for x in track.to_tlbr()]
            if not track.is_confirmed() or track.time_since_update > 1:
                if 2 <= track.time_since_update < 10: pass
                    #img_write (why?)
                logger.info('Skipped by time_since_update')
            logger.info(f'Frame #{frame_index} - Id: {track.track_id}')
            cname = get_color_name(show_img[bbox[1]:bbox[3],bbox[0]:bbox[2]])
            cv2.rectangle(show_img,
                          (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                          (255,255,255), 2)
            cv2.putText(show_img,
                        str(track.track_id)+cname,
                        (bbox[0], bbox[1]+30),
                        0, 5e-3 * 200,
                        (0,255,0), 2)

        # show image
        cv2.imwrite(os.path.join(args.out_dir, f'anno-{frame_index}.jpg'), show_img)
        cv2.imshow('', show_img)

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    logger.info(f'Missed obj: {tracker.missed_obj}, Missed frame: {tracker.missed_frame}')

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    main(args)
