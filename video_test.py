# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import cv2
import mmcv

from mmdet.apis import inference_detector, init_detector


# def parse_args():
#     parser = argparse.ArgumentParser(description='MMDetection video demo')
#     parser.add_argument('video', help='Video file')
#     parser.add_argument('config', help='Config file')
#     parser.add_argument('checkpoint', help='Checkpoint file')
#     parser.add_argument(
#         '--device', default='cuda:0', help='Device used for inference')
#     parser.add_argument(
#         '--score-thr', type=float, default=0.6, help='Bbox score threshold')
#     parser.add_argument('--out', type=str, help='Output video file')
#     parser.add_argument('--show', action='store_true', help='Show video')
#     parser.add_argument(
#         '--wait-time',
#         type=float,
#         default=1,
#         help='The interval of show (s), 0 is block')
#     args = parser.parse_args()
#     return args


def main():
    # args = parse_args()
    video = 'movie/movie.mp4'
    config= 'checkpoints/faster_rcnn_r50_fpn_2x_coco.py'
    checkpoint = 'checkpoints/latest.pth'
    device = 'cuda:0'
    out = 'result.mp4'
    show = 0
    wait_time = 0
    score_thr = 0.8
    assert out or show, \
        ('Please specify at least one operation (save/show the '
         'video) with the argument "--out" or "--show"')

    model = init_detector(config, checkpoint, device=device)

    video_reader = mmcv.VideoReader(video)
    video_writer = None
    if out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            out, fourcc, video_reader.fps,
            (video_reader.width, video_reader.height))

    for frame in mmcv.track_iter_progress(video_reader):
        result = inference_detector(model, frame)
        frame = model.show_result(frame, result, score_thr=score_thr)
        if show:
            cv2.namedWindow('video', 0)
            mmcv.imshow(frame, 'video', wait_time)
        if out:
            video_writer.write(frame)

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
