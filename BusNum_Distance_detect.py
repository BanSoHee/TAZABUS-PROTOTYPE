"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
    ex - python BusNum_Distance_detect.py --source 0 --weights BusNumbest.pt --conf 0.3
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync

#################################### add start code ##########################

from gtts import gTTS # gtts 사용
from playsound import playsound # playsound 음성 출력 사용
import pygame # Load the popular external library
import os

freq = 24000    # sampling rate 16000(Naver TTS), 24000(google TTS)
bitsize = -16   # signed 16 bit. support 8, -8, 16, -16
channels = 1    # 1 is mono, 2 is stereo
buffer = 2048   # number of samples (experiment to get right sound)

# focal length finder function
# < 초점 길이 계산 함수 >
# measured_distance = 물체와 카메라 사이의 거리 = 76.2cm
# real_width = 실제 물체 폭 (예 - 내 얼굴 폭 = 14.3cm)
# width_in_rf_image = 이미지에서의 물체 폭
def focal_length(measured_distance, real_width, width_in_rf_image):
    focal_length_value = (width_in_rf_image * measured_distance) / real_width
    return focal_length_value

# distance estimation function
# < 인수를 사용하여 물체와 카메라 사이의 거리를 추정하는 함수 >
# focal_length = focal_length 함수에 의해 반환되는 값
# real_face_width = 실제 물체의 폭 (예 - 버스 폭 = 320cm)
# face_width_in_frame = 이미지에 있는 물체의 폭 (-> yolo에서 bounding box의 y1, y2 좌표 차이값으로 계산하기)
def distance_finder(focal_length, real_face_width, face_width_in_frame):
    distance = (real_face_width * focal_length)/face_width_in_frame
    return distance

#################################### add start code ##########################

@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        ):

    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, pt, onnx = False, w.endswith('.pt'), w.endswith('.onnx')  # inference type
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        check_requirements(('onnx', 'onnxruntime'))
        import onnxruntime
        session = onnxruntime.InferenceSession(w, None)
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        if pt:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        elif onnx:
            img = img.astype('float32')
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        t1 = time_sync()
        if pt:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)[0]
        elif onnx:
            pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_sync()

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

#################################### add start code ##########################
        list_x1 = []
        list_num = []
        Distance = 0
#################################### add end code ##########################

        # Process predictions
        for i, det in enumerate(pred):  # detections per image # 모든 image loop

            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det): # 탐지되는 모든 물체 per frame, with for loop

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        
                        # 영상에 바운딩 박스 표시
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)

#################################### add start code ##########################

                        # Bounding box 좌표
                        x1 = int(xyxy[0].item()) # bounding box 좌표 x1
                        y1 = int(xyxy[1].item()) # bounding box 좌표 y1
                        x2 = int(xyxy[2].item()) # bounding box 좌표 x2
                        y2 = int(xyxy[3].item()) # bounding box 좌표 y2

                        confidence_score = conf # 임계값 설정 
                        class_index = cls # class 번호
                        object_name = names[int(cls)] # list에서 가져온 객체 이름 

                        # bus를 제외한 숫자만 list에 bounding box 좌표 추가
                        if object_name != 'bus':
                            list_x1.append(x1)
                            list_num.append(object_name)
                        
                        if object_name == 'bus':
                            delta_y = y2 - y1          # 이미지에서 인식한 버스 길이
                            ex_distance = 30           # 초점 길이를 계산하기 위한 distance 예
                            ex_width = 12.5            # 초점 길이를 계산하기 위한 물체의 width 예
                            ex_width_in_image = 215    # 초점 길이를 계산하기 위한 카메라에서의 물체 width(y2 - y1) 예
                            real_width = 12.5          # 실제 버스의 width

                            # < 초점 길이 계산 함수 > 호출 -------------> 초점 길이 계산
                            focal_length_found = focal_length(ex_distance, ex_width, ex_width_in_image)
                            # < 인수를 사용하여 물체와 카메라 사이의 거리를 추정하는 함수 > 호출 ------------> 거리 예측
                            Distance = distance_finder(focal_length_found, real_width, delta_y) # 단위 cm

                            # 측정한 거리 결과 저장
                            '''
                            f = open('detected_BusDistance.txt', 'w')
                            distancedata = 'Distace = {0}cm\n'.format(int(Distance))
                            f.write(distancedata)
                            f.close '''

                        # 객체 이름, class 번호, bounding box 좌표 출력
                        #print('detected object name = ', object_name)
                        #print('class index = ', class_index)
                        #print('bounding box = ', x1, y1, x2, y2)

                        original_img = im0 # original image
                        cropped_img = im0[y1:y2, x1:x2] # bounding box imag

                        time_stamp = str(int(time.time()))
                        #cv2.imwrite(time_stamp + '.png', original_img) # detected object image 폴더에 저장

#################################### add end code ##########################

                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0) # 창(Window)을 띄워서 image 출력
                if cv2.waitKey(1) == ord('q'):  # q to quit, 0.001초만큼 키보드 입력 기다림
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

#################################### add start code ##########################

                # 인식된 숫자들의 bounding box 좌표를 오름차순으로 정렬 per frame -> 버스 번호 인식
                for k in range(len(list_x1)):
                    for j in range(len(list_x1)-1):
                        if list_x1[j] > list_x1[j+1]:
                            list_x1[j], list_x1[j+1] = list_x1[j+1], list_x1[j]
                            list_num[j], list_num[j+1] = list_num[j+1], list_num[j]

                # 영어로 된 숫자 class name을 숫자로 변경
                bus_num = ''
                bus_number = ''
                for i in range(len(list_num)):
                    if list_num[i] == 'zero':
                        bus_number = '0'
                    elif list_num[i] == 'one':
                        bus_number = '1'
                    elif list_num[i] == 'two':
                        bus_number = '2'
                    elif list_num[i] == 'three':
                        bus_number = '3'
                    elif list_num[i] == 'four':
                        bus_number = '4'
                    elif list_num[i] == 'five':
                        bus_number = '5'
                    elif list_num[i] == 'six':
                        bus_number = '6'
                    elif list_num[i] == 'seven':
                        bus_number = '7'
                    elif list_num[i] == 'eight':
                        bus_number = '8'
                    elif list_num[i] == 'nine':
                        bus_number = '9'

                    bus_num = bus_num + bus_number

                # 인식된 버스 번호를 txt 파일에 저장
                '''
                f = open('‪detected_BusNum.txt', 'w')
                data = '{0}\n'.format(bus_num)
                if bus_num != '':
                    f.write(data)
                    f.close '''

                # 키보드 입력값 존재 + 버스(거리) 인식 + 버스 번호 인식이 모두 되었을 경우 전달할 정보를 음성으로 출력
                # cv2.waitKey(1) == ord('a')
                if (cv2.waitKey(1) == ord('a') and bus_num != '' and int(Distance) != 0):  # q to quit, 0.001초만큼 키보드 입력 기다림
                    
                    print('{0}번 버스가 {1}cm 앞에 있습니다'.format(bus_num, int(Distance)))

                    # 전달 정보 저장
                    f1 = open('fin_detected_Bus.txt', 'w')
                    fin_data = '{0}번 버스가 {1}cm 앞에 있습니다\n'.format(bus_num, int(Distance))
                    f1.write(fin_data)
                    f1.close

                    # with gTTS, text -> speech
                    tts = gTTS(text = fin_data, lang = 'ko', slow = False) 
                    tts.save('BusgTTS.mp3')

                    pygame.mixer.init(freq, bitsize, channels, buffer)
                    pygame.mixer.music.load("BusgTTS.mp3")
                    pygame.mixer.music.play()

                    clock = pygame.time.Clock()
                    while pygame.mixer.music.get_busy():
                        clock.tick(30)
                    pygame.mixer.quit() 
                
#################################### add end code ##########################

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(f'Done. ({time.time() - t0:.3f}s)')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    return opt


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)