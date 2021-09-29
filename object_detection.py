# Object Detection

# Importing the libraries
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description = "Object Detection using SSD")
    parser.add_argument("--input_video", "-iv", type = str, required = True,
                            help = "Path to input video")
    parser.add_argument("--output_video", "-ov", type = str, default = "output.mp4",
                            help = "Path to Output Video")
    parser.add_argument("--weights_file_path", "-wfp", type = str, 
                            default = "ssd300_mAP_77.43_v2.pth",
                            help = "Path to SSD Pretrained Weights file")
    return parser.parse_args()

# Defining a function that will do the detections
def detect(frame, net, transform):
    height, width = frame.shape[:2]
    frame_t = transform(frame)[0]
    x = torch.from_numpy(frame_t).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    y = net(x)
    detections = y.data
    scale = torch.Tensor([width, height, width, height])
    # detections = [batch, number of classes, number of occurence, (score, x0, Y0, x1, y1)]
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            pt = (detections[0, i, j, 1:] * scale).numpy()
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            j += 1
    return frame

# Parse Args
args = parse_args()

# Creating the SSD neural network
net = build_ssd('test')
net.load_state_dict(torch.load(args.weights_file_path, map_location = lambda storage, loc: storage))

# Creating the transformation
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

# Doing some Object Detection on a video
reader = imageio.get_reader(args.input_video)
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer(args.output_video, fps = fps)
for i, frame in enumerate(reader):
    frame = detect(frame, net.eval(), transform)
    writer.append_data(frame)
    print(i)
writer.close()
