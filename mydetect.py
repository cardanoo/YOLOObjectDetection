from __future__ import division
from models import *
from utils.utils import *
from utils.datasets import *
import cv2
from PIL import Image
import torch
from torch.autograd import Variable


def detectObj():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = Darknet("config/yolov3.cfg", img_size=416).to(device)
    model.load_darknet_weights("weights/yolov3.weights")
    model.eval()  # Set in evaluation mode

    classes = load_classes("data/coco.names")  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("/home/letmesleep/video/person.mp4")

    while True:
        ret, frame = cap.read()
        cvimgframe = frame
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = transforms.ToTensor()(frame)

        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, 416)
        input_imgs = Variable(img.type(Tensor))
        input_imgs = input_imgs.unsqueeze(0)

        try:
            # Get detections
            with torch.no_grad():
                detections = model(input_imgs)
                detections = non_max_suppression(detections, 0.8, 0.4)

            for i in range(0, len(detections)):
                detections[i] = detections[i].numpy()

            detections = torch.from_numpy(np.array(detections))
            detections = detections.squeeze(0)
        except:
            continue
            pass




        # img = np.array(frame)
        img = cvimgframe
        # Draw bounding boxes and labels of detections
        if detections is not None:

            # Rescale boxes to original image
            detections = rescale_boxes(detections, 416, img.shape[:2])
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                cvimgframe = cv2.rectangle(cvimgframe,(x1,y1),(x2,y2),(0,0,255),5)
                cvimgframe = cv2.putText(cvimgframe,classes[int(cls_pred)],(x1,y1),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                cv2.imshow("",cvimgframe)



        # 读取内容
        if cv2.waitKey(10) == ord("q"):
            break

    # 随时准备按q退出
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detectObj()
    pass
