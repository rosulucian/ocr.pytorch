import os
import torch
import time
import shutil
import numpy as np

from ocr import ocr
from PIL import Image
from glob import glob
from detect.ctpn_utils import gen_anchor

gpu = True
if not torch.cuda.is_available():
    gpu = False
device = torch.device('cuda:0' if gpu else 'cpu')

def single_pic_proc(image_file, anchors, resize=None):
    image = Image.open(image_file).convert('RGB')

    if resize is not None:
        w = int(image.size[0] * resize)
        h = int(image.size[1] * resize)
        dim = (w, h)

        image = image.resize(dim)

        # print(image.size)

    image = np.array(image)
    print(image.shape)

    result, image_framed = ocr(image, anchors)

    return result, image_framed


if __name__ == '__main__':
    image_files = glob('./test_images/*.*')
    result_dir = './test_result'

    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
        
    os.mkdir(result_dir)
    # os.mkdir(result_dir + '/test_images')

    h, w = (720, 1280)
    anchors = gen_anchor((int(h / 16), int(w / 16)), 16)
    anchors = torch.from_numpy(anchors).to(device)

    for image_file in sorted(image_files[:4]):
        t = time.time()

        result, image_framed = single_pic_proc(image_file, anchors)
        # result, image_framed = single_pic_proc(image_file, resize=None)

        print("Mission complete, it took {:.3f}s \n".format(time.time() - t))

        output_file = os.path.join(result_dir, os.path.basename(image_file))

        txt_file = os.path.join(result_dir, os.path.basename(image_file).split('.')[0]+'.txt')

        # print(txt_file)

        txt_f = open(txt_file, 'w+')

        Image.fromarray(image_framed).save(output_file)

        # print("Mission complete, it took {:.3f}s".format(time.time() - t))
        # print("\nRecognition Result:")

        # for key in result:
        #     print(result[key][1])
        #     # txt_f.write(result[key][1]+'\n')
        txt_f.close()
