import sys
import cv2  # imread
import torch
import numpy as np
from os.path import realpath, dirname, join

from net import SiamRPNBIG
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect
from util_test import *
import linecache

class Tracker(object):
    def __init__(self,
                 path_seq,
		 num_frames,
		 gt_first):
        
        self.path_seq = path_seq
        self.num_frames = num_frames
        self.gt_first = gt_first

        # load net
        self.net = SiamRPNBIG()
        # self.net.load_state_dict(torch.load("./SiamRPNBIG.model"))
        self.net.eval().cuda()

        #self.testing_config = testing_config
        self.cur_seq_name = os.path.split(path_seq)[1]
        self.cur_frame = None

    def on_tracking(self):        
        # warm up
        for i in range(10):
            self.net.template(torch.autograd.Variable(torch.FloatTensor(1, 3, 127, 127)).cuda())
            self.net(torch.autograd.Variable(torch.FloatTensor(1, 3, 255, 255)).cuda())

        i = 1
        pred_bbx = self.gt_first
        print("{}th frame: {} {} {} {}".format(i, pred_bbx[0],pred_bbx[1], pred_bbx[2], pred_bbx[3]))
        cx, cy, w, h = pred_bbx[0]+pred_bbx[2]/2.0, pred_bbx[1]+pred_bbx[3]/2.0, pred_bbx[2], pred_bbx[3]
        i += 1

        target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
        im = cv2.imread(self.path_seq + '/imgs/0001.jpg')  # HxWxC
        state = SiamRPN_init(im, target_pos, target_sz, self.net)  # init tracker
        
        while i <= self.num_frames:
            self.index_frame = i
            im = cv2.imread(self.path_seq + '/imgs/' + str(i).zfill(4) + '.jpg')
            state = SiamRPN_track(state, im)

            # convert cx, cy, w, h into rect
            res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            print(f"{i}th frame: ", res)
            i += 1


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('*****************TEST PHASE********************')
    import datetime
    testing_date = datetime.datetime.now().strftime('%b-%d-%y_%H:%M:%S')

    seq_list_path = '../data/whole_list.txt'

    seq_path_list = np.genfromtxt(seq_list_path, dtype='S', usecols=0)
    num_frames_list = np.genfromtxt(seq_list_path, dtype=int, usecols=1)
    if seq_path_list.ndim == 0:
        seq_path_list = seq_path_list.reshape(1)
        num_frames_list = num_frames_list.reshape(1)

    assert len(seq_path_list) == len(num_frames_list)
    total_seqs = len(seq_path_list)
    total_frames = sum(num_frames_list)
    for seq_index in range(len(seq_path_list)):
        path_seq = seq_path_list[seq_index].decode('utf-8')
        num_frames = num_frames_list[seq_index]

        seq_name = os.path.split(path_seq)[1]
        print(f"\nprocessing Sequence {seq_name} with {num_frames} frames...")

        global gt_file_name
        gt_file_name = path_seq + '/' + seq_name + "_gt.txt"

        gt_entry = linecache.getline(gt_file_name, 1)
        gt_first = parse_gt_entry(gt_entry)

        tracker = Tracker(path_seq=path_seq,
	                  num_frames=num_frames,
			  gt_first=gt_first)
        tracker.on_tracking()
