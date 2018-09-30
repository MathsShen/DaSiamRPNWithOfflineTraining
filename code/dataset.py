import numpy as np
import numpy.random as npr
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import os
import os.path as osp
import linecache
import cv2


class DaSiamTrainingSet(Dataset):
	def __init__(self, transform, z_size, x_size):
		self.root_path = "/home/lishen/Experiments/CLSTMT/dataset/test_set/OTB100/"
		self.domain2nseq = {}
		self.create_domain2nseq(osp.join(self.root_path, "whole_list.txt"))
		self.transform = transform
		self.z_size = z_size
		self.x_size = x_size

	def create_domain2nseq(self, list_fpath):
		with open(list_fpath, 'r') as f:
			while True:
				line = f.readline()
				if not line:
					break
				splits = line.strip().split()
				domain_name = splits[0].split('/')[-1]
				nseq = int(splits[1])
				self.domain2nseq[domain_name] = nseq		

	def __len__(self):
		return sum(self.domain2nseq.values()) // len(self.domain2nseq.values())

	def __getitem__(self, item):
		domain_list = list(self.domain2nseq.keys())
		domain_name = npr.choice(domain_list, size=1)[0]
		num_frames = self.domain2nseq[domain_name]
		
		pair_frame_nos = npr.choice(range(1, num_frames+1), size=2, replace=False)
		z_frame_no, x_frame_no = min(pair_frame_nos), max(pair_frame_nos)

		domain_dir = osp.join(self.root_path, "sequences", domain_name)
		gt_fpath = osp.join(domain_dir, domain_name + '_gt.txt')
		z_gt_bbx = tuple(map(int, linecache.getline(gt_fpath, z_frame_no).split()))
		x_gt_bbx = tuple(map(int, linecache.getline(gt_fpath, x_frame_no).split()))
	
		z_frame_img_name = str(z_frame_no).zfill(4) + '.jpg'
		x_frame_img_name = str(x_frame_no).zfill(4) + '.jpg'
		z_frame = cv2.imread(osp.join(domain_dir, 'imgs', z_frame_img_name))
		x_frame = cv2.imread(osp.join(domain_dir, 'imgs', x_frame_img_name))
	
		#print(z_gt_bbx)
		z = crop_roi(z_frame, convert_bbx2box(z_gt_bbx))
		z = cv2.resize(z, self.z_size)
	
		x_gt_box = convert_bbx2box(x_gt_bbx)
		sr_box = gen_sr_box(x_frame, x_gt_box)
		x = crop_roi(x_frame, sr_box)
		x = cv2.resize(x, self.x_size)
		
		translated_x_gt_box = np.array(trans_coord(sr_box, x_gt_box))

		sample = {
			'template': self.transform(z),
			'search_region': self.transform(x),
			'gt_box': translated_x_gt_box
		}
		return sample


def trans_coord(sr_box, x_gt_box):
	return (x_gt_box[0]-sr_box[0], x_gt_box[1]-sr_box[1], x_gt_box[2]-sr_box[0], x_gt_box[3]-sr_box[1])


def gen_sr_box(frame, gt_box):
	gt_x1, gt_y1, gt_x2, gt_y2 = gt_box
	h, w = gt_y2-gt_y1+1, gt_x2-gt_x1+1
	rand_cx = np.random.randint(gt_x1, gt_x2+1)
	rand_cy = np.random.randint(gt_y1, gt_y2+1)

	sr_x1, sr_y1, sr_x2, sr_y2 = rand_cx-w, rand_cy-h, rand_cx+w, rand_cy+h
	H, W = frame.shape[:2]
	return max(0, sr_x1), max(0, sr_y1), min(sr_x2, W-1), min(sr_y2, H-1)


def convert_bbx2box(bbx):
	x, y, w, h = bbx
	return (x, y, x+w-1, y+h-1)


def convert_box2bbx(box):
	x1, y1, x2, y2 = box
	return (x1, y1, x2-x1+1, y2-y1+1)


def crop_roi(frame, box):
	return frame[box[1]:box[3]+1, box[0]:box[2]+1, :]


def IoU(prop, gt):
	x1, y1, w1, h1 = map(prop, float)
	x2, y2, w2, h2 = map(gt, float)
	startx, endx = min(x1, x2), max(x1+w1, x2+w2)
	starty, endy = min(y1, y2), max(y1+h1, y2+h2)
	width = w1 + w2 - (endx - startx)
	height = h1 + h2 - (endy - starty)
	if width <= 0 or height <= 0:
		return 0
	else:
		area = width * height
		return 1.0*area/(w1*h1+w2*h2-area)


def load_data(batch_size, z_size, x_size):
	transform = transforms.Compose([
		# convert a PIL.Image instance of value range [0, 255] or an numpy.ndarray of shape (H, W, C) 
		#     into a torch.FloatTensor of shape (C, H, W) with value range (0, 1.0).
		transforms.ToTensor(), 
	])

	datasets = {
		'train': DaSiamTrainingSet(transform, z_size, x_size)
	}

	dataloaders = {ds: DataLoader(datasets[ds],
								  batch_size=batch_size,
								  shuffle=False,
								  pin_memory=True,
								  num_workers=8) for ds in datasets}

	return dataloaders


if __name__ == "__main__":
	da_siam_set = DaSiamTrainingSet(transforms.ToTensor(), (127, 127), (255, 255))

	domain_list = list(da_siam_set.domain2nseq.keys())
	domain_name = npr.choice(domain_list, size=1)[0]
	num_frames = da_siam_set.domain2nseq[domain_name]
	
	pair_frame_nos = npr.choice(range(num_frames), size=2, replace=False)
	z_frame_no, x_frame_no = min(pair_frame_nos), max(pair_frame_nos)

	domain_dir = osp.join(da_siam_set.root_path, "sequences", domain_name)
	gt_fpath = osp.join(domain_dir, domain_name + '_gt.txt')

	z_gt_bbx = tuple(map(int, linecache.getline(gt_fpath, z_frame_no).split()))
	x_gt_bbx = tuple(map(int, linecache.getline(gt_fpath, x_frame_no).split()))

	z_frame_img_name = str(z_frame_no).zfill(4) + '.jpg'
	x_frame_img_name = str(x_frame_no).zfill(4) + '.jpg'
	z_frame = cv2.imread(osp.join(domain_dir, 'imgs', z_frame_img_name))
	x_frame = cv2.imread(osp.join(domain_dir, 'imgs', x_frame_img_name))
		
	import pdb
	pdb.set_trace()

	z = crop_roi(z_frame, convert_bbx2box(z_gt_bbx))
	z = cv2.resize(z, da_siam_set.z_size)

	x_gt_box = convert_bbx2box(x_gt_bbx)
	sr_box = gen_sr_box(x_frame, x_gt_box)
	x = crop_roi(x_frame, sr_box)
	x = cv2.resize(x, da_siam_set.x_size)
	
	translated_x_gt_box = trans_coord(sr_box, x_gt_box)
	
	print('DONE.')
		
