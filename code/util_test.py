#import matplotlib as mpl 
#import matplotlib.cbook as cbook
import os
import cv2
import numpy as np
#import matplotlib.pyplot as plt

import datetime

"""
UTIL OF TEST version 1.0
"""

def convert_box2bbx(box):
	x1, y1, x2, y2 = box
	return (x1, y1, x2-x1+1, y2-y1+1)


def convert_bbx2box(bbx):
	x, y, w, h = bbx
	return (x, y, x+w-1, y+h-1)


def save_pred_bboxes_v2(pred_tuple_list, seq_name, testing_date):
	source_path = '/home/code/xuxiaqing/dataset/OTB100/{}/imgs'.format(seq_name)
	saving_path = './output/tracking_res/OTB100/{}/{}/'.format(testing_date, seq_name)

	if not os.path.exists(saving_path):
		os.makedirs(saving_path)

	list_file = open(os.path.join(saving_path, 'preds.txt'), 'w')
	for index, pred_tuple in enumerate(pred_tuple_list):
		pred_bbx, score = pred_tuple 
		raw_img_name = '%s' % (str(index+1).zfill(4)) + '.jpg'
		raw_img_path = os.path.join(source_path, raw_img_name)
		frame = cv2.imread(raw_img_path)

		left = int(round(pred_bbx[0]))
		top = int(round(pred_bbx[1]))
		right = int(round(pred_bbx[0] + pred_bbx[2] - 1))
		bottom = int(round(pred_bbx[1] + pred_bbx[3] - 1))

		##############################################################################
		cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 2)
		cv2.putText(frame, str(score), (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, 8)
		# cv2.putText(frame, str(thresh_eps), (right, bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 97, 255), 1, 8)
		cv2.imwrite(os.path.join(saving_path, raw_img_name), frame)

		entry = str(pred_bbx[0]) + ' ' + str(pred_bbx[1]) + ' ' + str(pred_bbx[2]) + ' ' + str(pred_bbx[3])
		list_file.write(entry + '\n')

	list_file.close()
	print('\nPredictions of Seq ' + seq_name + ' saved.')

def save_pred_bboxes(pred_bbx_list, score_list, seq_name):
	assert len(pred_bbx_list) == len(score_list), 'length of lists not equal'

	saving_path = cfg.ROOT_DIR + './output/tracking_res/{}/'.format(cfg.TEST.BENCHMARK_NAME) + seq_name
	source_path = '/home/lishen/Experiments/siamese_tracking_net/dataset/test_set/OTB100/' + seq_name + '/imgs'

	if not os.path.exists(saving_path):
		os.makedirs(saving_path)

	list_file = open(os.path.join(saving_path, 'preds.txt'), 'w')
	for index, pred_bbx in enumerate(pred_bbx_list):
		raw_img_name = '%s' % (str(index+1).zfill(4)) + '.jpg'
		raw_img_path = os.path.join(source_path, raw_img_name)
		frame = cv2.imread(raw_img_path)

		left = int(round(pred_bbx[0]))
		top = int(round(pred_bbx[1]))
		right = int(round(pred_bbx[0] + pred_bbx[2] - 1))
		bottom = int(round(pred_bbx[1] + pred_bbx[3] - 1))

		score = score_list[index]

		##############################################################################
		cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
		cv2.putText(frame, '{}'.format(score), (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, 8)
		cv2.imwrite(os.path.join(saving_path, raw_img_name), frame)

		entry = str(pred_bbx[0]) + ' ' + str(pred_bbx[1]) + ' ' + str(pred_bbx[2]) + ' ' + str(pred_bbx[3])
		list_file.write(entry + '\n')

	list_file.close()
	print('\nPredictions of Seq ' + seq_name + ' saved.')

def save_pred_bboxes_bbxr_exclusive(pred_bbx_list_before_reg, pred_bbx_list, score_list, seq_name):
	assert len(pred_bbx_list) == len(score_list), 'length of lists not equal'
	assert len(pred_bbx_list) == len(pred_bbx_list_before_reg), 'length of lists not equal'

	saving_path = cfg.ROOT_DIR + '/output/tracking_res/{}/'.format(cfg.TEST.BENCHMARK_NAME) + seq_name
	source_path = '/home/lishen/Experiments/siamese_tracking_net/dataset/test_set/Benchmark/' + seq_name + '/imgs'

	if not os.path.exists(saving_path):
		os.makedirs(saving_path)

	list_file = open(os.path.join(saving_path, 'preds.txt'), 'w')
	for index, pred_bbx in enumerate(pred_bbx_list):
		raw_img_name = '%s' % (str(index+1).zfill(4)) + '.jpg'
		raw_img_path = os.path.join(source_path, raw_img_name)
		frame = cv2.imread(raw_img_path)

		left = int(round(pred_bbx[0]))
		top = int(round(pred_bbx[1]))
		right = int(round(pred_bbx[0] + pred_bbx[2] - 1))
		bottom = int(round(pred_bbx[1] + pred_bbx[3] - 1))

		score = score_list[index]

		## predicted bbx before regression ##
		left_before = int(round(pred_bbx_list_before_reg[index][0]))
		top_before = int(round(pred_bbx_list_before_reg[index][1]))
		right_before = int(round(pred_bbx_list_before_reg[index][0] + pred_bbx_list_before_reg[index][2] - 1))
		bottom_before = int(round(pred_bbx_list_before_reg[index][1] + pred_bbx_list_before_reg[index][3] - 1))
		cv2.rectangle(frame, (left_before, top_before), (right_before, bottom_before), (0, 0, 255), 2)
		
		##############################################################################
		cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
		cv2.putText(frame, '{}'.format(score), (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, 8)
		cv2.imwrite(os.path.join(saving_path, raw_img_name), frame)

		entry = str(pred_bbx[0]) + ' ' + str(pred_bbx[1]) + ' ' + str(pred_bbx[2]) + ' ' + str(pred_bbx[3])
		list_file.write(entry + '\n')

	list_file.close()
	print('\nPredictions of Seq ' + seq_name + ' saved.')


def parse_gt_entry(gt_entry):
	split_gt_entry = gt_entry.split() #','

	left = float(split_gt_entry[0])
	top = float(split_gt_entry[1])
	width = float(split_gt_entry[2])
	height = float(split_gt_entry[3])
	return (left, top, width, height)


def crop_roi(frame, bbx):
	#box = (x1, y1, x2, y2)
	box = (int(round(bbx[0])), int(round(bbx[1])), int(round(bbx[0]+bbx[2])), int(round(bbx[1]+bbx[3])))
	return frame[box[1]:box[3], box[0]:box[2], :]


def crop_and_save(seq_name, raw_img, idx_frame, samples, type_str):
	root_dir = cfg.CODE_ROOT_DIR + '/output/finetuning_data/{}'.format(cfg.TEST.BENCHMARK_NAME)

	tar_dir = root_dir + '/' + seq_name + '/' + str(idx_frame) + '/' + type_str
	if not os.path.exists(tar_dir):
		os.makedirs(tar_dir)

	for idx in xrange(samples.shape[0]):
		bbx_sample = samples[idx, :]
		box = (int(round(bbx_sample[0])), int(round(bbx_sample[1])), int(round(bbx_sample[0]+bbx_sample[2]-1)), int(round(bbx_sample[1]+bbx_sample[3]-1)))
		patch = raw_img[box[1]:box[3], box[0]:box[2], :]
		path_patch = tar_dir + '/' + str(idx+1) + '.jpg'
		cv2.imwrite(path_patch, patch)


def sub_gen(gt, raw_img_size):
	sub_pos_samples = np.zeros((25, 4), dtype=np.float32)
	index = 0

	right_img = raw_img_size[1]
	bottom_img = raw_img_size[0]
	
	for dx in np.arange(-2, 3):
		for dy in np.arange(-2, 3):
			'''determine a new bounding box'''
			left = gt[0] + dx
			top = gt[1] + dy
			width = gt[2] + np.abs(dx)
			height = gt[3] + np.abs(dy)

			'''in case it lies beyond the boundary'''
			left = min(right_img, max(0, left))	# 0 <= left <= right_img
			top = min(bottom_img, max(0, top))
			
			right = left + width
			right = min(right_img, max(0, right))

			bottom = top + height
			bottom = min(bottom_img, max(0, bottom))

			width = right - left
			height = bottom - top

			sub_pos_samples[index, :] = np.array([[left, top, width, height]])
			index += 1

	return sub_pos_samples


def gen_positive_samples(gt, raw_img_size):
	'''This function will generate 50 positive samples using pixel-difference type'''

	#Generate the first 25 positives
	first_sub_pos_samples = sub_gen(gt, raw_img_size)

	#Generate the second 25 positives
	shifted_gt = (gt[0]-1, gt[1]-1, gt[2]+2, gt[3]+2)
	second_sub_pos_samples = sub_gen(shifted_gt, raw_img_size)

	pos_samples = np.vstack((first_sub_pos_samples, second_sub_pos_samples))
	return pos_samples


def IoU(prop, gt):
    x1, y1, w1, h1 = float(prop[0]), float(prop[1]), float(prop[2]), float(prop[3])
    x2, y2, w2, h2 = float(gt[0]), float(gt[1]), float(gt[2]), float(gt[3])
    startx, endx = min(x1, x2), max(x1 + w1, x2 + w2)
    starty, endy = min(y1, y2), max(y1 + h1, y2 + h2)
    width = w1 + w2 - (endx - startx)
    height = h1 + h2 - (endy - starty)
    if width <= 0 or height <= 0:
        return 0
    else :
        area = width * height
        return 1.0 * area / (w1*h1 + w2*h2 - area)


def post_proc(random_scalar):
	return max(-1, min(1, 0.5 * random_scalar))	#restrict it within the interval [-1, 1]


def gen_samples_box(sampling_type, 
					gt, 
					num_samples, 
					raw_img_size, 
					base_scalar=1.05, 
					trans_fac=0.1, 
					scale_fac=5, 
					pos_sampling=True, 
					pos_thresh=0.7, 
					neg_thresh=0.3, 
					iou_thresh_ignored=False):
		
	H = raw_img_size[0]
	W = raw_img_size[1]

	#sample = (cx, cy, w, h), where (cx, cy) is the coodinate of the gt image
	sample = np.array([gt[0]+gt[2]/2, gt[1]+gt[3]/2, gt[2], gt[3]], dtype = np.float32)
	samples = np.tile(sample, (num_samples, 1))	
	
	idx = 0
	while idx < num_samples:
		curr_sample = samples[idx, :].copy()

		if sampling_type == 'gaussian':
			lt_increment = trans_fac * round(np.mean(gt[2:4])) * np.array([post_proc(np.random.randn(1,)), post_proc(np.random.randn(1,))])
			curr_sample[:2] = curr_sample[:2] + lt_increment.reshape(2,)

			randn_vec = np.array([post_proc(np.random.randn(1,)), post_proc(np.random.randn(1,))])
			wh_factor = base_scalar ** (scale_fac * randn_vec)
			curr_sample[2:] = curr_sample[2:] * wh_factor.reshape(2,)

		elif sampling_type == 'uniform':	#uniform distribution within a searching area 2.5 times the size of bbx
			sr_ratio = 3.5 #cfg.TEST.UNIFORM_SAMPLING_RANGE_RATIO	#twice or 2.5 times???

			randn_vec = np.array([post_proc(np.random.randn(1,)), post_proc(np.random.randn(1,))])
			wh_factor = base_scalar ** (scale_fac * randn_vec)
			curr_sample[2:] = curr_sample[2:] * wh_factor.reshape(2,)

			cx_bound = (curr_sample[0]-curr_sample[2]*(sr_ratio/2), curr_sample[0]+curr_sample[2]*(sr_ratio/2))
			cy_bound = (curr_sample[1]-curr_sample[3]*(sr_ratio/2), curr_sample[1]+curr_sample[3]*(sr_ratio/2))
			cx = (cx_bound[1] - cx_bound[0]) * np.random.random_sample() + cx_bound[0]
			cy = (cy_bound[1] - cy_bound[0]) * np.random.random_sample() + cy_bound[0]

			curr_sample[0] = cx
			curr_sample[1] = cy

		elif sampling_type == 'whole':	#uniform distribution within the whole image
			randn_vec = np.array([post_proc(np.random.randn(1,)), post_proc(np.random.randn(1,))])
			wh_factor = base_scalar ** (scale_fac * randn_vec)
			curr_sample[2:] = curr_sample[2:] * wh_factor.reshape(2,)

			w = curr_sample[2]
			h = curr_sample[3]
			curr_sample[0] = (W - w) * np.random.random_sample() + w / 2.0
			curr_sample[1] = (H - h) * np.random.random_sample() + h / 2.0

		'''In case that samples experience abrupt scaling variation...''' ##########
		curr_sample[2] = max(5, min(W-5, curr_sample[2])) 	#w max(gt[2]/5.0, min(gt[2]*5.0, curr_sample[2]))
		curr_sample[3] = max(5, min(H-5, curr_sample[3]))	#h max(gt[3]/5.0, min(gt[2]*5.0, curr_sample[3]))

		half_w, half_h = curr_sample[2]/2.0, curr_sample[3]/2.0

		# bbx_sample = np.array([curr_sample[0]-curr_sample[2]/2, curr_sample[1]-curr_sample[3]/2, curr_sample[2], curr_sample[3]])
		# bbx_sample[0] = max(0, min(W-bbx_sample[2]-1, bbx_sample[0]))
		# bbx_sample[1] = max(0, min(H-bbx_sample[3]-1, bbx_sample[1]))

		"""The centre coordinate of candidate box should lie within the [half_w, W-half_w-1]x[half_h, H-half_h-1]"""
		curr_sample[0] = max(half_w, min(W-half_w-1, curr_sample[0]))
		curr_sample[1] = max(half_h, min(H-half_h-1, curr_sample[1]))

		x1, y1 = curr_sample[0]-half_w, curr_sample[1]-half_h
		x1, y1 = max(0, min(W-1, x1)), max(0, min(H-1, y1))	### for insurance
		x2, y2 = curr_sample[0]+half_w, curr_sample[1]+half_h
		x2, y2 = max(0, min(W-1, x2)), max(0, min(H-1, y2)) ### for insurance
		box_sample = np.array([x1, y1, x2, y2])
		
		if iou_thresh_ignored:	# this is exclusive for sampling candidates during online tracking
			samples[idx, :] = box_sample
			idx += 1
			continue

		overlap_ratio = IoU(convert_box2bbx(box_sample), gt)
		if overlap_ratio >= pos_thresh and pos_sampling: #if positive sampling is being performed and its overlapping ratio >= 0.7
			samples[idx, :] = box_sample
			idx += 1
		elif overlap_ratio < neg_thresh and not pos_sampling: #if negative sampling is being performed and its overlapping ratio < 0.3
			samples[idx, :] = box_sample
			idx += 1

	return samples


def gen_samples(sampling_type, 
				gt, 
				num_samples, 
				raw_img_size, 
				base_scalar=1.05, 
				trans_fac=0.1, 
				scale_fac=5, 
				pos_sampling=True, 
				pos_thresh=0.7, 
				neg_thresh=0.3, 
				iou_thresh_ignored=False):
	
	H = raw_img_size[0]
	W = raw_img_size[1]

	#sample = (cx, cy, w, h), where (cx, cy) is the coodinate of the gt image
	sample = np.array([gt[0]+gt[2]/2, gt[1]+gt[3]/2, gt[2], gt[3]], dtype = np.float32)
	samples = np.tile(sample, (num_samples, 1))	
	
	idx = 0
	while idx < num_samples:
		curr_sample = samples[idx, :].copy()

		if sampling_type == 'gaussian':
			lt_increment = trans_fac * round(np.mean(gt[2:4])) * np.array([post_proc(np.random.randn(1,)), post_proc(np.random.randn(1,))])
			curr_sample[:2] = curr_sample[:2] + lt_increment.reshape(2,)

			randn_vec = np.array([post_proc(np.random.randn(1,)), post_proc(np.random.randn(1,))])
			wh_factor = base_scalar ** (scale_fac * randn_vec)
			curr_sample[2:] = curr_sample[2:] * wh_factor.reshape(2,)

		elif sampling_type == 'uniform':	#uniform distribution within a searching area 2.5 times the size of bbx
			sr_ratio = 3.5 #cfg.TEST.UNIFORM_SAMPLING_RANGE_RATIO	#twice or 2.5 times???

			randn_vec = np.array([post_proc(np.random.randn(1,)), post_proc(np.random.randn(1,))])
			wh_factor = base_scalar ** (scale_fac * randn_vec)
			curr_sample[2:] = curr_sample[2:] * wh_factor.reshape(2,)

			cx_bound = (curr_sample[0]-curr_sample[2]*(sr_ratio/2), curr_sample[0]+curr_sample[2]*(sr_ratio/2))
			cy_bound = (curr_sample[1]-curr_sample[3]*(sr_ratio/2), curr_sample[1]+curr_sample[3]*(sr_ratio/2))
			cx = (cx_bound[1] - cx_bound[0]) * np.random.random_sample() + cx_bound[0]
			cy = (cy_bound[1] - cy_bound[0]) * np.random.random_sample() + cy_bound[0]

			curr_sample[0] = cx
			curr_sample[1] = cy

		elif sampling_type == 'whole':	#uniform distribution within the whole image
			randn_vec = np.array([post_proc(np.random.randn(1,)), post_proc(np.random.randn(1,))])
			wh_factor = base_scalar ** (scale_fac * randn_vec)
			curr_sample[2:] = curr_sample[2:] * wh_factor.reshape(2,)

			w = curr_sample[2]
			h = curr_sample[3]
			curr_sample[0] = (W - w) * np.random.random_sample() + w / 2.0
			curr_sample[1] = (H - h) * np.random.random_sample() + h / 2.0

		'''In case that samples experience abrupt scaling variation...''' ##########
		curr_sample[2] = max(5, min(W-5, curr_sample[2])) 	#w max(gt[2]/5.0, min(gt[2]*5.0, curr_sample[2]))
		curr_sample[3] = max(5, min(H-5, curr_sample[3]))	#h max(gt[3]/5.0, min(gt[2]*5.0, curr_sample[3]))

		half_w, half_h = curr_sample[2]/2.0, curr_sample[3]/2.0

		# bbx_sample = np.array([curr_sample[0]-curr_sample[2]/2, curr_sample[1]-curr_sample[3]/2, curr_sample[2], curr_sample[3]])
		# bbx_sample[0] = max(0, min(W-bbx_sample[2]-1, bbx_sample[0]))
		# bbx_sample[1] = max(0, min(H-bbx_sample[3]-1, bbx_sample[1]))

		"""The centre coordinate of candidate box should lie within the [half_w, W-half_w-1]x[half_h, H-half_h-1]"""
		curr_sample[0] = max(half_w, min(W-half_w-1, curr_sample[0]))
		curr_sample[1] = max(half_h, min(H-half_h-1, curr_sample[1]))

		x1, y1 = curr_sample[0]-half_w, curr_sample[1]-half_h
		x1, y1 = max(0, min(W-1, x1)), max(0, min(H-1, y1))	### for insurance
		x2, y2 = curr_sample[0]+half_w, curr_sample[1]+half_h
		x2, y2 = max(0, min(W-1, x2)), max(0, min(H-1, y2)) ### for insurance
		bbx_sample = np.array([x1, y1, x2-x1+1, y2-y1+1])
		
		if iou_thresh_ignored:	# this is exclusive for sampling candidates during online tracking
			samples[idx, :] = bbx_sample
			idx += 1
			continue

		overlap_ratio = IoU(bbx_sample, gt)
		if overlap_ratio >= pos_thresh and pos_sampling: #if positive sampling is being performed and its overlapping ratio >= 0.7
			samples[idx, :] = bbx_sample
			idx += 1
		elif overlap_ratio < neg_thresh and not pos_sampling: #if negative sampling is being performed and its overlapping ratio < 0.3
			samples[idx, :] = bbx_sample
			idx += 1

	return samples


def gen_negative_samples_polar_radius(num_samples, gt, raw_img_size):
	"""This function will generate num_samples negative samples using polar-radius based method"""
	frame_height = raw_img_size[0]
	frame_width = raw_img_size[1]

	theta_list = np.linspace(0, 2 * np.pi, 60)
	
	l_x, t_y, w, h = gt[0], gt[1], gt[2], gt[3]

	r_start = 0.2 * np.sqrt(w ** 2 + h ** 2)
	r_end = 0.5 * np.sqrt(w ** 2 + h ** 2)
	r_list = np.linspace(r_start, r_end, 10)

	c_x, c_y = l_x+w/2, t_y+h/2

	sample_cnt = 0
	sample_list = np.zeros((0, 4), dtype=np.float32)

	iter_cnt = 0
	while sample_cnt < num_samples:
		iter_cnt += 1
		if iter_cnt > 3:	break
		
		for theta in theta_list:
			if sample_cnt >= num_samples:	break

			angle_eps = np.pi/9
			if np.abs(theta) <= angle_eps \
				or np.abs(theta-np.pi/2) <= angle_eps \
				or np.abs(theta-np.pi) <= angle_eps \
				or np.abs(theta-1.5*np.pi) <= angle_eps \
				or np.abs(theta-2*np.pi) <= angle_eps:	continue

			for r in r_list:
				if sample_cnt >= num_samples:	break
				
				c_x__, c_y__ = c_x + r * np.cos(theta), c_y - r * np.sin(theta)
				if theta >= 0 and theta < np.pi/2: #theta in Region I
					h__ = 2.0 * (c_y - c_y__  + h / 2.0)
					w__ = 2.0 * (c_x__ - c_x + w / 2.0)

				elif theta >= np.pi/2 and theta < np.pi: #theta in Region II
					h__ = 2.0 * (c_y - c_y__ + h / 2.0)
					w__ = 2.0 * (c_x - c_x__ + w / 2.0)

				elif theta >= np.pi and theta < 1.5 * np.pi: #theta in Region III
					h__ = 2.0 * (c_y__ - c_y + h / 2.0)
					w__ = 2.0 * (c_x - c_x__ + w / 2.0)

				else: #theta in Region IV
					h__ = 2.0 * (c_y__ - c_y + h / 2.0)
					w__ = 2.0 * (c_x__ - c_x + w / 2.0)

				l_x__ = c_x__ - w__ / 2.0
				t_y__ = c_y__ - h__ / 2.0

				r_x__ = l_x__ + w__ - 1
				b_y__ = t_y__ + h__ - 1

				l_x__ = max(0, l_x__)
				t_y__ = max(0, t_y__)
				r_x__ = min(r_x__, frame_width - 1)
				b_y__ = min(b_y__, frame_height - 1)

				w__ = r_x__ - l_x__ + 1
				h__ = b_y__ - t_y__ + 1

				bbx_sample = np.array([l_x__, t_y__, w__, h__])
				overlap_ratio = IoU(bbx_sample, gt)
				#print 'overlap_ratio: {}'.format(overlap_ratio)

				if overlap_ratio <= 0.6:
					sample_list = np.vstack((sample_list, bbx_sample.reshape(1, 4)))
					sample_cnt += 1
	
	return sample_list


def display(frame, saving_path, fname):
	saving_dir = saving_root + '/' + saving_path
	if not os.path.exists(saving_dir):
		os.makedirs(saving_dir)
	plt.imsave(saving_dir + '/' + fname, frame)

	#image_file = cbook.get_sample_data(saving_path + '/' + fname)
	#image = plt.imread(image_file)
	#plt.imshow(image)
	#plt.show()


def vis_neg_finetuning_data_pool(seq_name, raw_img, idx_frame, neg_samples_gaussian, neg_samples_uniform, neg_samples_whole, neg_samples_polar_radius):
	root_dir = cfg.CODE_ROOT_DIR + '/output/finetuning_data/{}/'.format(cfg.TEST.BENCHMARK_NAME)

	tar_dir = root_dir + seq_name + '/' + str(idx_frame)
	if not os.path.exists(tar_dir):
		os.makedirs(tar_dir)

	for idx in xrange(neg_samples_gaussian.shape[0]):
		bbx_sample = neg_samples_gaussian[idx, :]
		box = (int(round(bbx_sample[0])), int(round(bbx_sample[1])), int(round(bbx_sample[0]+bbx_sample[2]-1)), int(round(bbx_sample[1]+bbx_sample[3]-1)))

		cv2.rectangle(raw_img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0))

	for idx in xrange(neg_samples_uniform.shape[0]):
		bbx_sample = neg_samples_uniform[idx, :]
		box = (int(round(bbx_sample[0])), int(round(bbx_sample[1])), int(round(bbx_sample[0]+bbx_sample[2]-1)), int(round(bbx_sample[1]+bbx_sample[3]-1)))

		cv2.rectangle(raw_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0))

	for idx in xrange(neg_samples_whole.shape[0]):
		bbx_sample = neg_samples_whole[idx, :]
		box = (int(round(bbx_sample[0])), int(round(bbx_sample[1])), int(round(bbx_sample[0]+bbx_sample[2]-1)), int(round(bbx_sample[1]+bbx_sample[3]-1)))

		cv2.rectangle(raw_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255))

	for idx in xrange(neg_samples_polar_radius.shape[0]):
		bbx_sample = neg_samples_polar_radius[idx, :]
		box = (int(round(bbx_sample[0])), int(round(bbx_sample[1])), int(round(bbx_sample[0]+bbx_sample[2]-1)), int(round(bbx_sample[1]+bbx_sample[3]-1)))

		cv2.rectangle(raw_img, (box[0], box[1]), (box[2], box[3]), (255, 255, 255))	
	
	now = datetime.datetime.now()
	jpg_name = now.strftime('%Y-%m-%d_%H:%M:%S') + '.jpg'
	cv2.imwrite(tar_dir + '/{}'.format(jpg_name), raw_img)


def unif_save_visualization(frame_dup, path_seq, index_new_frame, pred_bbx_score, cand_dict_list, index_order):
	seq_name = os.path.split(path_seq)[1]
	saving_path = seq_name + '/' + str(index_new_frame)
	saving_root = cfg.CODE_ROOT_DIR + '/output/experimental/test_phase/{}'.format(cfg.TEST.BENCHMARK_NAME)
	saving_dir = saving_root + '/' + saving_path
	fname = 'cands.jpg'

	if not os.path.exists(saving_dir):
		os.makedirs(saving_dir)
	cv2.imwrite(os.path.join(saving_dir, fname), frame_dup)

	if pred_bbx_score >= 0.90:
		corr_fobj = open(os.path.join(saving_dir, 'corr.txt'), 'w')
		for index in index_order[:20]:
			distance = cand_dict_list[index, -1]
			prob = cand_dict_list[index, -2]
			entry = '{} {} {}'.format(index, prob, distance)
			corr_fobj.write(entry + '\n')
		corr_fobj.close()


def unif_vis_cands_conf_weight(i, index, bbxes_Pk, frame_dup, path_seq, index_new_frame, cand_dict_list, i_dist_prob, i_factor):
	bbx_sample = bbxes_Pk[index, :]
	box = (int(round(bbx_sample[0])), int(round(bbx_sample[1])), int(round(bbx_sample[0]+bbx_sample[2]-1)), int(round(bbx_sample[1]+bbx_sample[3]-1)))
	cv2.rectangle(frame_dup, box[:2], box[2:], (0, 0, 255))
	cv2.putText(frame_dup, '{}'.format(index), box[:2], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def gen_config_report():
	tar_fpath = cfg.ROOT_DIR + '/output/tracking_res/{}/{}'.format(cfg.TEST.BENCHMARK_NAME, cfg.TEST.BENCHMARK_NAME) + '_config_rep.txt'
	tar_fobj = open(tar_fpath, 'w')

	for key in cfg.keys():
		if key == 'TEST' or key == 'TRAIN':
			continue
		
		info = '__C.' + key + ': ' + cfg[key]
		tar_fobj.write(info + '\n')

	for key in cfg.TEST.keys():
		info = '__C.TEST.{}: {}'.format(key, cfg.TEST[key])
		tar_fobj.write(info + '\n')

	tar_fobj.close()


def compute_Gaussian2D_prob(x, mu, cov):
	det_cov = cov[0, 0] * cov[1, 1]
	normalizer = 2 * np.pi * (det_cov ** 0.5)

	delta = x - mu
	mahalanoibis_dis = -0.5 * (delta[0] ** 2 / cov[0, 0] + delta[1] ** 2 / cov[1, 1])

	return (1.0 / normalizer) * np.exp(mahalanoibis_dis)


def compute_Laplacian2D_prob(x, mu, b):
	euclidean_dis = np.dot((x - mu), (x - mu)) ** (0.5)
	return 1.0 / (2.0 * b) * np.exp(-1.0 * euclidean_dis / b)


def determine_displacement(bbx1, bbx2):
	cx1 = bbx1[0] + bbx1[2] / 2.0
	cy1 = bbx1[1] + bbx1[3] / 2.0
	cx2 = bbx2[0] + bbx2[2] / 2.0
	cy2 = bbx2[1] + bbx2[3] / 2.0
	#return np.abs(cx1 - cx2), np.abs(cy1 - cy2)
	return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def func_iou(bb, gtbb):
	iou = 0
	iw = min(bb[2],gtbb[2]) - max(bb[0],gtbb[0]) + 1
	ih = min(bb[3],gtbb[3]) - max(bb[1],gtbb[1]) + 1

	if iw>0 and ih>0:
		ua = (bb[2]-bb[0]+1)*(bb[3]-bb[1]+1) + (gtbb[2]-gtbb[0]+1)*(gtbb[3]-gtbb[1]+1) - iw*ih
		iou = iw*ih/ua;

	return iou


def sample_regions_precompute(rad, nr_ang, stepsize, scales=[0.7071, 1, 1.4142]):
	nr_step = int(rad / stepsize)
	cos_values = np.cos(np.arange(0,2*np.pi,2*np.pi/nr_ang))
	sin_values = np.sin(np.arange(0,2*np.pi,2*np.pi/nr_ang))

	dxdys = np.zeros((2,nr_step*nr_ang+1))
	count = 0
	for ir in range(1,nr_step+1):
		offset = stepsize * ir
		for ia in range(1,nr_ang+1):

			dx = offset * cos_values[ia-1]
			dy = offset * sin_values[ia-1]
			count += 1
			dxdys[0, count-1] = dx
			dxdys[1, count-1] = dy
	
	samples = np.zeros((4,(nr_ang*nr_step+1)*len(scales)))
	count = 0
	jump = nr_step*nr_ang+1
	for s in scales:
		samples[0:2, count*jump:(count+1)*jump] = dxdys
		samples[2, count*jump:(count+1)*jump] = s;
		samples[3, count*jump:(count+1)*jump] = s;
		count = count + 1

	return samples # dx dy 1*s 1*s


def sample_regions(x, y, w, h, im_w, im_h, samples_template):
	samples = samples_template.copy()
	samples[0,:] += x
	samples[1,:] += y
	samples[2,:] *= w
	samples[3,:] *= h

	samples[2,:] = samples[0,:] + samples[2,:] - 1
	samples[3,:] = samples[1,:] + samples[3,:] - 1
	samples = np.round(samples)

	flags = np.logical_and(np.logical_and(np.logical_and(samples[0,:]>0, samples[1,:]>0), samples[2,:]<im_w), samples[3,:]<im_h)
	samples = samples[:,flags]

	return samples # x1 y1 x2 y2
