# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

from util_test import convert_box2bbx
from utils import get_subwindow_tracking
import py_nms
from train import generate_all_anchors

'''DEPRECATED.'''
def generate_anchor(total_stride, scales, ratios, score_size):
    anchor_num = len(ratios) * len(scales)  # 5 x 1
    anchor = np.zeros((anchor_num, 4),  dtype=np.float32)
    size = total_stride * total_stride
    count = 0
    for ratio in ratios:
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)
        for scale in scales:
            wws = ws * scale
            hhs = hs * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = - (score_size / 2) * total_stride
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor


class TrackerConfig(object):
    # These are the default hyper-params for DaSiamRPN 0.3827
    
    windowing = 'cosine'  # to penalize large displacements [cosine/uniform]
    
    # Params from the network architecture, have to be consistent with the training
    exemplar_size = 127  # input z size
    instance_size = 255  # input x size (search region), 271
    total_stride = 8
    score_size = (instance_size-exemplar_size)/total_stride+1 # 'Cuz examplar will be used as a kernel to convolve with instance
    delta_score_size = 17 # must be consistent with that of Siamese network, to be automatically linked

    context_amount = 0.5  # context amount for the exemplar
    ratios = [0.33, 0.5, 1, 2, 3]
    scales = [8, ]
    basic_anchor_num = len(ratios) * len(scales)
    anchors = []
    penalty_k = 0.055
    window_influence = 0.42
    lr = 0.295

    alpha_i = 1.0
    eta = 0.01
    alpha_hat = 0.5
    num_pts_half_bin = 2
    distractor_thresh = 0.5


def tracker_eval(net, x_crop, target_pos, target_sz, window, scale_z, p):
    delta, score = net(x_crop) # (1, 4K, 17, 17) and (1, 2K, 17, 17)

    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1, :].cpu().numpy()

    delta[0, :] = delta[0, :] * p.anchors[:, 2] + p.anchors[:, 0]
    delta[1, :] = delta[1, :] * p.anchors[:, 3] + p.anchors[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * p.anchors[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * p.anchors[:, 3]
    
    '''
    def change(r):
        return np.maximum(r, 1./r)

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)
    '''

    # size penalty
    s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz)))  # scale penalty
    r_c = change((target_sz[0] / target_sz[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty

    penalty = np.exp(-(r_c * s_c - 1.) * p.penalty_k)
    pscore = penalty * score

    # window float
    pscore = pscore * (1 - p.window_influence) + window * p.window_influence
    best_pscore_id = np.argmax(pscore)

    target = delta[:, best_pscore_id] / scale_z
    target_sz = target_sz / scale_z
    lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr

    res_x = target[0] + target_pos[0]
    res_y = target[1] + target_pos[1]

    res_w = target_sz[0] * (1 - lr) + target[2] * lr
    res_h = target_sz[1] * (1 - lr) + target[3] * lr

    target_pos = np.array([res_x, res_y])
    target_sz = np.array([res_w, res_h])
    return target_pos, target_sz, score[best_pscore_id]


def tracker_eval_distractor_aware(x_crop, target_sz, scale_z, state):
    p = state['p']  # tracking config
    net = state['net']
    window = state['window'] # cosine window
    target_pos = state['target_pos']

    delta, score, sr_feat = net(x_crop) # of shape (1, 4K, 17, 17), (1, 2K, 17, 17), (1, 22, 22, 512)

    delta = delta.contiguous().view(4, -1).data.cpu().numpy() # (4, K*17*17)
    score = F.softmax(score.contiguous().view(2, -1), dim=0).data[1, :].cpu().numpy() # (2, K*17*17)

    delta[0, :] = delta[0, :] * p.anchors[:, 2] + p.anchors[:, 0] # x
    delta[1, :] = delta[1, :] * p.anchors[:, 3] + p.anchors[:, 1] # y
    delta[2, :] = np.exp(delta[2, :]) * p.anchors[:, 2] # w
    delta[3, :] = np.exp(delta[3, :]) * p.anchors[:, 3] # h

    inds_inside = np.where(
        (delta[0, :] >= 0) &
        (delta[1, :] >= 0) &
        (delta[0, :] + delta[2, :] - 1 < p.instance_size) &
        (delta[1, :] + delta[3, :] - 1 < p.instance_size) 
    )[0]
    delta = delta[:, inds_inside]
    score = score[inds_inside]

    # for i in range(delta.shape[1]):
    #    print(delta[:, i])

    '''NMS is performed on delta according to pscore's'''
    dets = np.hstack(
        (delta.transpose(), score[np.newaxis, :].transpose())
        ) # in bbx format of (x, y, w, h)
    dets[:, 2] = dets[:, 0] + dets[:, 2] - 1
    dets[:, 3] = dets[:, 1] + dets[:, 3] - 1

    nms_indices_kept = py_nms.py_nms(dets, thresh=0.9) # now dets is in box format
    # dets_kept = dets[nums_ind_kept] # (N, 4+1)
    # print(dets.astype(int))

    def bilinear_interp(sr_feat, x_f, y_f):
        ub = sr_feat.shape[-1]-1
        x1, y1 = max(0, min(ub, int(x_f))), max(0, min(ub, int(y_f)))
        x2, y2 = max(0, min(ub, int(x_f)+1)), max(0, min(ub, int(y_f)+1))
        #print(f"{x1}, {y1}, {x2}, {y2}")

        fQ11, fQ12, fQ21, fQ22 = sr_feat[:, x1, y1], sr_feat[:, x1, y2], sr_feat[:, x2, y1], sr_feat[:, x2, y2]
        fQ11 = fQ11.cpu().detach().numpy()
        fQ12 = fQ12.cpu().detach().numpy()
        fQ21 = fQ21.cpu().detach().numpy()
        fQ22 = fQ22.cpu().detach().numpy()

        ret1 = (y2-y_f)/(y2-y1)*((x2-x_f)/(x2-x1)*fQ11 + (x_f-x1)/(x2-x1)*fQ21)
        ret2 = (y_f-y1)/(y2-y1)*((x2-x_f)/(x2-x1)*fQ12 + (x_f-x1)/(x2-x1)*fQ22)
        
        return ret1+ret2

    def binwise_max_pooling(meshgrid, num_bins, num_pts):
        assert meshgrid.shape[0] == meshgrid.shape[1] == num_bins*num_pts

        num_channels = meshgrid.shape[2]-2
        pooling_res = np.zeros((num_bins, num_bins, num_channels), dtype=np.float32)
        for channel in range(num_channels):
            for r in range(num_bins):
                for c in range(num_bins):
                    res_rc = meshgrid[r*num_pts, c*num_pts, 2+channel]
                    res_rc = max(res_rc, meshgrid[r*num_pts, c*num_pts+1, 2+channel])
                    res_rc = max(res_rc, meshgrid[r*num_pts+1, c*num_pts, 2+channel])
                    res_rc = max(res_rc, meshgrid[r*num_pts+1, c*num_pts+1, 2+channel])
                    pooling_res[r, c, channel] = res_rc

        return pooling_res

    '''Extract phi's of each region proposal using ROI-align'''
    W = H = p.instance_size # raw image size
    W_ = H_ = sr_feat.shape[2] # size of feature map of search region, expected to be 22
    num_props = len(nms_indices_kept)
    num_bins = state['template_feat'].shape[-1] # expect state['template_feat'] to be 6
    num_pts = p.num_pts_half_bin
    num_channels = state['template_feat'].shape[1] # expected to be 512
    roi_align_feats = np.empty((num_props, num_bins, num_bins, num_channels), dtype=np.float32)
    index2featkept_map = {} # a mapping from the original index to the new index
    for i in range(num_props):
        nms_index_kept = nms_indices_kept[i]

        x, y, w, h = convert_box2bbx(tuple(dets[nms_index_kept][:4]))
        x_, y_ = W_*(x+1)/W-1, H_*(y+1)/H-1
        w_, h_ = W_*(x+w)/W-x_, H_*(y+h)/H-y_ #W_*w/W, H_*h/H

        meshgrid = np.empty((num_bins*num_pts, num_bins*num_pts, 2+num_channels)) # `2+num_channels` means (x, y, val)
        h_stride = w_/num_bins/(num_pts+1)
        v_stride = h_/num_bins/(num_pts+1)

        for r in range(num_bins*num_pts):
            for c in range(num_bins*num_pts):
                h_delta = (c//num_pts)*((num_pts+1)*h_stride) + ((c%num_pts)+1)*h_stride
                v_delta = (r//num_pts)*((num_pts+1)*v_stride) + ((r%num_pts)+1)*v_stride

                meshgrid[r, c, :2] = np.array([x_+h_delta, y_+v_delta]) # can be disabled

                x_f, y_f = x_+h_delta, y_+v_delta
                # print(x_f, y_f)
                vals = bilinear_interp(sr_feat[0], x_f, y_f) # sr_feat (1, 512, 22, 22)
                meshgrid[r, c, 2:] = vals

        roi_align_res = binwise_max_pooling(meshgrid, num_bins, num_pts) # resulting in a tensor of shape (6, 6, 512)
        roi_align_feats[i, ...] = roi_align_res
        index2featkept_map[nms_index_kept] = i
    '''After RoI-align, we obtain roi_align_feats, which is a tensor of shape (N, 6, 6, 512)'''

    '''Distractor-aware incremental learning:'''
    # 1. Construct a distractor set, saving indices of the original set of proposals before NMS
    distractor_index_set = []
    running_idx = nms_indices_kept[0]
    running_max = np.sum(state['template_feat'][0].transpose(1, 2, 0) * roi_align_feats[0]) # element-wise multiplication and sum
    if running_max > p.distractor_thresh:
        distractor_index_set.append(running_idx)
    for i in range(1, num_props):
        nms_index_kept = nms_indices_kept[i]
        curr_val = np.sum(state['template_feat'][0].transpose(1, 2, 0) * roi_align_feats[i]) # element-wise multiplication and sum
        if curr_val > running_max:
            running_idx = nms_index_kept
            running_max = curr_val
        if curr_val > p.distractor_thresh:
            distractor_index_set.append(nms_index_kept)
    distractor_index_set.remove(running_idx)

    # 2. Incremental learning according Eqn. (4)
    sum_alpha_i = len(distractor_index_set) * p.alpha_i
    running_template = state['acc_beta_phi'] / state['acc_beta'] - state['acc_beta_alpha_phi'] / (state['acc_beta'] * sum_alpha_i)
    running_idx = nms_indices_kept[0]
    running_max = np.sum(running_template[0].transpose(1, 2, 0) * roi_align_feats[0])
    for i in range(1, num_props):
        nms_index_kept = nms_indices_kept[i]
        curr_val = np.sum(running_template[0].transpose(1, 2, 0) * roi_align_feats[index2featkept_map[nms_index_kept]])
        if curr_val > running_max:
            running_idx = nms_index_kept
            running_max = curr_val

    beta_t = p.eta/(1-p.eta)
    curr_beta_alpha_phi = np.zeros_like(state['acc_beta_alpha_phi'])
    for distractor_index in distractor_index_set:
        curr_beta_alpha_phi += p.alpha_i * roi_align_feats[index2featkept_map[distractor_index]].transpose(2, 0, 1)[np.newaxis, ...]
    curr_beta_alpha_phi *= p.alpha_hat * beta_t
    state['acc_beta_alpha_phi'] += curr_beta_alpha_phi
    state['acc_beta'] += beta_t
    '''---Distractor-aware incremental learning---'''

    best_pscore_id = running_idx
    target = delta[:, best_pscore_id] / scale_z
    target_sz = target_sz / scale_z
    lr = 0.1 #penalty[best_pscore_id] * score[best_pscore_id] * p.lr

    res_x = target[0] + target_pos[0]
    res_y = target[1] + target_pos[1]

    res_w = target_sz[0] * (1 - lr) + target[2] * lr
    res_h = target_sz[1] * (1 - lr) + target[3] * lr

    target_pos = np.array([res_x, res_y])
    target_sz = np.array([res_w, res_h])
    return target_pos, target_sz, score[best_pscore_id]


def SiamRPN_init(im, target_pos, target_sz, net):
    ## target_pos is (cx, cy)
    ## target_sz is (w, h)

    state = dict()
    p = TrackerConfig()
    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]

    if ((target_sz[0]*target_sz[1]) / float(state['im_h']*state['im_w'])) < 0.004:
        p.instance_size = 287  # small object big search region
    else:
        p.instance_size = 255 #271

    p.delta_score_size = int((p.instance_size-p.exemplar_size)/p.total_stride+1) # size of the last feature map, expected to be 17

    # all anchors of each aspect ratio and scale at each location are generated.
    p.anchors, _ = generate_all_anchors((p.delta_score_size, p.delta_score_size), 
                                     (p.instance_size, p.instance_size))
    # of shape (dropping from 2420 down to 433, 4)
    
    avg_chans = np.mean(im, axis=(0, 1)) #???????????

    wc_z = target_sz[0] + p.context_amount * sum(target_sz) # adding some context info
    hc_z = target_sz[1] + p.context_amount * sum(target_sz) # adding some context info
    s_z = round(np.sqrt(wc_z * hc_z))
    
    # initialize the exemplar
    z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)

    z = Variable(z_crop.unsqueeze(0))
    template_feat = net.template(z.cuda())

    if p.windowing == 'cosine':
        # return the outer product of two hanning vectors, which is a matrix of the same size as the feature map of search region
        window = np.outer(np.hanning(p.delta_score_size), np.hanning(p.delta_score_size)) ############### p.score_size???
    elif p.windowing == 'uniform':
        window = np.ones((p.delta_score_size, p.delta_score_size)) ################## p.score_size???

    # flatten and replicate the cosine window    
    window = np.tile(window.flatten(), p.basic_anchor_num)

    state['p'] = p
    state['net'] = net
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = 1.0

    # for distractor-aware incremental learning
    template_feat_cpu = template_feat.cpu().detach().numpy()
    state['template_feat'] = template_feat_cpu
    state['acc_beta_phi'] = template_feat_cpu
    state['acc_beta'] = 1.0
    state['acc_beta_alpha_phi'] = np.zeros_like(template_feat_cpu)

    return state


def SiamRPN_track(state, im):
    p = state['p']  # tracking config
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window'] # cosine window
    target_pos = state['target_pos'] # cx, cy of target in the previous frame
    target_sz = state['target_sz']   # w, h of target in the previous frame
    template_feat = state['template_feat']

    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.exemplar_size / s_z

    ###'Local to Global': if failure mode is activated then expand d_search; otherwise set d_search to normal
    d_search = (p.instance_size - p.exemplar_size) / 2
    if state['score'] < 0.3:
        d_search *= 2
        
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    # extract scaled crops for search region x at previous target position
    x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))
    # where the third argument is the model size and the fourth is the orginal size in the raw image.

    target_pos, target_sz, score = tracker_eval_distractor_aware(x_crop.cuda(), target_sz*scale_z, scale_z, state)
    
    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    return state
