import numpy as np
import os
import torch

from net import SiamRPNBIG
from gen_all_anchors import generate_all_anchors
from bbox_transform import bbox_transform
from config import cfg

import argparse
import dataset
from tqdm import tqdm


def bbox_overlaps(box, gt, phase='iou'):
    """
    Compute the overlaps between box and gt(_box)
    box: (N, 4) NDArray
    gt : (K, 4) NDArray
    return: (N, K) NDArray, stores Max(0, intersection/union) or Max(0, intersection/area_box)
    """
    # Note that the inputs are in box format: x1, y1, x2, y2
    
    N = box.shape[0]
    K = gt.shape[0]
    target_shape = (N, K, 4)
    b_box = np.broadcast_to(np.expand_dims(box, axis=1), target_shape)
    b_gt = np.broadcast_to(np.expand_dims(gt, axis=0), target_shape)

    iw = (np.minimum(b_box[:, :, 2], b_gt[:, :, 2]) -
          np.maximum(b_box[:, :, 0], b_gt[:, :, 0]))
    ih = (np.minimum(b_box[:, :, 3], b_gt[:, :, 3]) -
          np.maximum(b_box[:, :, 1], b_gt[:, :, 1]))
    inter = np.maximum(iw, 0) * np.maximum(ih, 0)

    # Use the broadcast to save some time
    area_box = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    area_gt = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    area_target_shape = (N, K)
    b_area_box = np.broadcast_to(np.expand_dims(area_box, axis=1), area_target_shape)
    b_area_gt = np.broadcast_to(np.expand_dims(area_gt, axis=0), area_target_shape)

    assert phase == 'iou' or phase == 'ioa'
    union = b_area_box + b_area_gt - inter if phase == 'iou' else b_area_box

    overlaps = np.maximum(inter / np.maximum(union, 1), 0)
    return overlaps
    

def smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, beta=1.0):
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smooth_l1_sign = (abs_in_box_diff < beta).detach().float()

    in_loss_box = smooth_l1_sign * 0.5 * torch.pow(in_box_diff, 2) / beta + \
                    (1-smooth_l1_sign) * (abs_in_box_diff-0.5*beta)

    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    N = loss_box.size(0)
    loss_box = loss_box.view(-1).sum(0) / N
    return loss_box


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""
    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4 #5

    return bbox_transform(ex_rois, gt_rois[:, :4].numpy()).astype(np.float32, copy=False)


def gen_anchor_target(cls_output_shape, xs_shape, gt_boxes):
    """
    Assign anchors to ground-truth targets. 
    Produces anchor classification labels and bounding-box regression targets.
    """
    height, width = cls_output_shape
    all_anchors, A = generate_all_anchors(cls_output_shape, xs_shape)
    # Note that anchors are in format (x1, y1, x2, y2)
    
    total_anchors = all_anchors.shape[0] 
    inds_inside = np.where(
        (all_anchors[:, 0] >= 0) &
        (all_anchors[:, 1] >= 0) &
        (all_anchors[:, 2] < xs_shape[1]) &
        (all_anchors[:, 3] < xs_shape[0])
    )[0]
    anchors = all_anchors[inds_inside, :]

    labels = np.zeros((1, 1, A*height, width))
    bbox_targets = np.zeros((1, 4*A, height, width))
    bbox_inside_weights = np.zeros((1, 4*A, height, width))
    bbox_outsied_weights = np.zeros((1, 4*A, height, width))

    # label: 1 is positive, 0 is negative, -1 is don't care
    labels = np.empty((len(inds_inside), ), dtype=np.float32)
    labels.fill(-1)

    # overlaps between anchors and gt boxes
    # overlaps.shape = (#total_anchors, #gts)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))

    argmax_overlaps = overlaps.argmax(axis=1) # of shape (#total_anchors, )
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps] # of shape (#total_anchors, )

    gt_argmax_overlaps = overlaps.argmax(axis=0) # of shape (#gt, )
    gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])] # of shape (#gt, )
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0 # 0.3
    labels[gt_argmax_overlaps] = 1
    labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1 # 0.7

    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    # _compute_targets() returns #sifted_anchors-by-4 tensor with each row being (dx, dy, dw, dy), 
    #     the increment to be learnt by bbx regressor
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])
    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS) #RPN_BBOX_INSIDE_WEIGHTS = [1.0, 1.0, 1.0, 1.0]

    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0: #cfg.TRAIN.RPN_POSITIVE_WEIGHT == -1.0
        # uniform weighting of examples (given non-uniform sampling)
        num_examples = np.sum(labels >= 0)  # num_examples is the sum of anchors labeled 1
        positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples
    else:
        assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
        positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT / np.sum(labels == 1))
        negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) / np.sum(labels == 0))
    
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights

     # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)    #labels.shape == (#total_anchors, )
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0) #bbox_targets.shape == (#total_anchors, 4)
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0) #bbox_inside_weights.shape == (#total_anchors, 4)
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0) #bbox_outside_weights.shape == (#total_anchors, 4)

    # labels
    labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2) # of shape (1, A, height, width)
    labels = labels.reshape((1, 1, A*height, width))
    # bbox_targets
    bbox_targets = bbox_targets.reshape((1, height, width, A*4)).transpose(0, 3, 1, 2) # of shape (1, 4*A, height, width)
    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights.reshape((1, height, width, A*4)).transpose(0, 3, 1, 2)  # of shape (1, 4*A, height, width)
    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights.reshape((1, height, width, A*4)).transpose(0, 3, 1, 2) # of shape (1, 4*A, height, width)

    return labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, A


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', default=0, help='GPU ID to use, e.g. \'0\'', type=int)

    return parser.parse_args()


def load_pretrained_weights(net, weight_file_path):
    ori_pretrained_dict = torch.load(weight_file_path)
    model_dict = net.state_dict()
    #pretrained_dict = {k: v for k, v in ori_pretrained_dict.items() if k in model_dict}
    
    import collections
    pretrained_dict = collections.OrderedDict()

    for k, v in ori_pretrained_dict.items():
        if k in model_dict and k.startswith('featureExtract'): # Only load the modified AlexNet weights
            pretrained_dict[k] = v
            # print(k)

    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)


if __name__ == '__main__':
    args = parse_args()
    gpu_id = args.gpu_id
    if gpu_id is None:
        DEVICE = torch.device(f'cpu')
    else:
        DEVICE = torch.device(f'cuda:{gpu_id}')

    z_size = (127, 127)
    x_size = (255, 255)
    batch_size = num_domains = 50
    num_epoches = 100

    loader = dataset.load_data(batch_size, z_size, x_size)['train']

    net = SiamRPNBIG()
    net.train().to(DEVICE)
    # load_pretrained_weights(net, "./SiamRPNBIG.model")
    optimizer = torch.optim.Adam(net.parameters(), weight_decay=0.001, lr=0.001)

    for i_ep in range(num_epoches):
        for i_iter, sample in tqdm(enumerate(loader), total=len(loader)):
            zs = sample['template'].to(DEVICE)
            xs = sample['search_region'].to(DEVICE)
            gt_boxes = sample['gt_box'] #.to(DEVICE)

            optimizer.zero_grad()

            net.template(zs)
            reg_output, cls_output, _ = net.forward(xs) # of shape (50, 4*5, 17, 17), (50, 2*5, 17, 17)

            feat_h, feat_w = tuple(cls_output.size()[-2:])
            assert zs.shape[0] == xs.shape[0] == gt_boxes.shape[0]
            total_loss = total_cls_loss = total_reg_loss = 0.0
            for i in range(zs.shape[0]):
                rpn_labels, \
                rpn_bbox_targets, \
                rpn_bbox_inside_weights, \
                rpn_bbox_outside_weights, \
                A \
                    = gen_anchor_target(cls_output[i].shape[-2:], xs[i].shape[-2:], gt_boxes[i][np.newaxis, :])

                #reg_loss_fn = torch.nn.SmoothL1Loss(reduce=False, size_average=False)
                reg_loss_fn = smooth_l1_loss
                reg_loss = reg_loss_fn(reg_output[i], torch.from_numpy(rpn_bbox_targets).to(DEVICE), torch.from_numpy(rpn_bbox_inside_weights).to(DEVICE), torch.from_numpy(rpn_bbox_outside_weights).to(DEVICE))

                cls_loss_fn = torch.nn.CrossEntropyLoss(reduce=False, size_average=False)
                
                rpn_labels = rpn_labels.reshape(A, feat_h, feat_w) # from (1, 1, A*17, 17) to (A, 17, 17)
                logits = cls_output[i].view(A, 2, feat_h, feat_w) # from (2*A, 17, 17) to (A, 2, 17, 17)
                cls_loss = cls_loss_fn(logits, torch.from_numpy(rpn_labels).to(DEVICE).long()) # (A, 17, 17)

                mask = np.ones_like(rpn_labels)
                mask[np.where(rpn_labels==-1)] = 0 # mask where we 'don't care'
                mask = torch.from_numpy(mask).to(DEVICE)
                cls_loss = torch.sum(cls_loss * mask) / torch.sum(mask)

                #print("{} + l * {} = {}".format(cls_loss, reg_loss, cls_loss+cfg.TRAIN.LAMBDA*reg_loss))
                
                total_cls_loss += cls_loss
                total_reg_loss += reg_loss
                total_loss += cls_loss + cfg.TRAIN.LAMBDA * reg_loss
            
            total_loss /= batch_size
            total_reg_loss /= batch_size
            total_cls_loss /= batch_size
            print(f"Epoch{i_ep} Iter{i_iter} --- total_loss: {total_loss:.4f}, cls_loss: {total_cls_loss:.4f}, reg_loss: {total_reg_loss:.4f}")
            total_loss.backward()
            optimizer.step()

        ######## Save the current model
        print("Saving model...")
        if not os.path.exists("./output/weights"):
            os.makedirs("./output/weights")
        torch.save(net.state_dict(), f"./output/weights/dasiam_{i_ep}.pkl")

    print("Training completed.")

