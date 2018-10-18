import numpy as np
from generate_anchors import generate_anchors

def generate_all_anchors(cls_output_shape, xs_shape):
    anchors = generate_anchors(ratios=[0.33, 0.5, 1, 2, 3], scales=np.array([8, ]))
    # anchors are in box format (x1, y1, x2, y2)

    A = anchors.shape[0]
    feat_stride = xs_shape[0] // cls_output_shape[0]

    allowed_border = 0
    height, width = cls_output_shape

    sr_size = xs_shape

    # 1. Generate proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, width) * feat_stride
    shift_y = np.arange(0, height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(),
                        shift_y.ravel(),
                        shift_x.ravel(),
                        shift_y.ravel())).transpose()

    # 2. Add K anochors (1, A, 4) to cell K shifts (K, 1, 4) 
    #    to get shift anchors (K, A, 4) and reshape to (K*A, 4) shifted anchors
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4))) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))

    all_anchors = all_anchors.reshape((K*A, 4)) # of shape (5x22x22, 4)

    """
    # total number of anchors == A * height * width, 
    # where height and width are the size of conv feature map
    total_anchors = int(K*A)

    # Only keep anchors inside the image
    inds_inside = np.where(
        (all_anchors[:, 0] >= -allowed_border) &
        (all_anchors[:, 1] >= -allowed_border) &
        (all_anchors[:, 2] < sr_size[1] + allowed_border) &
        (all_anchors[:, 3] < sr_size[0] + allowed_border)
    )[0]
    anchors = all_anchors[inds_inside, :]
    # after keeping-inside, #anchors drops from 2420 down to 433
    """

    return all_anchors, A  # anchors

if __name__ == '__main__':
    sr_shape = (255, 255)
    conv_shape = (17, 17)
    all_anchors = generate_all_anchors(conv_shape, sr_shape)
    print(all_anchors)

    import cv2
    img = cv2.imread("../data/SPRING2004B69.jpg")
    img = cv2.resize(img, sr_shape)
    for anchor in all_anchors:
        x1y1x2y2 = tuple(map(int, list(anchor)))
        cv2.rectangle(img, x1y1x2y2[:2], x1y1x2y2[2:], 2)
    cv2.imwrite("../data/result.jpg", img)

    print("DONE.")
