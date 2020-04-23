import matplotlib.pyplot as plt
import numpy as np
from easyimages.utils import change_box_order


def vis_image(img,
              annotations=None,
              box_order='tlbr',
              show_only=['tp', 'gt'],
              axis_off=False,
              score_threshold=0.5,
              draw_gt_mask=False,
              draw_pred_mask=False,
              coco=None,
              fontsize=15,
              figsize=(10, 10),
              ):
    """
    :param figsize:
    :param img: PIL.Image
    :param boxes: [[x1,y1,x2,y2], ... ]
    :param label_names:  ['car','dog' ... ]
    :param scores:  [0.5, 1]
    :param box_order: 'tlbr', 'tlwh'
    :param axis_off:
    :return:
    """

    # Plot image
    fig = plt.figure(figsize=figsize, frameon=False)
    fig.subplots_adjust(bottom=0)
    fig.subplots_adjust(top=1)
    fig.subplots_adjust(right=1)
    fig.subplots_adjust(left=0)
    ax = fig.add_subplot(1, 1, 1)

    if draw_gt_mask or draw_pred_mask:
        img = np.array(img)
        np.random.seed(42)
        color_masks = [
            np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            for _ in range(max([c['category_id'] for c in annotations]) + 1)
        ]

        for i, ann in enumerate(annotations[:]):
            if ann['type'] in show_only:
                if ann.get('score', 1) > score_threshold:
                    if ann['type'] == 'gt' and not draw_gt_mask:
                        continue
                    if ann['type'] != 'gt' and not draw_pred_mask:
                        continue

                    if ann['type'] == 'gt':
                        color_mask = np.array((0, 255, 0))
                    if ann['type'] != 'gt':
                        color_mask = np.array((255, 0, 0))

                    #                     color_mask = color_masks[ann['category_id']]
                    mask = coco.annToMask(ann).astype(np.bool)
                    img[mask] = img[mask] * 0.5 + color_mask * 0.5

    ax.imshow(img)

    for i, ann in enumerate(annotations[:]):
        caption = []
        if ann['type'] not in show_only or ann.get('score', 1) < score_threshold:
            continue

        bbox = change_box_order(np.array([ann['bbox']]), input_order='tlwh', output_order='tlwh')
        x, y, w, h = bbox[0]
        ann_type = ann.get('type')
        label = ann.get('label')
        score = ann.get('score', 1)

        type2color = {
            'gt': 'g',
            'pred': 'r',
            'fp': 'teal',
            'tp': 'orange'
        }

        if ann_type == 'gt':
            ax.add_patch(
                plt.Rectangle((x, y), w, h, fill=False, edgecolor=type2color[ann_type], linewidth=3, linestyle='-'))
            caption.append(label)
        elif ann_type == 'pred':
            ax.add_patch(
                plt.Rectangle((x, y), w, h, fill=False, edgecolor=type2color[ann_type], linewidth=3, linestyle='-'))
            caption.append(label)
            caption.append('{:.2f}'.format(score))
        elif ann_type == 'fp':
            ax.add_patch(
                plt.Rectangle((x, y), w, h, fill=False, edgecolor=type2color[ann_type], linewidth=3, linestyle='--'))
            caption.append(label)
            caption.append('{:.2f}'.format(score))
        elif ann_type == 'tp':
            ax.add_patch(
                plt.Rectangle((x, y), w, h, fill=False, edgecolor=type2color[ann_type], linewidth=5, linestyle='-'))
            caption.append(label)
            caption.append('{:.2f}'.format(score))

        if len(caption) > 0:
            ax.text(x - 10, y - 10,
                    ': '.join(caption),
                    style='italic',
                    size=fontsize,
                    bbox={'facecolor': type2color[ann_type], 'alpha': 0.3, 'pad': 2})

    # Show
    if axis_off:
        plt.axis('off')
    return fig, ax
