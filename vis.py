import matplotlib.pyplot as plt
import numpy as np
from easyimages.utils import change_box_order
from adjustText import adjust_text

def vis_image(img,
              annotations=None,
              box_order='tlbr',
              show_only=['tp', 'gt'],
              axis_off=False,
              score_threshold=0.5,
              draw_gt_mask=False,
              draw_pred_mask=False,
              adjust_labels=False,
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
            for _ in range(max([c['category_id'] for c in annotations], default=0) + 1)
        ]

        for i, ann in enumerate(annotations[:]):
            if ann['type'] in show_only:
                if ann.get('score', 1) > score_threshold:
                    if ann['type'] in ['gt', 'fn'] and not draw_gt_mask:
                        continue
                    if ann['type'] not in ['gt', 'fn'] and not draw_pred_mask:
                        continue

                    if ann['type'] in ['gt', 'fn']:
                        color_mask = np.array((0, 255, 0))
                    if ann['type'] not in ['gt', 'fn']:
                        color_mask = np.array((255, 0, 0))

                    #                     color_mask = color_masks[ann['category_id']]
                    mask = coco.annToMask(ann).astype(np.bool)
                    img[mask] = img[mask] * 0.5 + color_mask * 0.5

    ax.imshow(img)

    for i, ann in enumerate(annotations[:]):
        if ann['type'] == 'fn' and 'fn' not in show_only:
            ann['type'] = 'gt'
        if ann['type'] not in show_only or ann.get('score', 1) < score_threshold:
            continue

        bbox = change_box_order(np.array([ann['bbox']]), input_order='tlwh', output_order='tlwh')
        x, y, w, h = bbox[0]
        ann_type = ann.get('type')
        label = ann.get('label')
        score = ann.get('score', 1)

        type2color = {
            'gt': 'g',
            'fn': 'r',
            'fp': 'teal',
            'tp': 'orange'
        }

        caption = []
        rectargs = {'fill': False,
                    'edgecolor': type2color[ann_type],
                    'linewidth': max(1, fontsize//10),
                    'linestyle': '-'}
        if ann_type in ['gt', 'fn']:
            caption.append(label)
        elif ann_type == 'fp':
            rectargs['linestyle'] = '--'
            caption.append(label)
            caption.append('{:.2f}'.format(score))
        elif ann_type == 'tp':
            rectargs['linewidth'] = max(1, fontsize//8)
            caption.append(label)
            caption.append('{:.2f}'.format(score))

        ax.add_patch(plt.Rectangle((x, y), w, h, **rectargs))
        if len(caption) > 0:
            if adjust_labels:
                xt = x + 0.5 * w
                yt = y + 0.5 * h
            else:
                xt = x - 10
                yt = y - 10
            ax.text(xt, yt,
                    ': '.join(caption),
                    style='italic',
                    size=fontsize,
                    bbox={'facecolor': type2color[ann_type], 'alpha': 0.3,
                          'pad': max(1, fontsize//15)})

    if adjust_labels:
        adjust_text(ax.texts, add_objects=ax.patches,
                    arrowprops=dict(arrowstyle="->", lw=max(1, fontsize//10)))
        for arrow, patch in zip(ax.texts[len(ax.patches):], ax.patches):
            arrow.arrow_patch.set_color(patch.get_edgecolor())

    # Show
    if axis_off:
        plt.axis('off')
    return fig, ax
