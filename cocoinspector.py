import sys
from io import StringIO

import numpy as np
import pandas as pd
from PIL import Image

from pycoco import COCOeval
from vis import vis_image
import matplotlib.pyplot as plt


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


class CoCoInspector():

    def __init__(self, coco_gt, coco_det=None, base_path=None, iou='bbox', *args, **kwargs):
        self.coco_gt = coco_gt
        self.coco_dt = coco_det
        self.cocoeval = None
        self.iou = 'bbox'
        self.base_path = base_path
        self.cat2id = dict([(v['name'], v['id']) for v in self.coco_dt.cats.values()])

        self.threshold = 0.3  # default per image scoring threshold

    def evaluate(self):
        if self.coco_gt and self.coco_dt:
            self.cocoeval = COCOeval(self.coco_gt, self.coco_dt, self.iou)
            self.cocoeval.evaluate()
            self.cocoeval.accumulate()
            with Capturing() as capture:
                self.cocoeval.summarize()
            self.pcp = self.cocoeval.per_class_precisions
            self.cocoeval_scores = '\n'.join(capture)
            self.image_scores_agg, self.image_scores = self.per_image_scores(self.threshold)

        else:
            raise ValueError('Cant create eval coco without detections')

    def calculate_stats(self):
        df = pd.DataFrame(self.coco_gt.loadImgs(self.coco_gt.getImgIds()))
        df['aspect_ratio'] = df.width / df.height
        self.images_df = df

        all_anns = self.coco_gt.loadAnns(self.coco_gt.getAnnIds())
        dfannot = pd.DataFrame.from_records(all_anns)[['area', 'category_id', 'bbox']]

        dfannot['ann_ar'] = dfannot.bbox.apply(lambda x: x[2] / x[3])
        dfannot['category_name'] = dfannot.category_id.apply(lambda x: self.coco_gt.cats[x]['name'])

        self.annot_df = dfannot

    def per_image_scores(self, threshold=0.3):
        new_r = []
        for r in self.cocoeval.evalImgs:
            if r:
                ids_above_threshold = set([(a, b)[0] for a, b in zip(r['dtIds'], r['dtScores']) if b > threshold])
                gtids = set(list(r['gtMatches'][0]))  # 0 means for default IoU match == 0.5

                TP = ids_above_threshold & gtids
                FP = ids_above_threshold - gtids
                FN = gtids - ids_above_threshold
                new_r.append({'image_id': r['image_id'],
                              'tp_i': list(TP),
                              'fp_i': list(FP),
                              'fn_i': list(FN),
                              'gt_i': list(gtids),
                              'tp_c': len(TP),
                              'fp_c': len(FP),
                              'fn_c': len(FN),
                              'gtcount': len(gtids),
                              'category_id': r['category_id']})

        df = pd.DataFrame(new_r).drop_duplicates(subset=['image_id', 'category_id'])
        df = df.groupby('image_id').sum()
        df['precision'] = df.tp_c / (df.tp_c + df.fp_c)

        return df, pd.DataFrame(new_r)

    def ap_per_class(self):
        df = pd.DataFrame(self.cocoeval.per_class_precisions).T
        df.columns = self.cocoeval.ap_per_class_columns
        return df

    @property
    def image_ids(self):
        return list(self.coco_gt.imgs.keys())

    def get_random_images_with_category(self, category):
        allanno = self.coco_gt.loadAnns(self.coco_gt.getAnnIds())
        image_ids = [a['image_id'] for a in allanno if a['category_id'] == self.cat2id[category]]
        return image_ids

    @property
    def categories(self):
        categories = [{'name': v['name'], 'id': v['id']} for v in self.coco_dt.cats.values()]
        return sorted(categories, key=lambda x: x['id'])

    @staticmethod
    def _get_detections(coco, image_id):
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
        return annotations

    def _imageid2name(self, image_id):
        return self.coco_gt.loadImgs(ids=[image_id])[0]['file_name']

    def _imageid2path(self, image_id):
        return self.base_path + self._imageid2name(image_id)

    def get_detection_matches(self, image_id):
        try:
            gtmatches = np.concatenate([a['gtMatches'].ravel() for a in
                                        [c for c in self.cocoeval.evalImgs if c and c['image_id'] == image_id]]).astype(
                int)
        except ValueError:
            gtmatches = []
        try:
            dtmatches = np.concatenate([a['dtMatches'].ravel() for a in
                                        [c for c in self.cocoeval.evalImgs if c and c['image_id'] == image_id]]).astype(
                int)
        except ValueError:
            dtmatches = []
        return list(set(gtmatches)), list(set(dtmatches))

    def organize_annotations(self, all_annotations, gtmatches):
        collect = []
        for a in all_annotations:
            a['label'] = self.coco_gt.cats[a['category_id']]['name']
            if 'score' not in a and a['id']:
                a['type'] = 'gt'
                collect.append(a)
                continue
            if 'score' in a: a['type'] = 'pred'
            if a['id'] in gtmatches: a['type'] = 'tp'
            if a['id'] not in gtmatches: a['type'] = 'fp'

            collect.append(a)
        return collect

    def visualize_image(self, image_id,
                        show_only=('gt', 'tp'),
                        score_threshold=0.1,
                        draw_gt_mask=True,
                        draw_pred_mask=True,
                        fontsize=20,
                        figsize=(10, 10),
                        dpi=200,
                        ):
        annotations = self._get_detections(self.coco_gt, image_id)
        if self.coco_dt:
            dt_annotations = self._get_detections(self.coco_dt, image_id)
            gtmatches, dtmatches = self.get_detection_matches(image_id)
            annotations = annotations + dt_annotations
            annotations = self.organize_annotations(annotations, gtmatches)

        image = Image.open(self._imageid2path(image_id))
        # cannot work with 16/32 bit or float images due to Pillow#3011 Pillow#3159 Pillow#3838
        assert not image.mode.startswith(('I', 'F')), "image %d has unsupported color mode" % image_i
        image = image.convert('RGB')
        f = vis_image(image, annotations,
                      show_only=show_only,
                      score_threshold=score_threshold,
                      draw_gt_mask=draw_gt_mask,
                      coco=self.coco_gt,
                      axis_off=False,
                      figsize=figsize,
                      fontsize=fontsize,
                      draw_pred_mask=draw_pred_mask)

        fig1 = plt.gcf()
        # plt.show()
        plt.draw()
        fn = 'tmpfile.png'
        #fig1.savefig(fn, dpi=dpi)

        return f, fn
