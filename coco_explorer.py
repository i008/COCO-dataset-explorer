import argparse
import json
import os
import re

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from pycocotools.coco import COCO

from cocoinspector import CoCoInspector


@st.cache(allow_output_mutation=True)
def get_inspector(coco_train, coco_predictions, images_path, eval_type,
                  iou_min, iou_max, filter_categories):
    coco_gt = COCO(coco_train)
    if coco_predictions is None:
        coco_dt = coco_gt
    else:
        coco = json.load(open(coco_predictions))
        if isinstance(coco, dict) and 'annotations' in coco:
            coco = coco['annotations']
        coco_dt = coco.loadRes(coco)
    if filter_categories:
        filter_catids = [cat['id'] for cat in coco_gt.dataset['categories']
                         if cat['name'] in filter_categories.split(',')]
        for ann in coco_gt.anns.values():
            if ann['category_id'] in filter_catids:
                coco_gt.dataset['annotations'].remove(ann)
        coco_gt.createIndex()
        for ann in coco_dt.anns.values():
            if ann['category_id'] in filter_catids:
                coco_dt.dataset['annotations'].remove(ann)
        coco_dt.createIndex()
    inspector = CoCoInspector(coco_gt, coco_dt, base_path=images_path,
                              iou_type=eval_type, iou_min=iou_min, iou_max=iou_max)
    inspector.evaluate()
    inspector.calculate_stats()
    return inspector


def app(args):
    st.set_page_config(layout='wide')
    st.title('COCO Explorer')
    ioumin = st.sidebar.slider("Minimum IoU", min_value=0.0, max_value=1.0, value=args.iou_min)
    ioumax = st.sidebar.slider("Maximum IoU", min_value=0.0, max_value=1.0, value=args.iou_max)
    topbox = st.sidebar.selectbox("Choose what to do ", ['inspect predictions visually',
                                                         'inspect image statistics',
                                                         'inspect annotations',
                                                         'CoCo scores'
                                                         ])
    inspector = get_inspector(args.coco_train, args.coco_predictions, args.images_path,
                              args.eval_type, ioumin, ioumax, args.filter_categories)
    if topbox == 'inspect predictions visually':

        st.sidebar.subheader('Inspect predictions')

        vis_options = {'true positives': 'tp',
                       'ground truth': 'gt',
                       'false negatives': 'fn',
                       'false positives': 'fp',
                       }

        st.sidebar.text("""
        What to show on image
        TP - results matching GT (orange)
        FP - results not matching GT (teal)
        FN - GT not matching results (red)
        GT - all ground truth (green)
        """)
        ms = st.sidebar.multiselect("",
                                    list(vis_options.keys()),
                                    default=list(vis_options.keys())
                                    )

        st.sidebar.subheader('Visual settings')
        size = st.sidebar.slider('plot resolution', min_value=1, max_value=50, value=15)
        score = st.sidebar.slider('score threshold', min_value=0.0, max_value=1.0, value=0.5)

        draw_pred_mask = st.sidebar.checkbox("Draw predictions masks (red)")
        draw_gt_mask = st.sidebar.checkbox("Draw ground truth masks (green)")
        adjust_labels = st.sidebar.checkbox("Optimize label placement")

        r = st.sidebar.radio('Inspect by', options=['image_id', 'category', 'precision'])

        if r == 'image_id':
            path = st.text_input('select image by path or filter by regular expression:',)
            image_ids = inspector.image_ids
            if path:
                r = inspector._path2imageid(path)
                if r < 0:
                    r = 0
                    try:
                        pattern = re.compile(path)
                        image_ids = inspector.get_images_for_file_name(pattern.match)
                    except Exception:
                        image_ids = []
                    if not image_ids:
                        st.error('No such image file_name')
                        image_ids = inspector.image_ids
                else:
                    r = image_ids.index(r)
            else:
                r = 0
            if len(image_ids) > 1:
                r = st.slider('slider trough all images', value=r, min_value=0, max_value=len(image_ids)-1)
            path = inspector._imageid2path(image_ids[r])
            st.text(path)
            print(path)
            f, fn = inspector.visualize_image(image_ids[r],
                                              draw_gt_mask=draw_gt_mask,
                                              draw_pred_mask=draw_pred_mask,
                                              adjust_labels=adjust_labels,
                                              score_threshold=score,
                                              fontsize=size,
                                              show_only=[vis_options[o] for o in ms],
                                              figsize=(size, size))
            st.pyplot(f[0])
            imscores = inspector.image_scores_agg
            if image_ids[r] in imscores.index:
                st.dataframe(imscores.loc[image_ids[r]])

        if r == 'category':
            category = st.sidebar.selectbox(label='select by category',
                                            options=[c['name'] for c in inspector.categories])
            exclusive = st.sidebar.checkbox(label='Show only this category')
            print(category)
            if category:
                image_ids = inspector.get_images_with_category(category)
                image_ids = np.random.permutation(image_ids)
                for img in image_ids[:10]:
                    print(img)
                    f, fn = inspector.visualize_image(img,
                                                      draw_gt_mask=draw_gt_mask,
                                                      draw_pred_mask=draw_pred_mask,
                                                      adjust_labels=adjust_labels,
                                                      only_categories=[category] if exclusive else [],
                                                      score_threshold=score,
                                                      show_only=[vis_options[o] for o in ms],
                                                      fontsize=size,
                                                      figsize=(size, size))

                    st.pyplot(f[0])
                if len(image_ids) > 10:
                    st.button('Sample 10 more images')

        if r == 'precision':
            prec_min = st.slider(label='Minimum precision', min_value=0.0, max_value=1.0, value=0.0)
            prec_max = st.slider(label='Maximum precision', min_value=0.0, max_value=1.0, value=0.3)
            agg = inspector.image_scores_agg
            agg = agg[agg.precision.between(prec_min, prec_max)]

            image_ids = np.random.permutation(agg.index)
            for img in image_ids[:10]:
                print(img)
                f, fn = inspector.visualize_image(img,
                                                  draw_gt_mask=draw_gt_mask,
                                                  draw_pred_mask=draw_pred_mask,
                                                  adjust_labels=adjust_labels,
                                                  score_threshold=score,
                                                  show_only=[vis_options[o] for o in ms],
                                                  fontsize=size,
                                                  figsize=(size, size))

                st.pyplot(f[0])
            if len(image_ids) > 10:
                st.button('Sample 10 more images')


    elif topbox == 'inspect image statistics':
        st.plotly_chart(px.histogram(inspector.images_df, x='aspect_ratio', title='aspect ratio distribiution',
                                     hover_name=inspector.images_df.file_name))
        st.plotly_chart(px.histogram(inspector.images_df, x='width', title='image width distribiution'))
        st.plotly_chart(px.histogram(inspector.images_df, x='height', title="image height distribiution"))

    elif topbox == 'inspect annotations':
        df = pd.DataFrame(inspector.annot_df.category_name.value_counts().reset_index())
        dfarea = pd.DataFrame(
            inspector.annot_df.groupby('category_name')['area'].mean().sort_values(ascending=False)).reset_index()

        # annot_aspect = pd.DataFrame(inspector.annot_df.groupby('category_name')['ann_ar'].mean().sort_values(ascending=False)).reset_index()

        df.columns = ['category_name', 'category_count']

        st.plotly_chart(
            px.bar(df, x='category_name', y='category_count', title='annotation count per class'))

        st.plotly_chart(
            px.bar(dfarea, x='category_name', y='area', title='avg object size(area) per class'))

        st.plotly_chart(px.histogram(inspector.annot_df, x='ann_ar', title="Bounding box aspect ratio distribiution"))

    elif topbox == 'CoCo scores':
        st.subheader("Shows per class mAP scores as calculated by pycocotools")
        st.sidebar.header('Inspect predictions')
        df = inspector.ap_per_class()
        st.dataframe(df)
        st.subheader("Average mAP by class")
        st.dataframe(df.mean(axis=1))
        x = df.mean(axis=1).sort_values(ascending=False).reset_index()
        x.columns = ['category', 'AP']
        # print("ok")
        st.plotly_chart(px.bar(x, y='AP', x='category'))
        st.subheader("Original CoCoeval output")
        st.text(body=inspector.cocoeval_scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_train", type=str, required=True, metavar="PATH/TO/COCO.json",
                        help="COCO dataset to inspect")
    parser.add_argument("--coco_predictions", type=str, default=None, metavar="PATH/TO/COCO.json",
                        help="COCO annotations to compare to")
    parser.add_argument("--images_path", type=str, default=os.getcwd(), metavar="PATH/TO/IMAGES/",
                        help="Directory path to prepend to file_name paths in COCO")
    parser.add_argument("--eval_type", type=str, default="bbox", choices={"bbox", "segm", "keypoints"},
                        help="Mode of comparison (where to look for a 'match')")
    parser.add_argument("--iou_min", type=float, default=0.5,
                        help="Initial minimum IoU (overlap) (what constitutes a 'match')")
    parser.add_argument("--iou_max", type=float, default=0.95,
                        help="Initial maximum IoU (overlap) (what constitutes a 'match')")
    parser.add_argument("--filter_categories", type=str, default="", metavar="COMMA-SEPD-LIST",
                        help="Strip annotations for these categories after loading")
    args = parser.parse_args()
    if args.images_path[-1] != '/':
        args.images_path += '/'
    app(args)
