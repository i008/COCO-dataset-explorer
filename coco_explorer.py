import argparse

import pandas as pd
import plotly.express as px
import streamlit as st
from pycocotools.coco import COCO

from cocoinspector import CoCoInspector


@st.cache(allow_output_mutation=True)
def get_inspector(coco_train, coco_predictions, images_path):
    coco = COCO(coco_train)
    coco_dt = coco.loadRes(coco_predictions)
    inspector = CoCoInspector(coco, coco_dt, base_path=images_path)
    inspector.evaluate()
    inspector.calculate_stats()
    return inspector


def app(args):
    st.title('COCO Explorer')
    topbox = st.sidebar.selectbox("Choose what to do ", ['inspect predictions visually',
                                                         'inspect image statistics',
                                                         'inspect annotations',
                                                         'CoCo scores'
                                                         ])
    inspector = get_inspector(args.coco_train, args.coco_predictions, args.images_path)
    if topbox == 'inspect predictions visually':

        st.sidebar.subheader('Inspect predictions')

        vis_options = {'true positives': 'tp',
                       'ground truth': 'gt',
                       'false positives': 'fp',
                       }

        st.sidebar.text("""
        What to show on image
        TP - boxes matched with GT (orange)
        FP - boxes that did not match with GT (teal)
        GT - ground truth annotations (green)
        """)
        ms = st.sidebar.multiselect("",
                                    list(vis_options.keys()),
                                    default=list(vis_options.keys())
                                    )

        st.sidebar.subheader('Visual settings')
        score = st.sidebar.slider('score threshold', min_value=0.0, max_value=1.0, value=0.5)

        draw_pred_mask = st.sidebar.checkbox("Draw predictions masks (red)")
        draw_gt_mask = st.sidebar.checkbox("Draw ground truth masks (green)")

        r = st.sidebar.radio('Inspect by', options=['image_id', 'category', 'precision'])

        if r == 'image_id':
            r = st.slider('slider trough all images', min_value=0, max_value=len(inspector.image_ids))
            f, fn = inspector.visualize_image(inspector.image_ids[r],
                                              draw_gt_mask=draw_gt_mask,
                                              draw_pred_mask=draw_pred_mask,
                                              score_threshold=score,
                                              fontsize=33,
                                              show_only=[vis_options[o] for o in ms],
                                              figsize=(15, 15))

            st.image(fn, use_column_width=True)
            imscores = inspector.image_scores_agg
            st.dataframe(imscores.loc[inspector.image_ids[r]])

        if r == 'category':
            category = st.sidebar.selectbox(label='select by category',
                                            options=[c['name'] for c in inspector.categories])
            print(category)
            if category:
                random_ids = inspector.get_random_images_with_category(category)
                for id in random_ids[:10]:
                    print(id)
                    f, fn = inspector.visualize_image(id,
                                                  draw_gt_mask=draw_gt_mask,
                                                  draw_pred_mask=draw_pred_mask,
                                                  score_threshold=score,
                                                  show_only=[vis_options[o] for o in ms],
                                                  fontsize=30,
                                                  figsize=(20, 20))

                    st.pyplot(f[0])

        if r == 'precision':
            st.text("""
            This will select images in the range of 
            (prec-0.01 to prec+0.01)
            """)
            precision = st.slider(label='precision low->high', min_value=0.0, max_value=1.0)
            agg = inspector.image_scores_agg
            agg = agg[agg.precision.between(precision - 0.01, precision + 0.01)]

            for id in agg.index[:10]:
                f, fn = inspector.visualize_image(id,
                                              draw_gt_mask=draw_gt_mask,
                                              draw_pred_mask=draw_pred_mask,
                                              score_threshold=score,
                                              show_only=[vis_options[o] for o in ms],
                                              fontsize=30,
                                              figsize=(20, 20))

                st.pyplot(f[0])

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
    parser.add_argument("--coco_train", type=str, default=None)
    parser.add_argument("--coco_predictions", type=str, default=None)
    parser.add_argument("--images_path", type=str, default=None)
    args = parser.parse_args()
    app(args)
