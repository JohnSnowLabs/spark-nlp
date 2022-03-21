---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: Analytics  
permalink: /docs/en/alab/analytics
key: docs-training
modify_date: "2021-08-30"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
    nav: annotation-lab
---

## Inter-Annotator Agreement Charts (IAA)

Inter Annotator Agreement charts can be used by annotators, reviewers, and managers for identifying contradictions or disagreements within the stared completions. When multiple annotators work on the same tasks, IAA charts are handy to measure how well the annotations  created by different annotators align. 
IAA chart can also be used to identify outliers in the labeled data or to compare manual annotations with model predictions. 
To get access to the IAA charts, navigate on the third tab of the Analytics Dashboard of NER projects, called "Inter-Annotator Agreement". Several charts should appear on the screen with a default selection of annotators to compare. The dropdown boxes below each chart allow you to change annotators for comparison purposes. It is also possible to download the data generated for some charts in CSV format by clicking the download button present at the bottom right corner of each of them.

Note: Only the Submitted and starred (Ground Truth) completions are used to render these charts.
<img class="image image--xl" src="/assets/images/annotation_lab/2.0.0/high_level_IAA.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>
- Shows agreement with Label wise breakdown
<img class="image image--xl" src="/assets/images/annotation_lab/2.0.0/IAA_on_common_tasks.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>
- Shows whether two Annotators agree on every annotated Chunks
<img class="image image--xl" src="/assets/images/annotation_lab/2.0.0/comparison_by_chunk.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>
- Shows agreement between one Annotator and Preannotation result on every annotated Chunks
<img class="image image--xl" src="/assets/images/annotation_lab/2.0.0/predictions_gt.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>
- Shows Labels of each Chunk by one Annotator and context in the tasks
<img class="image image--xl" src="/assets/images/annotation_lab/2.0.0/all_chunks.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>
- Shows the frequency of Chunk-Label by one Annotator
<img class="image image--xl" src="/assets/images/annotation_lab/2.0.0/chunks_by.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>
- Shows the frequency of Chunk-Annotatory by one Label
<img class="image image--xl" src="/assets/images/annotation_lab/2.0.0/chunks_extracted.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>


## Disabled Analytics 

By default, the Analytics dashboards page is disabled. Users can request Admin to enable the Analytics page. The request can be seen by the Admin user’s account, on the Settings page. Once the Admin user approves the request, users can access the analytics page. 

![analytics-permission](https://user-images.githubusercontent.com/10126570/159010289-edbc211c-b2e9-405f-82ba-6a26662fc661.gif)

## Refresh Analytics 
A refresh button is added to this analytics page. Changes made by the annotators (like creating tasks, adding new completion, etc.) will not be automatically reflected in the Analytics charts. The latter can be updated by pressing the refresh button. 


## Download data used for charts

CSV file for specific charts can be downloaded using the new download button which will call specific API endpoints: /api/projects/{project_name}/charts/{chart_type}/download_csv
 
 ![Screen Recording 2022-03-08 at 3 47 49 PM](https://user-images.githubusercontent.com/17021686/158564836-691a2b79-f3ca-4317-ad31-51cfbc9d71df.gif)

## IAA between manager and annotator

Starting from version 2.8.0, Inter Annotator Agreement(IAA) charts allow the comparison between annotations produced by annotators, project managers, or reviewers.

 ![Screen Recording 2022-03-14 at 1 45 03 PM](https://user-images.githubusercontent.com/17021686/158566408-ea39764f-5ceb-4dd3-b1df-09a3ca791b83.gif)