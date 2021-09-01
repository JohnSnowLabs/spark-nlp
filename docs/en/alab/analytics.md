---
layout: docs
comment: no
header: true
title: Analytics  
permalink: /docs/en/alab/analytics
key: docs-training
modify_date: "2021-08-30"
use_language_switcher: "Python-Scala"
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

