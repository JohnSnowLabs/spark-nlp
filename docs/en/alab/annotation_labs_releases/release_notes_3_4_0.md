---
layout: docs
header: true
seotitle: Annotation Lab | John Snow Labs
title: Annotation Lab Release Notes 3.4.0
permalink: /docs/en/alab/annotation_labs_releases/release_notes_3_4_0
key: docs-licensed-release-notes
modify_date: 2022-08-01
show_nav: true
sidebar:
    nav: annotation-lab
---

<div class="h3-box" markdown="1">

## 3.4.0

Release date: **01-08-2022**

We are very excited to release Annotation Lab v3.4.0 with support for Visual NER Automated Preannotation and Model Training. Spark NLP and Spark NLP for Healthcare libraries are upgraded to version 4.0. As always known security and bug fixes are also included with it.
Here are the highlights of this release:

### Highlights
- Visual NER Training support. Annotation Lab offers the ability to train Visual NER models, apply active learning for automatic model training, and preannotate image-based tasks with existing models in order to accelerate annotation work. Floating or airgap licenses with scope ocr: inference and ocr: training are required for preannotation and training respectively. The minimal required training configuration is 64 GB RAM, 16 Core CPU for Visual NER Training.
- Visual NER Preannotation. For running preannotation on one or several tasks, the Project Owner or the Manager must select the target tasks and can click on the Preannotate button from the upper right side of the Tasks Page. The minimal required preannotation configuration is 32 GB RAM, 2 Core CPU for Visual NER Model.
- Spark NLP and Spark NLP for Healthcare upgrades. Annotation Lab 3.4.0 uses Spark NLP 4.0.0, Spark NLP for Healthcare 4.0.2 and Spark OCR 3.13.0. 
- Confusion Matrix for Classification Projects. A checkbox is now added on the training page to enable the generation of confusion matrix for classification projects. The confusion matrix is visible in the live training logs as well as in the downloaded training logs.
- Project Import Improvements. The name of the imported project is set according to the name of the imported zip file. Users can now make changes in the content of the exported zip and then zip it back for import into Annotation Lab.
- Task Pagination in Labeling page.  Tasks are paginated based on the number of characters they contain.
- Confidence filter slider is now visible only for preannotations. Previously the confidence filter was applied to both predictions and completions. Since all manual annotations have a confidence score of 1, we decided to only show and apply the confidence filter when the prediction widget is selected.
- Swagger Docs Changes. API docs have been restructured for an easier use and new methods have been added to mirror the new functionalities offered via the UI.
- Confidence score for Rules preannotations. Confidence of rule-based preannotations is now visible on the Labeling screen, the same as that of model-based preannotation.



</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination owl-carousel pagination_big">
    <li><a href="release_notes_3_4_1">3.4.1</a></li>
    <li class="active"><a href="release_notes_3_4_0">3.4.0</a></li>
    <li><a href="release_notes_3_3_1">3.3.1</a></li>
    <li><a href="release_notes_3_3_0">3.3.0</a></li>
    <li><a href="release_notes_3_2_0">3.2.0</a></li>
    <li><a href="release_notes_3_1_1">3.1.1</a></li>
    <li><a href="release_notes_3_1_0">3.1.0</a></li>
    <li><a href="release_notes_3_0_1">3.0.1</a></li>
    <li><a href="release_notes_3_0_0">3.0.0</a></li>
    <li><a href="release_notes_2_8_0">2.8.0</a></li>
    <li><a href="release_notes_2_7_2">2.7.2</a></li>
    <li><a href="release_notes_2_7_1">2.7.1</a></li>
    <li><a href="release_notes_2_7_0">2.7.0</a></li>
    <li><a href="release_notes_2_6_0">2.6.0</a></li>
    <li><a href="release_notes_2_5_0">2.5.0</a></li>
    <li><a href="release_notes_2_4_0">2.4.0</a></li>
    <li><a href="release_notes_2_3_0">2.3.0</a></li>
    <li><a href="release_notes_2_2_2">2.2.2</a></li>
    <li><a href="release_notes_2_1_0">2.1.0</a></li>
    <li><a href="release_notes_2_0_1">2.0.1</a></li>
</ul>