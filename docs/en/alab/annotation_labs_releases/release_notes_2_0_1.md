---
layout: docs
header: true
seotitle: Annotation Lab | John Snow Labs
title: Annotation Lab Release Notes 2.0.1
permalink: /docs/en/alab/annotation_labs_releases/release_notes_2_0_1
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: annotation-lab
---

<div class="h3-box" markdown="1">

## 2.0.1
### Highlights
- Inter-Annotation Agreement Charts. To get a measure of how well multiple annotators can make the same annotation decision for a certain category, we are shipping seven different charts. To see these charts users can click on the third tab “Inter-Annotator Agreement” of the Analytics Dashboard of NER projects. There are dropdown boxes to change annotators for comparison purposes. It is also possible to download the data of some charts in CSV format by clicking the download button present at the bottom right corner of each of them. 
- Updated CONLL Export. In previous versions, numerous files were created based on Tasks and Completions. There were issues in the Header and no sentences were detected. Also, some punctuations were not correctly exported or were missing. The new CONLL export implementation results in a single file and fixes all the above issues. As in previous versions, if only Starred completions are needed in the exported file, users can select the “Only ground truth” checkbox.
- Search tasks by label. Now, it is possible to list the tasks based on some annotation criteria. Examples of supported queries: "label: ABC", "label: ABC=DEF", "choice: Mychoice", "label: ABC=DEF".
- Validation of labels and models is done beforehand. An error message is shown if the label is incompatible with models.
- Transfer Learning support for Training Models. Now its is possible to continue model training from an already available model. If a Medical NER model is present in the system, the project owner or manager can go to Advanced Options settings of the Training section in the Setup Page and choose it to Fine Tune the model. When Fine Tuning is enabled, the embeddings that were used to train the model need to be present in the system. If present, it will be automatically selected, otherwise users need to go to the Models Hub page and download or upload it.
- Training Community Models without the need of License. In previous versions, Annotation Lab didn’t allow training without the presence of Spark NLP for Healthcare license. But now the training with community embeddings is allowed even without the presence of Valid license. 
- Support for custom training scripts. If users want to change the default Training script present within the Annotation Lab, they can upload their own training pipeline. In the Training section of the Project Setup Page, only admin users can upload the training scripts. At the moment we are supporting the NER custom training script only.
- Users can now see a proper message on the Modelshub page when annotationlab is not connected to the internet (AWS S3 to be more precise). This happens in air-gapped environments or some issues in the enterprise network.
- Users now have the option to download the trained models from the Models Hub page. The download option is available under the overflow menu of each Model on the “Available Models” tab.
- Training Live Logs are improved in terms of content and readability.
- Not all Embeddings present in the Models Hub are supported by NER and Assertion Status Training. These are now properly validated from the UI.
- Conflict when trying to use deleted embeddings. The existence of the embeddings in training as well as in deployment is ensured and a readable message is shown to users.
- Support for adding custom CA certificate chain. Follow the instructions described in instruction.md file present in the installation artifact.


### Bug fixes

- When multiple paged OCR file was imported using Spark OCR, the task created did not have pagination.
- Due to a bug in the Assertion Status script, the training was not working at all. 
- Any AdminUser could delete the main “admin” user as well as itself. We have added proper validation to avoid such situations.

{:.btn-block}
[Read more](https://www.johnsnowlabs.com/inter-annotator-agreement-charts-transfer-learning-training-without-license-custom-training-script-with-annotation-lab/){:.button.button--primary.button--rounded.button--lg}

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination owl-carousel pagination_big">
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
    <li class="active"><a href="release_notes_2_0_1">2.0.1</a></li>
</ul>