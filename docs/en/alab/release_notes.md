---
layout: docs
comment: no
header: true
seotitle: Release Notes | John Snow Labs
title: Release Notes
permalink: /docs/en/alab/release_notes
key: docs-training
modify_date: "2022-11-03"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
    nav: annotation-lab
---

<div class="h3-box" markdown="1">

## 4.2.0

Release date: **02-11-2022**

Annotation Lab 4.2.0 supports projects combining models trained with multiple embeddings for preannotation as well as predefined Demo projects that can be imported with the click of a button for easy experimentations and features testing. The Project Configuration page now has a new "View" step to configure the layout of the Labeling page. The release also includes stabilization and fixes bugs reported by our user community.

Here are the highlights of this release:

### Highlights
- Projects can reuse and combine models trained with different embeddings for pre-annotation. Now, it is easily possible to use models with different embeddings and deploy them as part of the same pre-annotation server. In the customize configuration page all the added models and their embeddings are listed. The list makes it easier for the user to delete the labels of a specific model.
- Demo Projects can be imported for experiments. To allow users access and experiment with already configured and populated projects we have added the option to import predefined Demo projects. This is for helping users understand the various features offered by the Annotation Lab. The user can import demo projects from the Import Project window, by clicking on the Import Demo Project option.
- Visual Update of the Annotation Screen Layout from the View Tab. A new tab - “View” - has been added to the project setup wizard after the “Content Type” selection tab. This gives users the ability to set different layouts based on their needs and preferences.
- Support for Granular License Scopes. This versions brings support for more granular license scopes such as Healthcare: Inference, Healthcare: Training, OCR: Inference or OCR: Training. This is in line with the latest developments of the John Snow Labs licenses.
- Easy Reuse and Editing of Pre-annotations. For an improved usability, when pre-annotations are available for a task, those will be shown by default when accessing the labeling screen. Users can filter them based on the confidence score and the either accept the visible annotations as a new submitted completion or start editing those as part of a new completion.
- Easy Export of Large Visual NER Projects. From version 4.2.0 users will be able to export large NER/ Visual NER projects with a size bigger than 500 MB.
- Smaller Project Tiles on the Projects Dashboard. The size of a project tile was compacted in this version in order to increase the number of project cards that could be displayed on the screen at one time.
- Confusion Matrix in Training Logs for NER projects. With the addition of confusion matrix it will be easier to understand the performance of the model and judge whether the model is underfitting or overfitting.

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination owl-carousel pagination_big">
	<li class="active"><a href="annotation_labs_releases/release_notes_4_2_0">4.2.0</a></li>
    <li><a href="annotation_labs_releases/release_notes_4_1_0">4.1.0</a></li>
    <li><a href="annotation_labs_releases/release_notes_3_5_0">3.5.0</a></li>
	<li><a href="annotation_labs_releases/release_notes_3_4_1">3.4.1</a></li>
    <li><a href="annotation_labs_releases/release_notes_3_4_0">3.4.0</a></li>
    <li><a href="annotation_labs_releases/release_notes_3_3_1">3.3.1</a></li>
    <li><a href="annotation_labs_releases/release_notes_3_3_0">3.3.0</a></li>
    <li><a href="annotation_labs_releases/release_notes_3_2_0">3.2.0</a></li>
    <li><a href="annotation_labs_releases/release_notes_3_1_1">3.1.1</a></li>
    <li><a href="annotation_labs_releases/release_notes_3_1_0">3.1.0</a></li>
    <li><a href="annotation_labs_releases/release_notes_3_0_1">3.0.1</a></li>
    <li><a href="annotation_labs_releases/release_notes_3_0_0">3.0.0</a></li>
    <li><a href="annotation_labs_releases/release_notes_2_8_0">2.8.0</a></li>
    <li><a href="annotation_labs_releases/release_notes_2_7_2">2.7.2</a></li>
    <li><a href="annotation_labs_releases/release_notes_2_7_1">2.7.1</a></li>
    <li><a href="annotation_labs_releases/release_notes_2_7_0">2.7.0</a></li>
    <li><a href="annotation_labs_releases/release_notes_2_6_0">2.6.0</a></li>
    <li><a href="annotation_labs_releases/release_notes_2_5_0">2.5.0</a></li>
    <li><a href="annotation_labs_releases/release_notes_2_4_0">2.4.0</a></li>
    <li><a href="annotation_labs_releases/release_notes_2_3_0">2.3.0</a></li>
    <li><a href="annotation_labs_releases/release_notes_2_2_2">2.2.2</a></li>
    <li><a href="annotation_labs_releases/release_notes_2_1_0">2.1.0</a></li>
    <li><a href="annotation_labs_releases/release_notes_2_0_1">2.0.1</a></li>
</ul>