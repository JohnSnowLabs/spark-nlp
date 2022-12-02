---
layout: docs
header: true
seotitle: Annotation Lab | John Snow Labs
title: Annotation Lab Release Notes 2.2.20
permalink: /docs/en/alab/annotation_labs_releases/release_notes_2_2_2
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: annotation-lab
---

<div class="h3-box" markdown="1">

## 2.2.2
### Highlights
- Support for pretrained Relation Extraction and Assertion Status models. A valid Spark NLP for HealthCare License is needed to download pretrained models via the Models Hub page. After download, they can be added to the Project Config and used for preannotations.
- Support for uploading local images. Until this version, only images from remote URLs could be uploaded for Image projects. With this version the Annotation Lab supports uploading images from you local storage/computer. It is possible to either import one image or multiple images by zipping them together. The maximum image file size is 16 MB. If you need to upload files exceding the default configuration, please contact your system administrator who will change the limit size in the installation artifact and run the upgrade script.
- Improved support for Visual NER projects. A sample task can be imported from the Import page by clicking the "Add Sample Task" button. Also default config for the Visual NER project contains zoom feature which supports maximum possible width for low resolution images when zooming.
- Improved Relation Labeling. Creating numerous relations in a single task can look a bit clumsy. The limited space in Labeling screen, the relation arrows and different relation types all at once could create difficulty to visualize them properly. We improved the UX for this feature:
  - Spaces between two lines if relations are present
  - Ability to Filter by certain relations
  - When hovered on one relation, only that is focused
- Miscellaneous. Generally when a first completion in a task is submitted, it is very likely for that completion to be the ground truth for that task. Starting with this version, the first submitted completion gets automatically starred. Hitting submit button on next completion, annotator are asked to either just submit or submit and star it.

### Bug fixes
- On restart of the Annotation Lab machine/VM all Downloaded models (from Models Hub) compatible with Spark NLP 3.1 version were deleted. We have now fixed this issue. Going forward, it is user's responsibility to remove any incompatible models. Those will only be marked as "Incompatible" in Models Hub.
- This version also fixes some reported issues in training logs.
- The CONLL exports were including Assertion Status labels too. Going forward Assertion Status labels will be excluded given correct Project Config is setup.

{:.btn-block}
[Read more](https://www.johnsnowlabs.com/active-learning-for-relation-extraction-and-assertion-status-models-with-annotation-lab/){:.button.button--primary.button--rounded.button--lg}

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
    <li class="active"><a href="release_notes_2_2_2">2.2.2</a></li>
    <li><a href="release_notes_2_1_0">2.1.0</a></li>
    <li><a href="release_notes_2_0_1">2.0.1</a></li>
</ul>