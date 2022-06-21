---
layout: docs
comment: no
header: true
seotitle: Release Notes | John Snow Labs
title: Release Notes
permalink: /docs/en/alab/release_notes
key: docs-training
modify_date: "2022-04-05"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
    nav: annotation-lab
---

<div class="h3-box" markdown="1">

## 3.3.0

Release date: **21-06-2022**

We are very excited to announce the release of Annotation Lab v3.3.0 which includes a highly requested new feature for displaying the confidence scores for NER preannotations as well as the ability to filter preannotations by confidence. Also, benchmarking data can now be checked for some of the models on the Models Hub page. This version also includes IAA charts for Visual NER Projects, upgrades of the Spark NLP libraries and fixes for some of the identified Common Vulnerabilities and Exposures (CVEs). Below are more details on the release content.

### Highlights

- **Confidence Scores for Preannotations**. When running preannotations on a Text project, one extra piece of information is now present for the automatic annotations - the confidence score. This score is used to show the confidence the model has for each of the labeled chunks. It is calculated based on the benchmarking information of the model used to preannotate and on the score of each prediction. The confidence score is available when working on Named Entity Recognition, Relation, Assertion, and Classification projects and is also generated when using NER Rules. On the Labeling screen, when selecting the Prediction widget, users can see that all preannotation in the Results section now have a score assigned to them.
- **IAA charts are now available for Visual NER Projects.** IAA (Inter-Annotator Agreement) charts were available only for text-based projects. With this release, Annotation Lab supports IAA charts for Visual NER project as well.
- **Auto-save completions**. The work of annotators is now automatically saved behind the scenes. This way, the user does not risk losing his/her work in case of unforeseen events and does not have to frequently hit the Save/Update button.
- **Improvement of UX for Active Learning**. Information about the previously triggered Active Learning is displayed along with the number of completions required for the next training. Also when the conditions that trigger active learning for a project using a healthcare model are met and all available licenses are in use, an error message appears on the Training and Active Learning page informing the user to make room for the new training server.
- **Upgraded Spark NLP and Spark NLP for Healthcare v3.5.3 and Spark OCR v3.13.0.** With this we have also updated the list of supported models into the Models Hub page.
- **Support for BertForSequenceClassification and MedicalBertForSequenceClassification models.** From this version on, support was added for BertForTokenClassification, MedicalBertForTokenClassifier, BertForSequenceClassification and MedicalBertForSequenceClassification.
 

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination owl-carousel pagination_big">
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