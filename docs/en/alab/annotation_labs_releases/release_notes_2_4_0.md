---
layout: docs
header: true
seotitle: Annotation Lab | John Snow Labs
title: Annotation Lab Release Notes 2.4.0
permalink: /docs/en/alab/annotation_labs_releases/release_notes_2_4_0
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: annotation-lab
---

<div class="h3-box" markdown="1">

## 2.4.0

Annotation Lab v2.4.0 adds relation creation features for Visual NER projects and redesigns the Spark NLP Pipeline Config on the Project Setup Page. Several bug fixes and stabilizations are also included. Following are the highlights:

### Highlights
- Relations on Visual NER Projects. Annotators can create relations between annotated tokens/regions in Visual NER projects. This functionality is similar to what we already had in text-based projects. It is also possible to assign relation labels using the contextual widget (the "+" sign displayed next to the relation arrow).
- SparkNLP Pipeline Config page was redesigned. The SparkNLP Pipeline Config in the Setup Page was redesigned to ease filtering, collapsing, and expanding models and labels. For a more intuitive use, the Add Label button was moved to the top right side of the tab and no longer scrolls with the config list.
- This version also adds many improvements to the new Setup Page. The Training and Active Learning Tabs are only available to projects for which Annotation Lab supports training. When unsaved changes are present in the configuration, the user cannot move to the Training and Active Learning Tab and/or Training cannot be started. When the OCR server was not deployed, imports in the Visual NER project were failing. With this release, when a valid Spark OCR license is present, the OCR server is deployed and the import of pdf and image files is executed.

### Bug fixes
- When a login session is timed out and cookies are expired, users had to refresh the page to get the login screen. This known issue has been fixed and the user will be redirected to the login page.
- When a task was assigned to an annotator who does not have completions for it, the task status was shown incorrectly. This was fixed in this version.
- While preannotating tasks with some specific types of models, only the first few lines were annotated. We have fixed the Spark NLP pipeline for such models and now the entire document gets preannotations. When Spark NLP for Healthcare license was expired, the deployment was allowed but it used to fail. Now a proper message is shown to Project Owners/Managers and the deployment is not started in such cases.
- Inconsistencies were present in training logs and some embeddings were not successfully used for models training.
- Along with these fixes, several UI bugs are also fixed in this release.

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
    <li class="active"><a href="release_notes_2_4_0">2.4.0</a></li>
    <li><a href="release_notes_2_3_0">2.3.0</a></li>
    <li><a href="release_notes_2_2_2">2.2.2</a></li>
    <li><a href="release_notes_2_1_0">2.1.0</a></li>
    <li><a href="release_notes_2_0_1">2.0.1</a></li>
</ul>