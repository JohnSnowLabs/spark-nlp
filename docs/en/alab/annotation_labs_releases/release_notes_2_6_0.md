---
layout: docs
header: true
seotitle: Annotation Lab | John Snow Labs
title: Annotation Lab Release Notes 2.6.0
permalink: /docs/en/alab/annotation_labs_releases/release_notes_2_6_0
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: annotation-lab
---

<div class="h3-box" markdown="1">

## 2.6.0
Annotation Lab v2.6.0 improves the performance of the `Project Setup` page, adds a "View as" option in the `Labeling` Page, improves the layout of OCR-ed documents, adds the option to stop training and model server deployment from UI. Many more cool features are also delivered in this version to enhance usability and stabilize the product. Here are details of features and bug fixes included in this release.

### Highlights
- Performance improvement in Setup page. In previous versions of Annotation Lab, changes in Project Configuration would take a long time to validate if that project included a high volume of completions. The configuration validation time is now almost instant, even for projects with thousand of tasks. Multiple tests were conducted on projects with more than 13K+ tasks and thousands of extractions per task. For all of those test situations, the validation of the Project Configuration took under 2 seconds. Those tests results were replicated for all types of projects including NER, Image, Audio, Classification, and HTML projects.
- New "View as" option in the labeling screen. When a user has multiple roles (Manager, Annotator, Reviewer), the Labeling Page should present and render different content and specific UX, depending on the role impersonated by the user. For a better user experience, this version adds a "View as" switch in the Labeling Page. Once the "View as" option is used to select a certain role, the selection is preserved even when the tab is closed or refreshed.
- OCR Layout improvement. In previous versions of the Annotation Lab, layout was not preserved in OCRed tasks. Recognized texts would be placed in a top to bottom approach without considering the paragraph each token belonged to. From this version on, we are using layout-preserving transformers from Spark OCR. As a result, tokens that belong to the same paragraph are now grouped together, producing more meaningful output.
- Ability to stop training and model server deployment. Up until now, training and model server deployment could be stopped by system admins only. This version of Annotation Lab provides Project Owners/Managers with the option to stop these processes simply by clicking a button in the UI. This option is necessary in many cases, such as when a manager/project owner starts the training process on a big project that takes a lot of resources and time, blocking access to preannotations to the other projects. 
- Display meaningful message when training fails due to memory issues. In case the training of a model fails due to memory issues, the reason for the failure are available via the UI (i.e. out of memory error).
- Allow combining NER labels and Classification classes from Spark NLP pipeline config. The earlier version had an issue with adding `choice` from the predefined classification model to an existing NER project. This issue has been fixed in this version.

### Bug Fixes
- Previously there was a UI reloading issue when a User was removed from the "Annotators" user group, which has now been fixed. The user can log in without the reloading issue, a warning is shown in UI regarding the missing annotator privilege.
- Also, setting up the `HTML NER Tagging` project was not possible in the earlier version which has been fixed in this release.
- On the labeling page, the renamed title of the next served task was not displayed. Similarly, in the Import page, the count of the tasks imported was missing in the Import Status dialog box. Now both these issues are fixed.

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
    <li class="active"><a href="release_notes_2_6_0">2.6.0</a></li>
    <li><a href="release_notes_2_5_0">2.5.0</a></li>
    <li><a href="release_notes_2_4_0">2.4.0</a></li>
    <li><a href="release_notes_2_3_0">2.3.0</a></li>
    <li><a href="release_notes_2_2_2">2.2.2</a></li>
    <li><a href="release_notes_2_1_0">2.1.0</a></li>
    <li><a href="release_notes_2_0_1">2.0.1</a></li>
</ul>