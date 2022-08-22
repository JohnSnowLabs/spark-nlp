---
layout: docs
header: true
seotitle: Annotation Lab | John Snow Labs
title: Annotation Lab Release Notes 2.8.0
permalink: /docs/en/alab/annotation_labs_releases/release_notes_2_8_0
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: annotation-lab
---

<div class="h3-box" markdown="1">

## 2.8.0

Release date: **19-03-2022**

Annotation Lab 2.8.0 simplifies the annotation workflows, adds dynamic pagination features, supports cross-page NER annotation and relation definition for text projects, adds UI features for infrastructure configuration and backup, updates the way the analytics dashboards are processed, offers improved support for rules and support for model training in German and Spanish.

### Highlights

New features offered by Annotation Lab:
- [Dynamic Task Pagination](/docs/en/alab/import#dynamic-task-pagination) replaced the `<pagebreak>` style pagination. 
- [Cross Page Annotation](/docs/en/alab/annotation#cross-page-annotation) is now supported for NER and Relation annotations. 
- [Simplified workflow](/docs/en/alab/annotation#simplified-workflow) are now supported for simpler projects. Furthermore, overall work progress has been added on the Labeling Page. 
- [Infrastucture](/docs/en/alab/infrastructure) used for preannotation and training can now be configured from the Annotation Lab UI.
- Support for [training German and Spanish models](/docs/en/alab/active_learning#train-german-and-spanish-models). 
- Some [changes in Analytics Dashboard](/docs/en/alab/analytics#disabled-analytics) were implemented. By default, the Analytics dashboard page is now disabled. Users can request Admin to enable the Analytics page. The refresh of the charts is done manually.
- [Import & Export Rules](/docs/en/alab/models_hub#import-and-export-rules) from the Model Hub page.
- [Download model dependencies](/docs/en/alab/models_hub#download-of-model-dependencies) is now automatic. 
- The [project configuration box](/docs/en/alab/settings#project-configuration-box) can now be edited in full screen mode. 
- [Trim leading and ending spaces in annotated chunks](/docs/en/alab/annotation#trim-leading-and-ending-spaces-in-annotated-chunks).
- [Reserved words](/docs/en/alab/project_setup#reserved-words-cannot-be-used-in-project-names) cannot be used in project names.
- Task numbering now start from 1.
- 'a' was removed as hotkey for VisualNER multi-chunk selection. Going forward only use 'shift' key for chunk selection. 
- Only alphanumeric characters can be used as the Task Tag Names.
- Allow the export of tasks without completions. 

### Bug Fixes

- On the Labeling Page, the following issues related to completions were identified and fixed:
  - In the Visual NER Project, when an annotator clicks on the Next button to load the next available task, the PDF was not correctly loaded and the text selection doesn't work properly. 
  - Shortcut keys were not working when creating new completions.
  - It happened that completions were no longer visible after creating relations. 
- Previously, after each project configuration edit, when validating the correctness of the configuration the cursor position was reset to the end of the config file. The user had to manually move the cursor back to its previous position to continue editing. Now, the cursor position is saved so that the editing can continue with ease.
- Removing a user from the "UserAdmins" group was not possible. This has been fixed. Any user can be added or removed from the "UserAdmins". 
- In previous versions, choosing an already existing name for the current project did not show any error messages. Now, an error message appears on the right side of the screen asking users to choose another name for the project.
- In the previous version, when a user was deleted and a new user with the same name was created through Keycloak, on the next login the UI did not load. Now, this issue was fixed. 
- Validations were added to Swagger API for completions data such that random values could not be added to completion data via API.
- Previously, when a rule was edited, previously deployed preannotation pipelines using the edited rules were not updated to use the latest version of the rule. Now the user is notified about the edited rule via an alert message "Redeploy preannotation server to apply these changes" on the rule edit form so that the users can redeploy the preannotation model. 

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination owl-carousel pagination_big">
    <li><a href="release_notes_3_2_0">3.2.0</a></li>
    <li><a href="release_notes_3_1_1">3.1.1</a></li>
    <li><a href="release_notes_3_1_0">3.1.0</a></li>
    <li><a href="release_notes_3_0_1">3.0.1</a></li>
    <li><a href="release_notes_3_0_0">3.0.0</a></li>
    <li class="active"><a href="release_notes_2_8_0">2.8.0</a></li>
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