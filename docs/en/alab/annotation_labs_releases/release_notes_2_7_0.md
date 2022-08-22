---
layout: docs
header: true
seotitle: Annotation Lab | John Snow Labs
title: Annotation Lab Release Notes 2.7.0
permalink: /docs/en/alab/annotation_labs_releases/release_notes_2_7_0
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: annotation-lab
---

<div class="h3-box" markdown="1">

## 2.7.0

Release date: **17-02-2022**

Annotation Lab 2.7.0 is here! This is another feature reach release from John Snow Labs - Annotation Lab Team. It is powered by the latest Spark NLP and Spark NLP for Healthcare libraries and offers improved support for Rule Base Annotation. With the upgrade of Spark NLP libraries, the Models Hub page inside the application gets more than 100 new models for English along with the introduction of Spanish and German models. In Visual NER projects it is now easier to annotate cross line chunks. As always, there are many security and stabilizations shipped.

### Highlights
- Annotation Lab 2.7.0 includes Spark NLP 3.4.1 and Spark NLP for Healthcare. Model training is now significantly faster and issues related to Rule-based annotation have been solved. The Models Hub has increased the list of models and old incompatible models are now marked as "incompatible". If there are any incompatible models downloaded on the machine, we recommend deleting them.
- Spanish and German Models have been added to Models Hub. In previous versions of the Annotation Lab, the Models Hub only offered English language models. But from version 2.7.0, models for two other languages are included as well, namely Spanish and German. It is possible to download or upload these models and use them for preannotation, in the same way as for English language models.

- Rule-Based Annotation improvement. Rule-based annotation, introduced in 2.6.0 with limited options, was improved in this release. The Rule creation UI form was simplified and extended, and help tips were added on each field. While creating a rule, the user can define the scope of the rule as being `sentence` or `document`. A new toggle parameter `Complete Match Regex` is added to the rules. It can be toggled on to preannotate the entity that exactly matches the regex or dictionary value regardless of the `Match Scope`. Also case-sensitive is always true (and hence the toggle is hidden in this case) for REGEX while the case-sensitive toggle for dictionary can be toggled on or off. Users can now download the uploaded dictionary of an existing rule. In the previous release, if a dictionary-based rule was defined with an invalid CSV file, the preannotation server would crash and would only recover when the rule was removed from the configuration. This issue has been fixed and it is also possible to upload both vertical and horizontal CSV files consisting of multi-token dictionary values.

- Flexible annotations for Visual NER Projects. The chunk annotation feature added to Visual NER projects, allows the annotation of several consecutive tokens as one chunk. It also supports multiple lines selection. Users can now select multiple tokens and annotate them together in Visual NER Projects. The label assigned to a connected group can be updated. This change will apply to all regions in the group.

- Constraints for relation labeling can be defined. While annotating projects with `Relations` between `Entities`, defining constraints (the direction, the domain, the co-domain) of relations is important. Annotation Lab 2.7.0 offers a way to define such constraints by editing the Project Configuration. The Project Owner or Project Managers can specify which `Relation` needs to be bound to which `Labels` and in which direction. This will hide some Relations in Labeling Page for NER Labels which will simplify the annotation process and will avoid the creation of any incorrect relations in the scope of the project.

### Security
Security issues related to SQL Injection Vulnerability and Host Header Attack were fixed in this release.

### Bug Fixes
- Issues related to chunk annotation; Incorrect bounding boxes, Multiple stacking of bounding boxes, Inconsistent IDs of the regions, unchanged labels of one connected region to other were identified and fixed and annotators can now select multiple tokens at once and annotate them as a single chunk
- In the previous release, after an Assertion Status model was trained, it would get deployed without the NER model and hence the preannotation was not working as expected. Going forward, the trained Assertion Model cannot be deployed for projects without a NER model. For this to happen, the "Yes" button in the confirmation box for deploying an assertion model right after training is enabled only when the Project Configuration consists of at least one NER model.
- A bug in the default Project templates (Project Setup page) was preventing users to create projects using “Conditional classification” and “Pairwise comparison" templates. These default Project templates can be used with no trouble as any other 40+ default templates.
- Reviewers were able to view unassigned submitted tasks via the "Next" button on the Labeling page. This bug is also fixed now and the reviewers can only see tasks that are assigned to them both on the Task List page or while navigating through the "Next" button.
- For better user experience, the labeling page has been optimized and the tasks on the page render quicker than in previous versions. When adding a user to the UserAdmins group, the delay in enabling the checkbox has been fixed.

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
    <li class="active"><a href="release_notes_2_7_0">2.7.0</a></li>
    <li><a href="release_notes_2_6_0">2.6.0</a></li>
    <li><a href="release_notes_2_5_0">2.5.0</a></li>
    <li><a href="release_notes_2_4_0">2.4.0</a></li>
    <li><a href="release_notes_2_3_0">2.3.0</a></li>
    <li><a href="release_notes_2_2_2">2.2.2</a></li>
    <li><a href="release_notes_2_1_0">2.1.0</a></li>
    <li><a href="release_notes_2_0_1">2.0.1</a></li>
</ul>