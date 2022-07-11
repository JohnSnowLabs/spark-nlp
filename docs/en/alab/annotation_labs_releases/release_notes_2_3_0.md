---
layout: docs
header: true
seotitle: Annotation Lab | John Snow Labs
title: Annotation Lab Release Notes 2.3.0
permalink: /docs/en/alab/annotation_labs_releases/release_notes_2_3_0
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: annotation-lab
---

<div class="h3-box" markdown="1">

## 2.3.0
### Highlights

- Multipage PDF annotation. Annotation Lab 2.3.0 supports the complete flow of import, annotation, and export for multi-page PDF files. Users have two options for importing a new PDF file into the Visual NER project:
  - Import PDF file from local storage
  - Add a link to the PDF file in the file attribute. 


After import, the user can see a task on the task page with a file name. On the labeling page, the user can view the PDF file with pagination so that the user can annotate the PDF one page at a time. After completing the annotation, the user can submit a task and it is ready to be exported to JSON and the user can import this exported file into any Visual NER project. [Note: Export in COCO format is not yet supported for PDF file]
- Redesign of the Project Setup Page. With the addition of many new features on every release, the Project Setup page became crowded and difficult to digest by users. In this release we have split its main components into multiple tabs: Project Description, Sharing, Configuration, and Training.
- Train and Test Dataset. Project Owner/Manager can tag the tasks that will be used for train and for test purposes. For this, two predefined tags will be available in all projects: `Train` and `Test`.
- Enhanced Relation Annotation. The user experience while annotating relations on the Labeling page has been improved. Annotators can now select the  desired relation(s) by clicking the plus "+" sign present next to the relations arrow.
- Other UX improvements: 
  - Multiple items selection with Shift Key in Models Hub and Tasks page,
  - Click instead of hover on more options in Models Hub and Tasks page,
  - Tabbed View on the ModelsHub page.

### Bug fixes
- Validations related to Training Settings across different types of projects were fixed.
- It is not very common to upload an expired license given a valid license is already present. But in case users did that there was an issue while using a license in the spark session because only the last uploaded license was used. Now it has been fixed to use any valid license no matter the order of upload.
- Sometimes search and filters in the ModelsHub page were not working. Also, there was an issue while removing defined labels on the Upload Models Form. Both of these issues are fixed.

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
    <li class="active"><a href="release_notes_2_3_0">2.3.0</a></li>
    <li><a href="release_notes_2_2_2">2.2.2</a></li>
    <li><a href="release_notes_2_1_0">2.1.0</a></li>
    <li><a href="release_notes_2_0_1">2.0.1</a></li>
</ul>