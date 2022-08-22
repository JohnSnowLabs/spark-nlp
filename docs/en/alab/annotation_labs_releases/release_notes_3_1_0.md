---
layout: docs
header: true
seotitle: Annotation Lab | John Snow Labs
title: Annotation Lab Release Notes 3.1.0
permalink: /docs/en/alab/annotation_labs_releases/release_notes_3_1_0
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: annotation-lab
---

<div class="h3-box" markdown="1">

## 3.1.0

Release date: **04-05-2022**

We are very excited to release Annotation Lab v3.1.0 which includes support for training large documents, improvements for Visual NER Projects, security fixes and stabilizations. Here are the highlights:

### Highlights

- Support Training of large documents. Spark NLP feature called Memory Optimization Approach is enabled when the training data is greater then 5MB which enables training of model on machines with lower memory resources.
- Improvements in Visual NER Projects:
  - Users can provide title in the input JSON along with the URL for tasks to import. This sets the title of the task accordingly.
  - JSON export for the Visual NER projects contains both chunk and token-level annotations.
  - Sample tasks can be imported into the Visual NER project using any available OCR server (created by another project).
  - Multi-chunk annotation can be done without changing the start token when the end token is the last word on the document.
  - For Visual NER project, users can export tasks in the VOC format for multi-page tasks with/without completions.
- During restoring backup file in the previous versions, the SECRETS (kubernetes) of the old machine needed manual transfer to the target machine. With v3.1.0, all the SECRETS are backed-up automatically along with database backup and hence they are restored without any hassle.
- Integration with my.johnsnowlabs.com, this means the available licenses can be easily imported by Admin users of Annotation Lab without having to download or copy them manually.
- The maximum number of words/tokens that can be set in a single page in labeling screen is now limited to 1000.
- For a large number of multiple relations, the previous version of Annotation Lab used Prev and Next identifiers which was not optimal for mapping to the correct pairs. For increased usability and clarity , the pair connections now use numerical values.
- While creating new (Contextual Parser) Rules using dictionary, the uploaded CSV file is validated based on: CSV should not contain any null values, CSV should either be a single row or single column.
- Admin users are now able to remove unused licenses. 

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination owl-carousel pagination_big">
    <li><a href="release_notes_3_2_0">3.2.0</a></li>
    <li><a href="release_notes_3_1_1">3.1.1</a></li>
    <li class="active"><a href="release_notes_3_1_0">3.1.0</a></li>
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