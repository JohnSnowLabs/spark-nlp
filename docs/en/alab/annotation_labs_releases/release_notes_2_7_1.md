---
layout: docs
header: true
seotitle: Annotation Lab | John Snow Labs
title: Annotation Lab Release Notes 2.7.1
permalink: /docs/en/alab/annotation_labs_releases/release_notes_2_7_1
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: annotation-lab
---

<div class="h3-box" markdown="1">

## 2.7.1

Release date: **22-02-2022**
 
Annotation Lab v2.7.1 introduces an upgrade to K3s v1.22.4 and support for Redhat. It also includes improvements and fixes for identified bug. Below are the highlights of this release. 

### Highlights 
- For new installations, Annotation Lab is now installed on top of K3s v1.22.4. In near future we will provide similar support for existing installations. AWS market place also runs using the upgraded version. 
- With this release Annotation lab can be installed on RedHat servers. 
- Annotation lab 2.7.1 included release version of Spark NLP 3.4.1 and Spark NLP for Healthcare 

### Bug Fixes 
- In the previous release, saving Visual NER project configuration took a long time. With this release, the issue has been fixed and the Visual NER project can be created instantly. 
- Due to a bug in Relation Constraint, all the relations we visible when the UI was refreshed. This issue has been resolved and only a valid list of relations is shown after the UI is refreshed. 
- Previously, labels with spaces at the end were considered different. This has been fixed such that the label name with or without space at the end is treated as the same label. 
- Importing multiple images as a zip file was not working correctly in the case of Visual NER. This  issue was fixed.
- This version also fixes issues in Transfer Learning/Fine Tuning some NER models. 

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>

{%- include docs-annotation-pagination.html -%}