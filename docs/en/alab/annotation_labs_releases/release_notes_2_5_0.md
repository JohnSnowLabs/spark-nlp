---
layout: docs
header: true
seotitle: Annotation Lab | John Snow Labs
title: Annotation Lab Release Notes 2.5.0
permalink: /docs/en/alab/annotation_labs_releases/release_notes_2_5_0
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: annotation-lab
---

<div class="h3-box" markdown="1">

## 2.5.0

Annotation Lab v2.5.0 introduces support for rule based annotations, new search feature and COCO format export for Visual NER projects. It also includes fixes for the recently identified security issues and other known bugs. Below are the highlights of this release.

### Highlights
- Rule Based Annotations. Spark NLP for Healthcare supports rule-based annotations via the ContextualParser Annotator. In this release Annotationlab adds support for creating and using ContextualParser rules in NER project. Any user with admin privilegis can see rules under the `Available Rules` tab on the Models Hub page and can create new rules using the `+ Add Rule` button. After adding a rule on `Models Hub` page, the `Project Owner` or `Manager` can add the rule to the configuration of the project where he wants to use it. This can be done via the Rules tab from the `Project Setup` page under the `Project Configuration` tab. A valid Spark NLP for Healthcare licence is required to deploy rules from project config. Two types of rules are supported:1. `Regex Based`: User can enter the Regex which matches to the entities of the required label; and 2. `Dictionary Based`: User can create a dictionary of labels and user can upload the CSV of the list of entity that comes under the label.
- Search through Visual NER Projects. For the Visual NER Projects, it is now possible to search for a keyword inside of image/pdf based tasks using the search box available on the top of the `Labeling` page. Currently, the search is performed on the current page only. 
Furthermore, we have also extended the keyword-based task search already available for text-based projects for Visual NER Projects. On the `Tasks` page, use the search bar on the upper right side of the screen like you would do in other text-based projects, to identify all image/pdf tasks containing a given text.
- COCO export for pdf tasks in `Visual NER Projects`. Up until now, the COCO format export was limited to simple image documents. With version 2.5.0, this functionality is extended to single-page or multi-page pdf documents. 
- In **Classification Project**, users are now able to use different layouts for the list of choices:
   - **`layout="select"`**: It will change choices from list of choices inline to dropdown layout. Possible values are `"select", "inline", "vertical"`
   - **`choice="multiple"`**: Allow user to select multiple values from dropdown. Possible values are: `"single", "single-radio", "multiple"`
- Better Toasts, Confirmation-Boxes and Masking UI on potentially longer operations.

### Security Fixes
- Annotationlab v2.5.0 got different Common Vulnerabilities and Exposures(CVE) issues fixed. As always, in this release we performed security scans to detect CVE issues, upgraded python packages to eliminate known vulnerabilities and also we made sure the CVE-2021-44228 (Log4j2 issue) is not present in any images used by Annotation Lab.
- A reported issue when logout endpoint was sometimes redirected to insecure http after access token expired was also fixed.

### Bug Fixes

- The `Filters` option in the `Models Hub` page was not working properly. Now the "Free/Licensed" filter can be selected/deselected without getting any error.
- After creating relations and saving/updating annotations for the Visual NER projects with multi-paged pdf files, the annotations and relations were not saved. 
- An issue with missing text tokens in the exported JSON file for the Visual NER projects also have been fixed.

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
    <li class="active"><a href="release_notes_2_5_0">2.5.0</a></li>
    <li><a href="release_notes_2_4_0">2.4.0</a></li>
    <li><a href="release_notes_2_3_0">2.3.0</a></li>
    <li><a href="release_notes_2_2_2">2.2.2</a></li>
    <li><a href="release_notes_2_1_0">2.1.0</a></li>
    <li><a href="release_notes_2_0_1">2.0.1</a></li>
</ul>