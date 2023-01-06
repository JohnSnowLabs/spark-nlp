---
layout: docs
header: true
seotitle: Annotation Lab | John Snow Labs
title: Annotation Lab Release Notes 4.4.0
permalink: /docs/en/alab/annotation_labs_releases/release_notes_4_4_0
key: docs-licensed-release-notes
modify_date: 2022-12-12
show_nav: true
sidebar:
  nav: annotation-lab
---

<div class="h3-box" markdown="1">

## 4.4.0

Release date: **05-12-2022**

Annotation Lab 4.4.0 brings performance matrix and benchmarking information for NER and classification models - both imported from NLP Models Hub and/or trained locally. Furthermore, with this release, tasks can be explicitly assigned to Project Owners for annotators and/or reviewers. The release also includes several improvements and fixes for issues reported by our users.

Here are the highlights of this release:

### Highlights

- **Benchmarking information for Classification models**. Benchmarking information is now available for Classification models. It includes the confusion matrix in the training logs and is also available on the models on the Models page, which is accessible by clicking on the benchmarking icon.
- **Task Assignment for Project Owners**. Project Owners can now be explicitly assigned as annotators and/or reviewers for tasks. It is useful when working in a small team and when the Project Owners are also involved in the annotation process. A new option "Only Assigned" checkbox is now available on the labeling page that allows Project Owners to filter the tasks explicitly assigned to them when clicking the Next button.
- **New Role available on the Team Page**. On the Team Setup page, the project creator is now clearly identified by the "Owner" role.
- **Rules Available in the Finance and Legal Editions**. Rules can now be deployed and used for pre-annotation using the Legal and Finance licenses.
- **UX Improvement for Completion**. The action icons are now available on the completions, and users can directly execute the appropriate action without having to select the completion first.
- **IAA chart improvements**. NER labels and Assertion Status labels are now handled separately in the IAA charts on the Analytics page. The filter for selecting the label type is added on the respective charts.
- **Import tasks with title field**. Users can now import the tasks with title information pre-defined in the JSON/CSV. The title field was also added to the sample task file that can be downloaded from the Import page.
- **Rename Models Hub page**. The name Models HUB on the left navigation panel has been changed to Hub.

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>

{%- include docs-annotation-pagination.html -%}
