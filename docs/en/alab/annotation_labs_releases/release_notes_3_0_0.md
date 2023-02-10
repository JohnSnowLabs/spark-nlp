---
layout: docs
header: true
seotitle: Annotation Lab | John Snow Labs
title: Annotation Lab Release Notes 3.0.0
permalink: /docs/en/alab/annotation_labs_releases/release_notes_3_0_0
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: annotation-lab
---

<div class="h3-box" markdown="1">

## 3.0.0

Release date: **06-04-2022**

We are very excited to release Annotation Lab 3.0.0 with support for Floating Licenses and for parallel training and preannotation jobs, created on demand by Project Owners and Managers across various projects. Below are more details about the release.

### Highlights

- Annotation Lab now supports [floating licenses](/docs/en/alab/byol#support-for-floating-licenses) with different scopes (ocr: training, ocr: inference, healthcare: inference, healthcare: training). Depending on the scope of the available license, users can perform model training and/or deploy preannotation servers. Licenses are a must only for training Spark NLP for Healthcare models and for deploying Spark NLP for Healthcare models as preannotation servers.
- [Parallel Trainings](/docs/en/alab/active_learning#deploy-a-new-training-job) and [Preannotations](/docs/en/alab/preannotations#start-preannotation). Annotation Lab now offers support for running model training and document preannotation across multiple projects and/or teams in parallel. If the infrastructure dedicated to the Annotation Lab includes sufficient resources, each team/project can run smoothly without being blocked.
- On demand deployment of preannotation servers and training jobs:
  - [Deploy a new training job](/docs/en/alab/active_learning#deploy-a-new-training-job)
  - [Deploy a new preannotation server](/docs/en/alab/preannotations#start-preannotation)
  - [OCR and Visual NER servers](/docs/en/alab/visual_ner#ocr-and-visual-ner-servers)
- The infrastucture page now hosts a new tab for managing [preannotation, training and OCR servers.](/docs/en/alab/infrastructure#management-of-preannotation-and-training-servers)
- New options available on [preannotate](/docs/en/alab/preannotations#start-preannotation) action. 
- Updates for the [license page](/docs/en/alab/byol#license-page).

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>

{%- include docs-annotation-pagination.html -%}