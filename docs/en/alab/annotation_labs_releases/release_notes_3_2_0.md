---
layout: docs
header: true
seotitle: Annotation Lab | John Snow Labs
title: Annotation Lab Release Notes 3.2.0
permalink: /docs/en/alab/annotation_labs_releases/release_notes_3_2_0
key: docs-licensed-release-notes
modify_date: 2022-05-31
show_nav: true
sidebar:
    nav: annotation-lab
---

<div class="h3-box" markdown="1">

## 3.2.0

Release date: **31-05-2022**

We are very excited to announce the release of Annotation Lab v3.2.0 which includes new and exciting features such as Project cloning and Project backup, Evaluation of Pretrained Models, and Search feature in the Visual NER Project. Support for Multiple files import, ability to view statuses of Model Servers and Training Jobs, and prioritization of completions for CONLL export. Spark NLP and Spark OCR libraries were also upgraded, and some security fixes and stabilizations were also implemented. Here are the highlights:

### Highlights

- Import/export of an entire Project. All project-related items (tasks, project configuration, project members, task assignments) can be imported/exported. In addition, users can also clone an existing project.
- Evaluate Named Entity Models. Project Owner and/or Manager can now test and evaluate annotated tasks against the Pretrained NER models in the Training & Active Learning Settings tab, configured NER models will be tested against the tasks tagged as test.
- Statuses of Training and Preannotation Server. A new column, status, is added to the server page that gives the status of training and preannotation servers. Also if any issues are encountered during server initialization, those are displayed on mouse-over the status value.
- Import Multiple Files. Project Owners or Managers can now upload multiple files at once in bulk.
- Prioritize Annotators For Data Export. When multiple completions are available for the same task, the CONLL export will include completions from higher priority members.
- Network Policies have been implemented which specify how a pod is allowed to communicate with various network "entities" over the network. The entities that are required to function in Annotation Lab were clearly identified and only traffic coming from them is now allowed.
- Support for airgap licenses with scope. Previously airgap licenses with scopes were missrecognized as floating licenses.
- Upgraded Spark NLP and Spark NLP for Health Care v3.4.1 and Spark OCR v3.12.0 

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>

{%- include docs-annotation-pagination.html -%}