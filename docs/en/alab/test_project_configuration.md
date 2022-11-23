---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: Test Project Configuration
permalink: /docs/en/alab/test_project_configuration
key: docs-training
modify_date: "2022-11-23"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
  nav: annotation-lab
---

Annotation Lab offer testing features for projects that reuse existing models/rules. In other words, if a  project's configuration references one or several (pre)trained models/rules it is possible to check how efficient those are when applied on custom data. 
The `Test Configuration` feature is available on the `Train` page, accessible from the Project Menu. During the training, a progress bar is shown on the top of the train page to show the status of the testing.  

<img class="image image__shadow" src="/assets/images/annotation_lab/4.2.0/test.png" style="width:100%;"/>

>**Note** This feature is available for **Project Owners** or **Managers**. 

{:.info}
The `Test Configuration` feature applies to project tasks with status **submitted** or **reviewed**, and which are tagged as **Test**. 

After the evaluaton is completed, the resulting logs can be downloaded to view the performance metrics.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.2.0/testSteps.gif" style="width:100%;"/>

> **Note:** Model evaluation can only be triggered in the presence of a valid Healthcare, Finance or/and Legal NLP license. 

