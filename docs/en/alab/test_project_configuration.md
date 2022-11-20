---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: Test Project Configuration
permalink: /docs/en/alab/test_project_configuration
key: docs-training
modify_date: "2022-11-20"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
  nav: annotation-lab
---

A **Project Owner** or a **Manager** can use the completed tasks (completions) from a project to test the pre-trained model. The project "Test Configuration" feature can be found on the train page, accessible from the Project Menu. During the training, a progress bar is shown on the top of the train page to show the status of the 

<img class="image image__shadow" src="/assets/images/annotation_lab/4.2.0/test.png" style="width:100%;"/>

To evaluate a project, the project's configuration need to contain pre-defined labels. After the user mark the tasks with "Test" tag, "Test Configuration" can be executed. This will evaluate annotated tasks tagged as "Test" against the configured pre-trained NER models. After the evaluaton is completed, the resultant logs can be downloaded to view the performance metrics.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.2.0/testSteps.gif" style="width:100%;"/>

> **Note:** Project evaluation can only be triggered if the deployment has a valid healthcare air-gapped license. 

