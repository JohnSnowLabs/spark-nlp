---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: Infrastructure Configuration
permalink: /docs/en/alab/infrastructure
key: docs-training
modify_date: "2022-11-28"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
  nav: annotation-lab
---

The _admin_ user can now define the infrastructure configurations for the prediction and training tasks.

### Resource allocation for Training and Preannotation

Annotation Lab gives users the ability to change the configuration for the training and pre-annotation processes. This is done from the `Settings` > `Infrastructure` page. The settings can be edited by _admin_ user and they are read-only for the other users. The Infrastructure page consists of three sections namely `Resource For Training`, `Resource For Preannotation Server`, `Resources for Prenotation Pipeline`.

**Resources Inclusion:**

1.  Memory Limit – Represents the maximum memory size to allocate for the training/pre-annotation processes.
2.  CPU Limit – Specifies this maximum number of CPUs to use by the training/pre-annotation server.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.3.0/infrastructure.png" style="width:100%;"/>

> **Note:** If the specified configurations exceed the available resources, the server will not start.
