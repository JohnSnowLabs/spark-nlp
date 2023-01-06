---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 2.4.1
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_2_4_1
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

<div class="h3-box" markdown="1">

### 2.4.1

#### Overview

Introducing Spark NLP for Healthcare 2.4.1 after all the feedback we received in the form of issues and suggestions on our different communication channels.
Even though 2.4.0 was very stable, version 2.4.1 is here to address minor bug fixes that we summarize in the following lines.

</div><div class="h3-box" markdown="1">

#### Bugfixes

* Changing the license Spark property key to be "jsl" instead of "sparkjsl" as the latter generates inconsistencies
* Fix the alignment logic for tokens and chunks in the ChunkEntityResolverSelector because when tokens and chunks did not have the same begin-end indexes the resolution was not executed

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>
{%- include docs-healthcare-pagination.html -%}