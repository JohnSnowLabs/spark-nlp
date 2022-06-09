---
layout: docs
header: true
seotitle: Spark NLP
title: Spark NLP release notes 3.0.2
permalink: /docs/en/spark_nlp_versions/release_notes_3_0_2
key: docs-release-notes
modify_date: "2022-01-06"
show_nav: true
sidebar:
    nav: sparknlp
---

### 3.0.2

#### John Snow Labs Spark-NLP 3.0.2: New multilingual models, confidence scores for entities and all NER tags, first support for community models, bug fixes, and other improvements!

Overview

We are glad to release Spark NLP 3.0.2! We have added some new features, improvements, trained some new multi-lingual models, and fixed some bugs in this release.

As always, we would like to thank our community for their feedback, questions, and feature requests.

New Features

* Experimental support for community models and pipelines (uploaded by users) https://github.com/JohnSnowLabs/spark-nlp/pull/2743
* Provide confidence scores for all available tags in NerDLModel and NerCrfModel https://github.com/JohnSnowLabs/spark-nlp/pull/2760

```python
# NerDLModel and NerCrfModel before 3.0.2
[[named_entity, 0, 4, B-LOC, [word -> Japan, confidence -> 0.9998], []]

# Now in Spark NLP 3.0.2
[[named_entity, 0, 4, B-LOC, [B-LOC -> 0.9998, I-ORG -> 0.0, I-MISC -> 0.0, I-LOC -> 0.0, I-PER -> 0.0, B-MISC -> 0.0, B-ORG -> 1.0E-4, word -> Japan, O -> 0.0, B-PER -> 0.0], []]
```
* Calculate confidence score for entities in NerConverter https://github.com/JohnSnowLabs/spark-nlp/pull/2784
```python
[chunk, 30, 41, Barack Obama, [entity -> PERSON, sentence -> 0, chunk -> 0, confidence -> 0.94035]
```

**For more details please check the official [release notes](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/3.0.2)**

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_3_0_1">Version 3.0.1</a>
    </li>
    <li>
        <strong>Version 3.0.2</strong>
    </li>
    <li>
        <a href="release_notes_3_0_3">Version 3.0.3</a>
    </li>
</ul>

<ul class="pagination pagination_big">
  <li><a href="release_notes_3_4_0">3.4.0</a></li>
  <li><a href="release_notes_3_3_4">3.3.4</a></li>
  <li><a href="release_notes_3_3_3">3.3.3</a></li>
  <li><a href="release_notes_3_3_2">3.3.2</a></li>
  <li><a href="release_notes_3_3_1">3.3.1</a></li>
  <li><a href="release_notes_3_3_0">3.3.0</a></li>
  <li><a href="release_notes_3_2_3">3.2.3</a></li>
  <li><a href="release_notes_3_2_2">3.2.2</a></li>
  <li><a href="release_notes_3_2_1">3.2.1</a></li>
  <li><a href="release_notes_3_2_0">3.2.0</a></li>
  <li><a href="release_notes_3_1_3">3.1.3</a></li>
  <li><a href="release_notes_3_1_2">3.1.2</a></li>
  <li><a href="release_notes_3_1_1">3.1.1</a></li>
  <li><a href="release_notes_3_1_0">3.1.0</a></li>
  <li><a href="release_notes_3_0_3">3.0.3</a></li>
  <li class="active"><a href="release_notes_3_0_2">3.0.2</a></li>
  <li><a href="release_notes_3_0_1">3.0.1</a></li>
  <li><a href="release_notes_3_0_0">3.0.0</a></li>
  <li><a href="release_notes_2_7_5">2.7.5</a></li>
  <li><a href="release_notes_2_7_4">2.7.4</a></li>
  <li><a href="release_notes_2_7_3">2.7.3</a></li>
  <li><a href="release_notes_2_7_2">2.7.2</a></li>
  <li><a href="release_notes_2_7_1">2.7.1</a></li>
  <li><a href="release_notes_2_7_0">2.7.0</a></li>
  <li><a href="release_notes_2_6_5">2.6.5</a></li>
  <li><a href="release_notes_2_6_4">2.6.4</a></li>
  <li><a href="release_notes_2_6_3">2.6.3</a></li>
  <li><a href="release_notes_2_6_2">2.6.2</a></li>
  <li><a href="release_notes_2_6_1">2.6.1</a></li>
  <li><a href="release_notes_2_6_0">2.6.0</a></li>
  <li><a href="release_notes_2_5_5">2.5.5</a></li>
  <li><a href="release_notes_2_5_4">2.5.4</a></li>
  <li><a href="release_notes_2_5_3">2.5.3</a></li>
  <li><a href="release_notes_2_5_2">2.5.2</a></li>
  <li><a href="release_notes_2_5_1">2.5.1</a></li>
  <li><a href="release_notes_2_5_0">2.5.0</a></li>
  <li><a href="release_notes_2_4_5">2.4.5</a></li>
  <li><a href="release_notes_2_4_4">2.4.4</a></li>
  <li><a href="release_notes_2_4_3">2.4.3</a></li>
  <li><a href="release_notes_2_4_2">2.4.2</a></li>
  <li><a href="release_notes_2_4_1">2.4.1</a></li>
  <li><a href="release_notes_2_4_0">2.4.0</a></li>
</ul>