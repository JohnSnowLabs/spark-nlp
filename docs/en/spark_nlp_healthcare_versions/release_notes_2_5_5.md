---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 2.5.5
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_2_5_5
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

<div class="h3-box" markdown="1">

### 2.5.5

#### Overview

We are very happy to release Spark NLP for Healthcare 2.5.5 with a new state-of-the-art RelationExtraction annotator to identify relationships between entities coming from our pretrained NER models.
This is also the first release to support Relation Extraction with the following two (2) models: `re_clinical` and `re_posology` in the `clinical/models` repository.
We also include multiple bug fixes as usual.

</div><div class="h3-box" markdown="1">

#### New Features

* RelationExtraction annotator that receives `WORD_EMBEDDINGS`, `POS`, `CHUNK`, `DEPENDENCY` and returns the CATEGORY of the relationship and a confidence score.

</div><div class="h3-box" markdown="1">

#### Enhancements

* AssertionDL Annotator now keeps logs of the metrics while training
* DeIdentification now has a default behavior of merging entities close in Levenshtein distance with `setConsistentObfuscation` and `setSameEntityThreshold` params.
* DeIdentification now has a specific parameter `setObfuscateDate` to obfuscate dates (which will be otherwise just masked). The only formats obfuscated when the param is true will be the ones present in `dateFormats` param.
* NerConverterInternal now has a `greedyMode` param that will merge all contiguous tags of the same type regardless of boundary tags like "B","E","S".
* AnnotationToolJsonReader includes `mergeOverlapping` parameter to merge (or not) overlapping entities from the Annotator jsons i.e. not included in the assertion list.

</div><div class="h3-box" markdown="1">

#### Bugfixes

* DeIdentification documentation bug fix (typo)
* DeIdentification training bug fix in obfuscation dictionary
* IOBTagger now has the correct output type `NAMED_ENTITY`

</div><div class="h3-box" markdown="1">

#### Deprecations

* EnsembleEntityResolver has been deprecated

Models

* We have 2 new `english` Relationship Extraction model for Clinical and Posology NERs:
   - `re_clinical`: with `ner_clinical` and `embeddings_clinical`
   - `re_posology`: with `ner_posology` and `embeddings_clinical`

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_2_5_3">Version 2.5.3</a>
    </li>
    <li>
        <strong>Version 2.5.5</strong>
    </li>
    <li>
        <a href="release_notes_2_6_0">Version 2.6.0</a>
    </li>
</ul>

<ul class="pagination owl-carousel pagination_big">
    <li><a href="release_notes_4_1_0">4.1.0</a></li>
    <li><a href="release_notes_4_0_2">4.0.2</a></li>
    <li><a href="release_notes_4_0_0">4.0.0</a></li>
    <li><a href="release_notes_3_5_3">3.5.3</a></li>
    <li><a href="release_notes_3_5_2">3.5.2</a></li>
    <li><a href="release_notes_3_5_1">3.5.1</a></li>
    <li><a href="release_notes_3_5_0">3.5.0</a></li>
    <li><a href="release_notes_3_4_2">3.4.2</a></li>
    <li><a href="release_notes_3_4_1">3.4.1</a></li>
    <li><a href="release_notes_3_4_0">3.4.0</a></li>
    <li><a href="release_notes_3_3_4">3.3.4</a></li>
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
    <li><a href="release_notes_3_0_2">3.0.2</a></li>
    <li><a href="release_notes_3_0_1">3.0.1</a></li>
    <li><a href="release_notes_3_0_0">3.0.0</a></li>
    <li><a href="release_notes_2_7_6">2.7.6</a></li>
    <li><a href="release_notes_2_7_5">2.7.5</a></li>
    <li><a href="release_notes_2_7_4">2.7.4</a></li>
    <li><a href="release_notes_2_7_3">2.7.3</a></li>
    <li><a href="release_notes_2_7_2">2.7.2</a></li>
    <li><a href="release_notes_2_7_1">2.7.1</a></li>
    <li><a href="release_notes_2_7_0">2.7.0</a></li>
    <li><a href="release_notes_2_6_2">2.6.2</a></li>
    <li><a href="release_notes_2_6_0">2.6.0</a></li>
    <li class="active"><a href="release_notes_2_5_5">2.5.5</a></li>
    <li><a href="release_notes_2_5_3">2.5.3</a></li>
    <li><a href="release_notes_2_5_2">2.5.2</a></li>
    <li><a href="release_notes_2_5_0">2.5.0</a></li>
    <li><a href="release_notes_2_4_6">2.4.6</a></li>
    <li><a href="release_notes_2_4_5">2.4.5</a></li>
    <li><a href="release_notes_2_4_2">2.4.2</a></li>
    <li><a href="release_notes_2_4_1">2.4.1</a></li>
    <li><a href="release_notes_2_4_0">2.4.0</a></li>
</ul>