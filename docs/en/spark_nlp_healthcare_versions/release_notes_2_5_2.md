---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 2.5.2
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_2_5_2
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

<div class="h3-box" markdown="1">

### 2.5.2

#### Overview

We are really happy to bring you Spark NLP for Healthcare 2.5.2, with a couple new features and several enhancements in our existing annotators.
This release was mainly dedicated to generate adoption in our AnnotationToolJsonReader, a connector that provide out-of-the-box support for out Annotation Tool and our practices.
Also the ChunkMerge annotator has ben provided with extra functionality to remove entire entity types and to modify some chunk's entity type
We also dedicated some time in finalizing some refactorization in DeIdentification annotator, mainly improving type consistency and case insensitive entity dictionary for obfuscation.
Thanks to the community for all the feedback and suggestions, it's really comfortable to navigate together towards common functional goals that keep us agile in the SotA.

</div><div class="h3-box" markdown="1">

#### New Features

* Brand new IOBTagger Annotator
* NerDL Metrics provides an intuitive DataFrame API to calculate NER metrics at tag (token) and entity (chunk) level

</div><div class="h3-box" markdown="1">

#### Enhancements

* AnnotationToolJsonReader includes parameters for document cleanup, sentence boundaries and tokenizer split chars
* AnnotationToolJsonReader uses the task title if present and uses IOBTagger annotator
* AnnotationToolJsonReader has improved alignment in assertion train set generation by using an `alignTol` parameter as tollerance in chunk char alignment
* DeIdentification refactorization: Improved typing and replacement logic, case insensitive entities for obfuscation
* ChunkMerge Annotator now handles:
 - Drop all chunks for an entity
 - Replace entity name
 - Change entity type for a specific (chunk, entity) pair
 - Drop specific (chunk, entity) pairs
* `caseSensitive` param to EnsembleEntityResolver
* Output logs for AssertionDLApproach loss
* Disambiguator is back with improved dependency management

</div><div class="h3-box" markdown="1">

#### Bugfixes

* Bugfix in python when Annotators shared domain parts across public and internal
* Bugfix in python when ChunkMerge annotator was loaded from disk
* ChunkMerge now weights the token coverage correctly when multiple multi-token entities overlap

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_2_5_0">Version 2.5.0</a>
    </li>
    <li>
        <strong>Version 2.5.2</strong>
    </li>
    <li>
        <a href="release_notes_2_5_3">Version 2.5.3</a>
    </li>
</ul>

<ul class="pagination owl-carousel pagination_big">
    <li><a href="release_notes_4_2_0">4.2.0</a></li>
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
    <li><a href="release_notes_2_5_5">2.5.5</a></li>
    <li><a href="release_notes_2_5_3">2.5.3</a></li>
    <li class="active"><a href="release_notes_2_5_2">2.5.2</a></li>
    <li><a href="release_notes_2_5_0">2.5.0</a></li>
    <li><a href="release_notes_2_4_6">2.4.6</a></li>
    <li><a href="release_notes_2_4_5">2.4.5</a></li>
    <li><a href="release_notes_2_4_2">2.4.2</a></li>
    <li><a href="release_notes_2_4_1">2.4.1</a></li>
    <li><a href="release_notes_2_4_0">2.4.0</a></li>
</ul>