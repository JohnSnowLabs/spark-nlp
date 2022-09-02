---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 2.4.0
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_2_4_0
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

<div class="h3-box" markdown="1">

### 2.4.0

#### Overview

We are glad to announce Spark NLP for Healthcare 2.4.0. This is an important release because of several refactorizations achieved in the core library, plus the introduction of several state of the art algorithms, new features and enhancements.
We have included several architecture and performance improvements, that aim towards making the library more robust in terms of storage handling for Big Data.
In the NLP aspect, we have introduced a ContextualParser, DocumentLogRegClassifier and a ChunkEntityResolverSelector.
These last two Annotators also target performance time and memory consumption by lowering the order of computation and data loaded to memory in each step when designed following a hierarchical pattern.
We have put a big effort on this one, so please enjoy and share your comments. Your words are always welcome through all our different channels.
Thank you very much for your important doubts, bug reports and feedback; they are always welcome and much appreciated.

</div><div class="h3-box" markdown="1">

#### New Features

* BigChunkEntityResolver Annotator: New experimental approach to reduce memory consumption at expense of disk IO.
* ContextualParser Annotator: New entity parser that works based on context parameters defined in a JSON file.
* ChunkEntityResolverSelector Annotator: New AnnotatorModel that takes advantage of the RecursivePipelineModel + LazyAnnotator pattern to annotate with different LazyAnnotators at runtime.
* DocumentLogregClassifier Annotator: New Annotator that provides a wrapped TFIDF Vectorizer + LogReg Classifier for TOKEN AnnotatorTypes (either at Document level or Chunk level)

</div><div class="h3-box" markdown="1">

#### Enhancements

* `normalizedColumn` Param is no longer required in ChunkEntityResolver Annotator (defaults to the `labelCol` Param value).
* ChunkEntityResolverMetadata now has more data to infer whether the match is meaningful or not.

</div><div class="h3-box" markdown="1">

#### Bugfixes

* Fixed a bug on ContextSpellChecker Annotator where unrecognized tokens would cause an exception if not in vocabulary.
* Fixed a bug on ChunkEntityResolver Annotator where undetermined results were coming out of negligible confidence scores for matches.
* Fixed a bug on ChunkEntityResolver Annotator where search would fail if the `neighbours` Param was grater than the number of nodes in the tree. Now it returns up to the number of nodes in the tree.

</div><div class="h3-box" markdown="1">

#### Deprecations

* OCR Moves to its own JSL Spark OCR project.

</div><div class="prev_ver h3-box" markdown="1">

#### Infrastructure

* Spark NLP License is now required to utilize the library. Please follow the instructions on the shared email.

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>
<ul class="pagination">
    <li>
        <strong>Version 2.4.0</strong>
    </li>
    <li>
        <a href="release_notes_2_4_1">Version 2.4.1</a>
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
    <li><a href="release_notes_2_5_5">2.5.5</a></li>
    <li><a href="release_notes_2_5_3">2.5.3</a></li>
    <li><a href="release_notes_2_5_2">2.5.2</a></li>
    <li><a href="release_notes_2_5_0">2.5.0</a></li>
    <li><a href="release_notes_2_4_6">2.4.6</a></li>
    <li><a href="release_notes_2_4_5">2.4.5</a></li>
    <li><a href="release_notes_2_4_2">2.4.2</a></li>
    <li><a href="release_notes_2_4_1">2.4.1</a></li>
    <li class="active"><a href="release_notes_2_4_0">2.4.0</a></li>
</ul>