---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 2.4.2
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_2_4_2
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

<div class="h3-box" markdown="1">

### 2.4.2

#### Overview

We are glad to announce Spark NLP for Healthcare 2.4.2. As a new feature we are happy to introduce our new Disambiguation Annotator,
which will let the users resolve different kind of entities based on Knowledge bases provided in the form of Records in a RocksDB database.
We also enhanced / fixed DocumentLogRegClassifier, ChunkEntityResolverModel and ChunkEntityResolverSelector Annotators.

</div><div class="h3-box" markdown="1">

#### New Features

* Disambiguation Annotator (NerDisambiguator and NerDisambiguatorModel) which accepts annotator types CHUNK and SENTENCE_EMBEDDINGS and
returns DISAMBIGUATION annotator type. This output annotation type includes all the matches in the result and their similarity scores in the metadata.

</div><div class="h3-box" markdown="1">

#### Enhancements

* ChunkEntityResolver Annotator now supports both EUCLIDEAN and COSINE distance for the KNN search and WMD calculation.

</div><div class="h3-box" markdown="1">

#### Bugfixes

* Fixed a bug in DocumentLogRegClassifier Annotator to support its serialization to disk.
* Fixed a bug in ChunkEntityResolverSelector Annotator to group by both SENTENCE and CHUNK at the time of forwarding tokens and embeddings to the lazy annotators.
* Fixed a bug in ChunkEntityResolverModel in which the same exact embeddings was not included in the neighbours.

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>
{%- include docs-healthcare-pagination.html -%}