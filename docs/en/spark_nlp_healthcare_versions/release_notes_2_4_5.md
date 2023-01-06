---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 2.4.5
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_2_4_5
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

<div class="h3-box" markdown="1">

### 2.4.5

#### Overview

We are glad to announce Spark NLP for Healthcare 2.4.5. As a new feature we are happy to introduce our new EnsembleEntityResolver which allows our Entity Resolution architecture to scale up in multiple orders of magnitude and handle datasets of millions of records on a sub-log computation increase
We also enhanced our ChunkEntityResolverModel with 5 new distance calculations with weighting-array and aggregation-strategy params that results in more levers to finetune its performance against a given dataset.

</div><div class="h3-box" markdown="1">

#### New Features

* EnsembleEntityResolver consisting of an integrated TFIDF-Logreg classifier in the first layer + Multiple ChunkEntityResolvers in the second layer (one per each class)
* Five (5) new distances calculations for ChunkEntityResolver, namely:
    - Token Based: TFIDF-Cosine, Jaccard, SorensenDice
    - Character Based: JaroWinkler and Levenshtein
* Weight parameter that works as a multiplier for each distance result to be considered during their aggregation
* Three (3) aggregation strategies for the enabled distance in a particular instance, namely: AVERAGE, MAX and MIN

</div><div class="h3-box" markdown="1">

#### Enhancements

* ChunkEntityResolver can now compute distances over all the `neighbours` found and return the metadata just for the best `alternatives` that meet the `threshold`;
before it would calculate them over the neighbours and return them all in the metadata
* ChunkEntityResolver now has an `extramassPenalty` parameter to accoun for penalization of token-length difference in compared strings
* Metadata for the ChunkEntityResolver has been updated accordingly to reflect all new features
* StringDistances class has been included in utils to aid in the calculation and organization of different types of distances for Strings
* HasFeaturesJsl trait has been included to support the serialization of Features including [T] <: AnnotatorModel[T] types

</div><div class="h3-box" markdown="1">

#### Bugfixes

* Frequency calculation for WMD in ChunkEntityResolver has been adjusted to account for real word count representation
* AnnotatorType for DocumentLogRegClassifier has been changed to CATEGORY to align with classifiers in Open Source library

</div><div class="h3-box" markdown="1">

#### Deprecations

* Legacy EntityResolver{Approach, Model} classes have been deprecated in favor of ChunkEntityResolver classes
* ChunkEntityResolverSelector classes has been deprecated in favor of EnsembleEntityResolver

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>
{%- include docs-healthcare-pagination.html -%}