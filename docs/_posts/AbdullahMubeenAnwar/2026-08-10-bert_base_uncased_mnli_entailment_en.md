---
layout: model
title: BERT for Natural Language Inference (MNLI)
author: John Snow Labs
name: bert_base_uncased_mnli_entailment
date: 2026-08-10
tags: [en, open_source, bert, mnli, nli, entailment, sample_entailment_matrix]
task: Text Classification
language: en
edition: Spark NLP 7.0.0
spark_version: 3.4
supported: true
annotator: SampleEntailmentMatrix
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

A BERT-base-uncased checkpoint fine-tuned for natural language inference on the MultiNLI (MNLI) corpus. Given a premise and a hypothesis, it predicts whether the hypothesis is entailed by the premise, neutral toward it, or contradicts it.

In Spark NLP, load it with SampleEntailmentMatrix to compute bidirectional entailment between pairs of texts - two texts are treated as equivalent when each entails the other. This is the textbook approach to deciding whether two pieces of text "mean the same thing," independent of exact wording, and is useful anywhere you need to cluster or deduplicate text by meaning rather than surface form.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_uncased_mnli_entailment_en_7.0.0_3.4_1786376157041.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_base_uncased_mnli_entailment_en_7.0.0_3.4_1786376157041.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
entailment = SampleEntailmentMatrix.pretrained("bert_base_uncased_mnli_entailment", "en").setInputCols("completions").setOutputCol("completions_with_entailment")
```
```scala
val entailment = SampleEntailmentMatrix.pretrained("bert_base_uncased_mnli_entailment", "en").setInputCols("completions").setOutputCol("completions_with_entailment")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_base_uncased_mnli_entailment|
|Compatibility:|Spark NLP 7.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[completions]|
|Output Labels:|[completions_with_entailment]|
|Language:|en|
|Size:|409.4 MB|