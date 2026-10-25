---
layout: model
title: MARS Token Importance
author: John Snow Labs
name: mars_token_importance
date: 2026-08-10
tags: [en, open_source, bert, mars, token_importance]
task: Named Entity Recognition
language: en
edition: Spark NLP 7.0.0
spark_version: 3.4
supported: true
annotator: MarsTokenImportance
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

A BERT-base-uncased token classification model that scores how much each token in a piece of text actually matters to its meaning, rather than treating every token equally. Given a question and an answer, it assigns an importance weight to each token in the answer - so a content word like "Paris" gets weighted far more heavily than a filler word like "the" or "is".

This is MARS (Bakman et al., via TruthTorchLM): the intended use is token-importance-weighted confidence scoring, where a model's uncertainty about the parts of an answer that carry meaning should count more than its uncertainty about grammatical filler. In Spark NLP, load it with MarsTokenImportance, which attaches per-token importance scores as metadata for downstream use.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mars_token_importance_en_7.0.0_3.4_1786376403960.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mars_token_importance_en_7.0.0_3.4_1786376403960.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
mars = MarsTokenImportance.pretrained("mars_token_importance", "en").setInputCols(["question", "completions"]).setOutputCol("completions_with_mars")
```
```scala
val mars = MarsTokenImportance.pretrained("mars_token_importance", "en").setInputCols(Array("question", "completions")).setOutputCol("completions_with_mars")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mars_token_importance|
|Compatibility:|Spark NLP 7.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[question, completions]|
|Output Labels:|[completions_with_mars]|
|Language:|en|
|Size:|407.2 MB|