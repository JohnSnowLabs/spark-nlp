---
layout: model
title: Question Pair Classifier Pipeline
author: John Snow Labs
name: classifierdl_electra_questionpair_pipeline
date: 2021-08-25
tags: [quora, question_pair, public, en, open_source]
task: Text Classification
language: en
edition: Spark NLP 3.2.0
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pre-trained pipeline identifies whether the two question sentences are semantically repetitive or different.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_electra_questionpair_pipeline_en_3.2.0_2.4_1629892687975.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

The sample code snippet may not contain all required fields of a pipeline. In this case, you can reach out a related colab notebook containing the end-to-end pipeline and more by clicking the "Open in Colab" link above.


- The question pairs should be identified with "q1" and "q2" in the text. The input text format should be as follows : `text = "q1: What is your name? q2: Who are you?"`.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
python pipeline = PretrainedPipeline("classifierdl_electra_questionpair_pipeline", "en")

result1 = pipeline.fullAnnotate("q1: What is your favorite movie? q2: Which movie do you like most?")
result2 = pipeline.fullAnnotate("q1: What is your favorite movie? q2: Which movie genre would you like to watch?")
```
```scala
val pipeline = new PretrainedPipeline("classifierdl_electra_questionpair_pipeline", "en")

val result1 = pipeline.fullAnnotate("q1: What is your favorite movie? q2: Which movie do you like most?")(0)
val result2 = pipeline.fullAnnotate("q1: What is your favorite movie? q2: Which movie genre would you like to watch?")(0)
```
</div>

## Results

```bash
['almost_same']
['not_same']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_electra_questionpair_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.2.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|

## Included Models

- DocumentAssembler
- BertSentenceEmbeddings
- ClassifierDLModel