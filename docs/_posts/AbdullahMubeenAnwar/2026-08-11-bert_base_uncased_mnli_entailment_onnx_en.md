---
layout: model
title: BERT for Natural Language Inference (MNLI)
author: John Snow Labs
name: bert_base_uncased_mnli_entailment_onnx
date: 2026-08-11
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_uncased_mnli_entailment_onnx_en_7.0.0_3.4_1786426318793.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_base_uncased_mnli_entailment_onnx_en_7.0.0_3.4_1786426318793.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

document_assembler = DocumentAssembler() \
    .setInputCol("completions") \
    .setOutputCol("document")

entailment = SampleEntailmentMatrix.pretrained("bert_base_uncased_mnli_entailment_onnx", "en") \
    .setInputCols(["document"]) \
    .setOutputCol("completions_with_entailment")

nlp_pipeline = Pipeline(stages=[
    document_assembler,
    entailment
])

data = spark.createDataFrame([[[
    "The capital of France is Paris.",
    "Paris is France's capital.",
    "The capital of France is London."
]]]).toDF("completions")

result = nlp_pipeline.fit(data).transform(data)
result.selectExpr("completions_with_entailment[0].metadata['entailment_matrix'] as entailment_matrix").show(truncate=False)
```
```scala
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("completions")
  .setOutputCol("document")

val entailment = SampleEntailmentMatrix.pretrained("bert_base_uncased_mnli_entailment_onnx", "en")
  .setInputCols("document")
  .setOutputCol("completions_with_entailment")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  entailment
))

val data = Seq(Seq(
  "The capital of France is Paris.",
  "Paris is France's capital.",
  "The capital of France is London."
)).toDF("completions")

val result = pipeline.fit(data).transform(data)
result.selectExpr("completions_with_entailment[0].metadata['entailment_matrix'] as entailment_matrix").show(false)
```
</div>

## Results

```bash

+----------------------------------------------------------------------------------------------------------------------------------------------+
|entailment_matrix                                                                                                                             |
+----------------------------------------------------------------------------------------------------------------------------------------------+
|[[1.0,0.9935469627380371,3.128613461740315E-4],[0.9925073981285095,1.0,5.720576737076044E-4],[4.627297748811543E-4,0.0011771831195801497,1.0]]|
+----------------------------------------------------------------------------------------------------------------------------------------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_base_uncased_mnli_entailment_onnx|
|Compatibility:|Spark NLP 7.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[completions]|
|Output Labels:|[completions_with_entailment]|
|Language:|en|
|Size:|409.4 MB|
|Case sensitive:|false|