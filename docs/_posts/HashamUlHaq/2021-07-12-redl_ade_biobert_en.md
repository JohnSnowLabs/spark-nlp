---
layout: model
title: Relation extraction between Drugs and ADE
author: John Snow Labs
name: redl_ade_biobert
date: 2021-07-12
tags: [licensed, relation_extraction, clinical, en]
task: Relation Extraction
language: en
edition: Spark NLP for Healthcare 3.1.2
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is an end-to-end trained BioBERT model, capable of Relating Drugs and adverse reactions caused by them; It predicts if an adverse event is caused by a drug or not.

## Predicted Entities

`1` : Shows the adverse event and drug entities are related.

 `0` : Shows the adverse event and drug entities are not related.

{:.btn-box}
[Live Demo](https://nlp.johnsnowlabs.com/demo){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/redl_ade_biobert_en_3.1.2_3.0_1626103580615.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
words_embedder = WordEmbeddingsModel() \
    .pretrained("embeddings_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens"]) \
    .setOutputCol("embeddings")

ner_tagger = NerDLModel() \
    .pretrained("ner_ade_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens", "embeddings"]) \
    .setOutputCol("ner_tags")

ner_converter = NerConverter() \
    .setInputCols(["sentences", "tokens", "ner_tags"]) \
    .setOutputCol("ner_chunks")

dependency_parser = DependencyParserModel() \
    .pretrained("dependency_conllu", "en") \
    .setInputCols(["sentences", "pos_tags", "tokens"]) \
    .setOutputCol("dependencies")

# Set a filter on pairs of named entities which will be treated as relation candidates
re_ner_chunk_filter = RENerChunksFilter() \
    .setInputCols(["ner_chunks", "dependencies"])\
    .setMaxSyntacticDistance(10)\
    .setOutputCol("re_ner_chunks").setRelationPairs(['ade-drug', 'drug-ade'])

# The dataset this model is trained to is sentence-wise. 
# This model can also be trained on document-level relations - in which case, while predicting, use "document" instead of "sentence" as input.
re_model = RelationExtractionDLModel()\
    .pretrained('redl_ade_biobert', 'en', "clinical/models") \
    .setPredictionThreshold(0.5)\
    .setInputCols(["re_ner_chunks", "sentences"]) \
    .setOutputCol("relations")

pipeline = Pipeline(stages=[documenter, sentencer, tokenizer, pos_tagger, words_embedder, ner_tagger, ner_converter, dependency_parser, re_ner_chunk_filter, re_model])

text ="""A 44-year-old man taking naproxen for chronic low back pain and a 20-year-old woman on oxaprozin for rheumatoid arthritis presented with tense bullae and cutaneous fragility on the face and the back of the hands."""

p_model = pipeline.fit(spark.createDataFrame([[text]]).toDF("text"))
result = p_model.transform(data)
```
```scala
...
val words_embedder = WordEmbeddingsModel()
    .pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens"))
    .setOutputCol("embeddings")

val ner_tagger = NerDLModel()
    .pretrained("ner_ade_clinical", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens", "embeddings"))
    .setOutputCol("ner_tags")

val ner_converter = NerConverter()
    .setInputCols(Array("sentences", "tokens", "ner_tags"))
    .setOutputCol("ner_chunks")

val dependency_parser = DependencyParserModel()
    .pretrained("dependency_conllu", "en")
    .setInputCols(Array("sentences", "pos_tags", "tokens"))
    .setOutputCol("dependencies")

// Set a filter on pairs of named entities which will be treated as relation candidates
val re_ner_chunk_filter = RENerChunksFilter()
    .setInputCols(Array("ner_chunks", "dependencies"))
    .setMaxSyntacticDistance(10)
    .setOutputCol("re_ner_chunks").setRelationPairs(Array("drug-ade", "ade-drug"))

// The dataset this model is trained to is sentence-wise. 
// This model can also be trained on document-level relations - in which case, while predicting, use "document" instead of "sentence" as input.
val re_model = RelationExtractionDLModel()
    .pretrained("redl_ade_biobert", "en", "clinical/models")
    .setPredictionThreshold(0.5)
    .setInputCols(Array("re_ner_chunks", "sentences"))
    .setOutputCol("relations")

val pipeline = new Pipeline().setStages(Array(documenter, sentencer, tokenizer, pos_tagger, words_embedder, ner_tagger, ner_converter, dependency_parser, re_ner_chunk_filter, re_model))

val data = Seq("A 44-year-old man taking naproxen for chronic low back pain and a 20-year-old woman on oxaprozin for rheumatoid arthritis presented with tense bullae and cutaneous fragility on the face and the back of the hands.").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
|    | chunk1     | entity1  | chunk2                                                    | entity2 |   result |
|---:|:-----------|:---------|:----------------------------------------------------------|:--------|---------:|
|  0 | naproxen   | DRUG     | tense bullae                                              | ADE     |        1 |
|  1 | naproxen   | DRUG     | cutaneous fragility on the face and the back of the hands | ADE     |        1 |
|  2 | oxaprozin  | DRUG     | tense bullae                                              | ADE     |        1 |
|  3 | oxaprozin  | DRUG     | cutaneous fragility on the face and the back of the hands | ADE     |        1 |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|redl_ade_biobert|
|Compatibility:|Spark NLP for Healthcare 3.1.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[redl_ner_chunks, document]|
|Output Labels:|[relations]|
|Language:|en|

## Data Source

This model is trained on custom data annotated by JSL.

## Benchmarking

```bash
Relation           Recall Precision        F1   Support

0                   0.937     0.949     0.943      1060
1                   0.966     0.957     0.961      1540

Avg.                0.951     0.953     0.952

Weighted Avg.       0.954     0.954     0.954
```