---
layout: model
title: Relation extraction between dates and clinical entities
author: John Snow Labs
name: re_date_clinical
date: 2021-01-18
task: Relation Extraction
language: en
edition: Spark NLP for Healthcare 2.7.1
tags: [en, relation_extraction, clinical, licensed]
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Relation extraction between date and related other entities

## Predicted Entities

`1` : Shows there is a relation between the date entity and other clinical entities
 `0` : Shows there is no relation between the date entity and other clinical entities

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_date_clinical_en_2.7.1_2.4_1611000334654.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, PerceptronModel, DependencyParserModel, WordEmbeddingsModel, NerDLModel, NerConverter, RelationExtractionModel.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
ner_tagger = sparknlp.annotators.NerDLModel()\ .pretrained('jsl_ner_wip_greedy_clinical','en','clinical/models')\ 
  .setInputCols("sentences", "tokens", "embeddings")\ 
  .setOutputCol("ner_tags")

re_model = RelationExtractionModel()
.pretrained("re_date", "en", 'clinical/models')
.setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])
.setOutputCol("relations")
.setMaxSyntacticDistance(3)\ #default: 0 .setPredictionThreshold(0.9)\ #default: 0.5 .setRelationPairs(["test-date", "symptom-date"]) # Possible relation pairs. Default: All Relations.

nlp_pipeline = Pipeline(stages=[ documenter, sentencer,tokenizer, words_embedder, pos_tagger, ner_tagger,ner_chunker, dependency_parser,re_model])

light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

annotations = light_pipeline.fullAnnotate('''This 73 y/o patient had CT on 1/12/95, with progressive memory and cognitive decline since 8/11/94.''')
```


</div>

## Results

```bash
|   | relations | entity1 | entity1_begin | entity1_end | chunk1                                   | entity2 | entity2_end | entity2_end | chunk2  | confidence |
|---|-----------|---------|---------------|-------------|------------------------------------------|---------|-------------|-------------|---------|------------|
| 0 | 1         | Test    | 24            | 25          | CT                                       | Date    | 31          | 37          | 1/12/95 | 1.0        |
| 1 | 1         | Symptom | 45            | 84          | progressive memory and cognitive decline | Date    | 92          | 98          | 8/11/94 | 1.0        |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|re_date_clinical|
|Type:|re|
|Compatibility:|Spark NLP 2.7.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[embeddings, pos_tags, train_ner_chunks, dependencies]|
|Output Labels:|[relations]|
|Language:|en|
|Dependencies:|embeddings_clinical|

## Data Source

Trained on data gathered and manually annotated by John Snow Labs

## Benchmarking

```bash
| relation | recall | precision | f1   |
|----------|--------|-----------|------|
| 0        | 0.74   | 0.71      | 0.72 |
| 1        | 0.94   | 0.95      | 0.94 |
```