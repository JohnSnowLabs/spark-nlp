---
layout: model
title: Relation extraction between body parts and problem entities (ReDL)
author: John Snow Labs
name: redl_bodypart_problem_biobert
date: 2021-02-04
task: Relation Extraction
language: en
edition: Spark NLP 2.7.3
tags: [licensed, clinical, en, relation_extraction]
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Relation extraction between body parts and problem entities in clinical texts.

## Predicted Entities

`1` : Shows that there is a relation between body part entity and entities labeled as problem ( diagnosis, symptom etc.) 
`0` : Shows that there no relation between body part and problem entities

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.1.Clinical_Relation_Extraction_BodyParts_Models.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/redl_bodypart_problem_biobert_en_2.7.3_2.4_1612446369008.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
    .pretrained("jsl_ner_wip_greedy_clinical", "en", "clinical/models") \
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
    .setOutputCol("re_ner_chunks")#.setRelationPairs(['SYMPTOM-EXTERNAL_BODY_PART_OR_REGION'])

# The dataset this model is trained to is sentence-wise. 
# This model can also be trained on document-level relations - in which case, while predicting, use "document" instead of "sentence" as input.
re_model = RelationExtractionDLModel()\
    .pretrained('redl_bodypart_problem_biobert', 'en', "clinical/models") \
    .setPredictionThreshold(0.5)\
    .setInputCols(["re_ner_chunks", "sentences"]) \
    .setOutputCol("relations")

pipeline = Pipeline(stages=[documenter, sentencer, tokenizer, pos_tagger, words_embedder, ner_tagger, ner_converter, dependency_parser, re_ner_chunk_filter, re_model])

text ="No neurologic deficits other than some numbness in his left hand."
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
    .pretrained("ner_clinical", "en", "clinical/models")
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
    .setOutputCol("re_ner_chunks").setRelationPairs(Array("SYMPTOM-EXTERNAL_BODY_PART_OR_REGION"))

// The dataset this model is trained to is sentence-wise. 
// This model can also be trained on document-level relations - in which case, while predicting, use "document" instead of "sentence" as input.
val re_model = RelationExtractionDLModel()
    .pretrained("redl_bodypart_problem_biobert", "en", "clinical/models")
    .setPredictionThreshold(0.5)
    .setInputCols(Array("re_ner_chunks", "sentences"))
    .setOutputCol("relations")
val pipeline = new Pipeline().setStages(Array(documenter, sentencer, tokenizer, pos_tagger, words_embedder, ner_tagger, ner_converter, dependency_parser, re_ner_chunk_filter, re_model))

val result = pipeline.fit(Seq.empty["No neurologic deficits other than some numbness in his left hand."].toDS.toDF("text")).transform(data)
```

</div>

## Results

```bash
|    |   relation | entity1   | chunk1   | entity2                      | chunk2   |   confidence |
|---:|-----------:|:----------|:---------|:-----------------------------|:---------|-------------:|
|  0 |          1 | Symptom   | numbness | External_body_part_or_region | hand     |            1 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|redl_bodypart_problem_biobert|
|Compatibility:|Spark NLP 2.7.3+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|

## Data Source

Trained on internal dataset.

## Benchmarking

```bash
Relation           Recall Precision        F1   Support
0                   0.762     0.814     0.787       315
1                   0.938     0.917     0.927       885
Avg.                0.850     0.865     0.857
```