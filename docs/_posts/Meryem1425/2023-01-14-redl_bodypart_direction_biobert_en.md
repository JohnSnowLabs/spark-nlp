---
layout: model
title: Relation Extraction Between Body Parts and Direction Entities (ReDL)
author: John Snow Labs
name: redl_bodypart_direction_biobert
date: 2023-01-14
tags: [licensed, en, clinical, relation_extraction, tensorflow]
task: Relation Extraction
language: en
nav_key: models
edition: Healthcare NLP 4.2.4
spark_version: 3.0
supported: true
engine: tensorflow
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Relation extraction between body parts entities like Internal_organ_or_component, External_body_part_or_region etc. and direction entities like upper, lower in clinical texts. 1 : Shows the body part and direction entity are related, 0 : Shows the body part and direction entity are not related.

## Predicted Entities

`1`, `0`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.1.Clinical_Relation_Extraction_BodyParts_Models.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/redl_bodypart_direction_biobert_en_4.2.4_3.0_1673710170047.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/redl_bodypart_direction_biobert_en_4.2.4_3.0_1673710170047.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documenter = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentencer = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentences")

tokenizer = Tokenizer()\
    .setInputCols(["sentences"])\
    .setOutputCol("tokens")

pos_tagger = PerceptronModel()\
    .pretrained("pos_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens"])\
    .setOutputCol("pos_tags")

words_embedder = WordEmbeddingsModel() \
    .pretrained("embeddings_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens"]) \
    .setOutputCol("embeddings")

ner_tagger = MedicalNerModel.pretrained("ner_jsl_greedy", "en", "clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags") 

ner_converter = NerConverterInternal() \
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
    .setOutputCol("re_ner_chunks")\
    .setRelationPairs(['direction-external_body_part_or_region', 
    'external_body_part_or_region-direction',
    'direction-internal_organ_or_component',
    'internal_organ_or_component-direction'
    ])

# The dataset this model is trained to is sentence-wise. 
# This model can also be trained on document-level relations - in which case, while predicting, use "document" instead of "sentence" as input.
re_model = RelationExtractionDLModel()\
    .pretrained('redl_bodypart_direction_biobert', 'en', "clinical/models") \
    .setPredictionThreshold(0.5)\
    .setInputCols(["re_ner_chunks", "sentences"]) \
    .setOutputCol("relations")

pipeline = Pipeline(stages=[documenter, sentencer, tokenizer, pos_tagger, words_embedder, ner_tagger, ner_converter, dependency_parser, re_ner_chunk_filter, re_model])

data = spark.createDataFrame([[''' MRI demonstrated infarction in the upper brain stem , left cerebellum and  right basil ganglia ''']]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documenter = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")

val sentencer = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentences")

val tokenizer = new Tokenizer()
    .setInputCols("sentences")
    .setOutputCol("tokens")

val pos_tagger = PerceptronModel()
    .pretrained("pos_clinical", "en", "clinical/models") 
    .setInputCols(Array("sentences", "tokens"))
    .setOutputCol("pos_tags")

val words_embedder = WordEmbeddingsModel()
    .pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens"))
    .setOutputCol("embeddings")

val ner_tagger = MedicalNerModel.pretrained("ner_jsl_greedy", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens", "embeddings"))
    .setOutputCol("ner_tags") 

val ner_converter = new NerConverterInternal()
    .setInputCols(Array("sentences", "tokens", "ner_tags"))
    .setOutputCol("ner_chunks")

val dependency_parser = DependencyParserModel()
    .pretrained("dependency_conllu", "en")
    .setInputCols(Array("sentences", "pos_tags", "tokens"))
    .setOutputCol("dependencies")

// Set a filter on pairs of named entities which will be treated as relation candidates
val re_ner_chunk_filter = new RENerChunksFilter()
    .setInputCols(Array("ner_chunks", "dependencies"))
    .setMaxSyntacticDistance(10)
    .setOutputCol("re_ner_chunks")
    .setRelationPairs(Array("direction-external_body_part_or_region", 
    "external_body_part_or_region-direction",
    "direction-internal_organ_or_component",
    "internal_organ_or_component-direction"))

// The dataset this model is trained to is sentence-wise. 
// This model can also be trained on document-level relations - in which case, while predicting, use "document" instead of "sentence" as input.
val re_model = RelationExtractionDLModel()
    .pretrained("redl_bodypart_direction_biobert", "en", "clinical/models")
    .setPredictionThreshold(0.5)
    .setInputCols(Array("re_ner_chunks", "sentences"))
    .setOutputCol("relations")

val pipeline = new Pipeline().setStages(Array(documenter, sentencer, tokenizer, pos_tagger, words_embedder, ner_tagger, ner_converter, dependency_parser, re_ner_chunk_filter, re_model))

val data = Seq("MRI demonstrated infarction in the upper brain stem , left cerebellum and  right basil ganglia").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
| index | relations | entity1                     | entity1_begin | entity1_end | chunk1     | entity2                     | entity2_end | entity2_end | chunk2        | confidence |
|-------|-----------|-----------------------------|---------------|-------------|------------|-----------------------------|-------------|-------------|---------------|------------|
| 0     | 1         | Direction                   | 35            | 39          | upper      | Internal_organ_or_component | 41          | 50          | brain stem    | 0.9999989  |
| 1     | 0         | Direction                   | 35            | 39          | upper      | Internal_organ_or_component | 59          | 68          | cerebellum    | 0.99992585 |
| 2     | 0         | Direction                   | 35            | 39          | upper      | Internal_organ_or_component | 81          | 93          | basil ganglia | 0.9999999  |
| 3     | 0         | Internal_organ_or_component | 41            | 50          | brain stem | Direction                   | 54          | 57          | left          | 0.999811   |
| 4     | 0         | Internal_organ_or_component | 41            | 50          | brain stem | Direction                   | 75          | 79          | right         | 0.9998203  |
| 5     | 1         | Direction                   | 54            | 57          | left       | Internal_organ_or_component | 59          | 68          | cerebellum    | 1.0        |
| 6     | 0         | Direction                   | 54            | 57          | left       | Internal_organ_or_component | 81          | 93          | basil ganglia | 0.97616416 |
| 7     | 0         | Internal_organ_or_component | 59            | 68          | cerebellum | Direction                   | 75          | 79          | right         | 0.953046   |
| 8     | 1         | Direction                   | 75            | 79          | right      | Internal_organ_or_component | 81          | 93          | basil ganglia | 1.0        |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|redl_bodypart_direction_biobert|
|Compatibility:|Healthcare NLP 4.2.4+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|401.7 MB|

## References

Trained on an internal dataset.

## Benchmarking

```bash
label              Recall Precision        F1   Support
0                   0.856     0.873     0.865       153
1                   0.986     0.984     0.985      1347
Avg.                0.921     0.929     0.925         -
```
