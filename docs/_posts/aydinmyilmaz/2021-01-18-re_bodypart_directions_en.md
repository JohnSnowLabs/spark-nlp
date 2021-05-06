---
layout: model
title: Relation extraction between body parts and direction entities
author: John Snow Labs
name: re_bodypart_directions
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

Relation extraction between body parts entites [Internal_organ_or_component, External_body_part_or_region] and Direction entity in clinical texts

## Predicted Entities

`1` : Shows there is a relation between the body part entity and the direction entity
`0` : Shows there is no relation between the body part entity and the direction entity

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb#scrollTo=D8TtVuN-Ee8s){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_bodypart_directions_en_2.7.1_2.4_1610983817042.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

words_embedder = WordEmbeddingsModel()\
    .pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentences", "tokens"])\
    .setOutputCol("embeddings")

ner_tagger = sparknlp.annotators.NerDLModel()\
    .pretrained("jsl_ner_wip_greedy_clinical","en","clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags")    

pair_list = ['direction-internal_organ_or_component', 'internal_organ_or_component-direction']

re_model = RelationExtractionModel().pretrained("re_bodypart_directions","en","clinical/models")\
    .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
    .setOutputCol("relations")\
    .setMaxSyntacticDistance(4)\
    .setRelationPairs(pair_list)


pipeline = Pipeline(stages=[documenter, sentencer, tokenizer, words_embedder, pos_tagger, ner_tagger, ner_chunker, dependency_parser, re_model])
model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = LightPipeline(model).fullAnnotate(''' MRI demonstrated infarction in the upper brain stem , left cerebellum and  right basil ganglia ''')
```

```scala
...
val words_embedder = WordEmbeddingsModel()
    .pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens"))
    .setOutputCol("embeddings")

val ner_tagger = sparknlp.annotators.NerDLModel()
    .pretrained("jsl_ner_wip_greedy_clinical","en","clinical/models")
    .setInputCols("sentences", "tokens", "embeddings")
    .setOutputCol("ner_tags")    

val pair_list = Array('direction-internal_organ_or_component', 'internal_organ_or_component-direction')

val re_model = RelationExtractionModel().pretrained("re_bodypart_directions","en","clinical/models")
    .setInputCols(Array("embeddings", "pos_tags", "ner_chunks", "dependencies"))
    .setOutputCol("relations")
    .setMaxSyntacticDistance(4)
    .setRelationPairs(pair_list)

val nlpPipeline = new Pipeline().setStages(Array(documenter, sentencer, tokenizer, words_embedder, pos_tagger, ner_tagger, ner_chunker, dependency_parser, re_model))
val result = nlpPipeline.fit(Seq.empty[""].toDS.toDF("text")).transform(data)

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
|Model Name:|re_bodypart_directions|
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
| 0        | 0.87   | 0.9       | 0.88 |
| 1        | 0.99   | 0.99      | 0.99 |
```
