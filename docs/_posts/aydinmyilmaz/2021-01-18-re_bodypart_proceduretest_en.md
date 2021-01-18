---
layout: model
title: Relation extraction between body parts and procedures
author: John Snow Labs
name: re_bodypart_proceduretest
date: 2021-01-18
tags: [en, relation_extraction, clinical, licensed]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Relation extraction between body parts entites ['Internal_organ_or_component','External_body_part_or_region'] and procedure and test entities

## Predicted Entities

  `1`
  `0`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb#scrollTo=D8TtVuN-Ee8s){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_bodypart_proceduretest_en_2.7.1_2.4_1610989267602.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, PerceptronModel, DependencyParserModel, WordEmbeddingsModel, NerDLModel, NerConverter, RelationExtractionModel.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

ner_tagger = sparknlp.annotators.NerDLModel()\
    .pretrained('jsl_ner_wip_greedy_clinical','en','clinical/models')\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags") 

re_model = RelationExtractionModel()\
    .pretrained("re_bodypart_proceduretest", "en", 'clinical/models')\
    .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
    .setOutputCol("relations")\
    .setMaxSyntacticDistance(4)\ #default: 0
    .setPredictionThreshold(0.9)\ #default: 0.5
    .setRelationPairs(["external_body_part_or_region-test"]) # Possible relation pairs. Default: All Relations.

nlp_pipeline = Pipeline(stages=[ documenter, sentencer,tokenizer, words_embedder, pos_tagger,  clinical_ner_tagger,ner_chunker, dependency_parser,re_model])

light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

annotations = light_pipeline.fullAnnotate(''''TECHNIQUE IN DETAIL: After informed consent was obtained from the patient and his mother, the chest was scanned with portable ultrasound.'''')
```


</div>

## Results

```bash
| index | relations | entity1                      | entity1_begin | entity1_end | chunk1 | entity2 | entity2_end | entity2_end | chunk2              | confidence |
|-------|-----------|------------------------------|---------------|-------------|--------|---------|-------------|-------------|---------------------|------------|
| 0     | 1         | External_body_part_or_region | 94            | 98          | chest  | Test    | 117         | 135         | portable ultrasound | 1.0        |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|re_bodypart_proceduretest|
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
| 0        | 0.55   | 0.35      | 0.43 |
| 1        | 0.73   | 0.86      | 0.79 |

```
