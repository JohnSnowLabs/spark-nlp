---
layout: model
title: Extract relations between problem, test, and findings in reports
author: John Snow Labs
name: re_test_problem_finding
date: 2021-04-19
tags: [en, relation_extraction, licensed, clinical]
task: Relation Extraction
language: en
edition: Spark NLP for Healthcare 2.7.1
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Find relations between diagnosis, tests and imaging findings in radiology reports.

## Predicted Entities

`1` : The two entities are related. `0` : The two entities are not related

{:.btn-box}
[Live Demo](https://nlp.johnsnowlabs.com/demo){:.button.button-orange}
[Open in Colab](https://githubtocolab.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_test_problem_finding_en_2.7.1_2.4_1618830922197.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
ner_tagger = sparknlp.annotators.NerDLModel()\
    .pretrained('jsl_ner_wip_clinical',"en","clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags") 

re_model = RelationExtractionModel()\
    .pretrained("re_test_problem_finding", "en", 'clinical/models')\
    .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
    .setOutputCol("relations")\
    .setMaxSyntacticDistance(4)\ #default: 0
    .setPredictionThreshold(0.9)\ #default: 0.5
    .setRelationPairs(["procedure-symptom"]) # Possible relation pairs. Default: All Relations.

nlp_pipeline = Pipeline(stages=[ documenter, sentencer,tokenizer, words_embedder, pos_tagger,  clinical_ner_tagger,ner_chunker, dependency_parser,re_model])

light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

annotations = light_pipeline.fullAnnotate(''''Targeted biopsy of this lesion for histological correlation should be considered.'')
```

</div>

## Results

```bash
| index | relations    | entity1      | chunk1              | entity2      |  chunk2 |
|-------|--------------|--------------|---------------------|--------------|---------|
| 0     | 1            | PROCEDURE    | biopsy              | SYMPTOM      |  lesion | 
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|re_test_problem_finding|
|Type:|re|
|Compatibility:|Spark NLP for Healthcare 2.7.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[embeddings, pos_tags, train_ner_chunks, dependencies]|
|Output Labels:|[relations]|
|Language:|en|

## Data Source

Trained on internal datasets.