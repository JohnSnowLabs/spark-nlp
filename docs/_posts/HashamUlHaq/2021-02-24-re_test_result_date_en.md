---
layout: model
title: Relation Extraction between Tests, Results, and Dates
author: John Snow Labs
name: re_test_result_date
date: 2021-02-24
tags: [licensed, en, clinical, relation_extraction]
task: Relation Extraction
language: en
edition: Spark NLP for Healthcare 2.7.4
spark_version: 2.4
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Relation extraction between lab test names, their findings, measurements, results, and date.

## Predicted Entities

`is_finding_of`, `is_result_of`, `is_date_of`, `O`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/RE_CLINICAL_DATE/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb#scrollTo=D8TtVuN-Ee8s){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_test_result_date_en_2.7.4_2.4_1614168615976.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, PerceptronModel, DependencyParserModel, WordEmbeddingsModel, NerDLModel, NerConverter, RelationExtractionModel


In the table below, `re_test_result_date` RE model, its labels, optimal NER model, and meaningful relation pairs are illustrated.



|       RE MODEL      |                     RE MODEL LABES                     | NER MODEL | RE PAIRS                                                                                                                                                                                                                                                                                                                                 |
|:-------------------:|:------------------------------------------------------:|:---------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| re_test_result_date | is_finding_of, <br>is_result_of, <br>is_date_of, <br>O |  ner_jsl  | [“test-test_result”, <br>“test_result-test”,<br>“test-date”, “date-test”,<br>“test-imagingfindings”, <br>“imagingfindings-test”,<br>“test-ekg_findings”, <br>“ekg_findings-test”,<br>“date-test_result”, <br>“test_result-date”,<br>“date-imagingfindings”, <br>“imagingfindings-date”,<br>“date-ekg_findings”, <br>“ekg_findings-date”] |



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
ner_tagger = sparknlp.annotators.NerDLModel()\
.pretrained('jsl_ner_wip_clinical',"en","clinical/models")\
.setInputCols("sentences", "tokens", "embeddings")\
.setOutputCol("ner_tags") 

re_model = RelationExtractionModel()\
.pretrained("re_test_result_date", "en", 'clinical/models')\
.setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
.setOutputCol("relations")\
.setMaxSyntacticDistance(4)\ #default: 0
.setPredictionThreshold(0.9)\ #default: 0.5
.setRelationPairs(["external_body_part_or_region-test"]) # Possible relation pairs. Default: All Relations.

nlp_pipeline = Pipeline(stages=[ documenter, sentencer,tokenizer, words_embedder, pos_tagger,  clinical_ner_tagger,ner_chunker, dependency_parser,re_model])

light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

annotations = light_pipeline.fullAnnotate(''''He was advised chest X-ray or CT scan after checking his SpO2 which was <= 93%''')
```



{:.nlu-block}
```python
import nlu
nlu.load("en.relation.test_result_date").predict("""He was advised chest X-ray or CT scan after checking his SpO2 which was <= 93%""")
```

</div>

## Results

```bash
| index | relations    | entity1      | chunk1              | entity2      |  chunk2 |
|-------|--------------|--------------|---------------------|--------------|---------|
| 0     | O            | TEST         | chest X-ray         | MEASUREMENTS |  93%    | 
| 1     | O            | TEST         | CT scan             | MEASUREMENTS |  93%    |
| 2     | is_result_of | TEST         | SpO2                | MEASUREMENTS |  93%    |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|re_test_result_date|
|Type:|re|
|Compatibility:|Spark NLP for Healthcare 2.7.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[embeddings, pos_tags, train_ner_chunks, dependencies]|
|Output Labels:|[relations]|
|Language:|en|

## Data Source

Trained on internal data.

## Benchmarking

```bash
| relation        | prec |
|-----------------|------|
| O               | 0.77 |
| is_finding_of   | 0.80 |
| is_result_of    | 0.96 |
| is_date_of      | 0.94 |

```