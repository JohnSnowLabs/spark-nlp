---
layout: model
title: ICD10CM Poison Entity Resolver
author: John Snow Labs
name: chunkresolve_icd10cm_poison_ext_clinical
class: ChunkEntityResolverModel
language: en
repository: clinical/models
date: 2020-04-28
task: Entity Resolution
edition: Healthcare NLP 2.4.5
spark_version: 2.4
tags: [clinical,licensed,entity_resolution,en]
deprecated: true
annotator: ChunkEntityResolverModel
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Entity Resolution model Based on KNN using Word Embeddings + Word Movers Distance.

## Predicted Entities
ICD10-CM Codes and their normalized definition with ``clinical_embeddings``.

{:.btn-box}

[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_ICD10_CM/){:.button.button-orange.button-orange-trans.arr.button-icon}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ER_ICD10_CM.ipynb){:.button.button-orange.button-orange-trans.arr.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_icd10cm_poison_ext_clinical_en_2.4.5_2.4_1588106053455.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/chunkresolve_icd10cm_poison_ext_clinical_en_2.4.5_2.4_1588106053455.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}
{:.h2_title}
## How to use
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
...
model = ChunkEntityResolverModel.pretrained("chunkresolve_icd10cm_poison_ext_clinical","en","clinical/models")\
	.setInputCols("token","chunk_embeddings")\
	.setOutputCol("icd10_code")

pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings, ner_model, ner_chunker, chunk_embeddings, model])

light_pipeline  = LightPipeline(pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

light_pipeline.fullAnnotate("""The patient is a 5-month-old infant who presented initially on Monday with a cold, cough, and runny nose for 2 days. She had no difficulty breathing and her cough was described as dry and hacky. At that time, physical exam showed a right TM, which was red. Left TM was okay. She was fairly congested but looked happy and playful. She was started on Amoxil and Aldex and we told to recheck in 2 weeks to recheck her ear. Mom returned to clinic again today because she got much worse overnight. She was having difficulty breathing. She was much more congested and her appetite had decreased significantly today. She also spiked a temperature yesterday of 102.6 and always having trouble sleeping secondary to congestion.""")

```

```scala
...
val model = ChunkEntityResolverModel.pretrained("chunkresolve_icd10cm_poison_ext_clinical","en","clinical/models")
	.setInputCols("token","chunk_embeddings")
	.setOutputCol("icd10_code")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings, ner_model, ner_chunker, chunk_embeddings, model))

val data = Seq("The patient is a 5-month-old infant who presented initially on Monday with a cold, cough, and runny nose for 2 days. She had no difficulty breathing and her cough was described as dry and hacky. At that time, physical exam showed a right TM, which was red. Left TM was okay. She was fairly congested but looked happy and playful. She was started on Amoxil and Aldex and we told to recheck in 2 weeks to recheck her ear. Mom returned to clinic again today because she got much worse overnight. She was having difficulty breathing. She was much more congested and her appetite had decreased significantly today. She also spiked a temperature yesterday of 102.6 and always having trouble sleeping secondary to congestion.").toDF("text")
val result = pipeline.fit(data).transform(data)
```
</div>

## Result
```bash
| # |                chunk | begin | end |  entity |                                 icd10_description | icd10_code |
|--:|---------------------:|------:|----:|--------:|--------------------------------------------------:|------------|
| 0 |        a cold, cough |    75 |  87 | PROBLEM | Chronic obstructive pulmonary disease, unspeci... |       J449 |
| 1 |           runny nose |    94 | 103 | PROBLEM |                                  Nasal congestion |      R0981 |
| 2 | difficulty breathing |   210 | 229 | PROBLEM |                               Shortness of breath |      R0602 |
| 3 |            her cough |   235 | 243 | PROBLEM |                                             Cough |        R05 |
| 4 |     fairly congested |   365 | 380 | PROBLEM |                                Edema, unspecified |       R609 |
| 5 | difficulty breathing |   590 | 609 | PROBLEM |                               Shortness of breath |      R0602 |
| 6 |       more congested |   625 | 638 | PROBLEM |                                Edema, unspecified |       R609 |
| 7 |     trouble sleeping |   759 | 774 | PROBLEM |                                Activity, sleeping |      Y9384 |
| 8 |           congestion |   789 | 798 | PROBLEM |                                  Nasal congestion |      R0981 |
```

{:.model-param}
## Model Information

{:.table-model}
|----------------|------------------------------------------|
| Name:           | chunkresolve_icd10cm_poison_ext_clinical |
| Type:    | ChunkEntityResolverModel                 |
| Compatibility:  | Spark NLP 2.4.5+                                    |
| License:        | Licensed                                 |
|Edition:|Official|                               |
|Input labels:         | [token, chunk_embeddings]                  |
|Output labels:        | [icd10_code]                                   |
| Language:       | en                                       |
| Case sensitive: | True                                     |
| Dependencies:  | embeddings_clinical                      |

{:.h2_title}
## Data Source
Trained on ICD10CM Dataset Range: T1500XA-T879
https://www.icd10data.com/ICD10CM/Codes/S00-T88
