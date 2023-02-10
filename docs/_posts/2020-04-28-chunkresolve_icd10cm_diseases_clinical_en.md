---
layout: model
title: ICD10CM ChunkResolver
author: John Snow Labs
name: chunkresolve_icd10cm_diseases_clinical
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_icd10cm_diseases_clinical_en_2.4.5_2.4_1588105984876.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/chunkresolve_icd10cm_diseases_clinical_en_2.4.5_2.4_1588105984876.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}
{:.h2_title}
## How to use
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
...

icd10cmResolver = ChunkEntityResolverModel.pretrained('chunkresolve_icd10cm_diseases_clinical', 'en', "clinical/models")\
    .setEnableLevenshtein(True)\
    .setNeighbours(200).setAlternatives(5).setDistanceWeights([3,3,2,0,0,7])\
    .setInputCols('token', 'chunk_embs_jsl')\
    .setOutputCol('icd10cm_resolution')

pipeline_icd10 = Pipeline(stages = [documentAssembler, sentenceDetector, tokenizer, stopwords, word_embeddings, jslNer, drugNer, jslConverter, drugConverter, jslChunkEmbeddings, drugChunkEmbeddings, icd10cmResolver])

empty_df = spark.createDataFrame([[""]]).toDF("text")

data = ["""This is an 82 - year-old male with a history of prior tobacco use , hypertension , chronic renal insufficiency , COPD , gastritis , and TIA who initially presented to Braintree with a non-ST elevation MI and Guaiac positive stools , transferred to St . Margaret's Center for Women & Infants for cardiac catheterization with PTCA to mid LAD lesion complicated by hypotension and bradycardia requiring Atropine , IV fluids and transient dopamine possibly secondary to vagal reaction , subsequently transferred to CCU for close monitoring , hemodynamically stable at the time of admission to the CCU"""]

pipeline_model = pipeline_icd10.fit(empty_df)

light_pipeline = LightPipeline(pipeline_model)

result = light_pipeline.annotate(data)
```

```scala
...

val icd10cmResolver = ChunkEntityResolverModel.pretrained('chunkresolve_icd10cm_diseases_clinical', 'en', "clinical/models")
    .setEnableLevenshtein(True)
    .setNeighbours(200).setAlternatives(5).setDistanceWeights(Array(3,3,2,0,0,7))
    .setInputCols('token', 'chunk_embs_jsl')
    .setOutputCol('icd10cm_resolution')

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, stopwords, word_embeddings, jslNer, drugNer, jslConverter, drugConverter, jslChunkEmbeddings, drugChunkEmbeddings, icd10cmResolver))

val data = Seq("This is an 82 - year-old male with a history of prior tobacco use , hypertension , chronic renal insufficiency , COPD , gastritis , and TIA who initially presented to Braintree with a non-ST elevation MI and Guaiac positive stools , transferred to St . Margaret's Center for Women & Infants for cardiac catheterization with PTCA to mid LAD lesion complicated by hypotension and bradycardia requiring Atropine , IV fluids and transient dopamine possibly secondary to vagal reaction , subsequently transferred to CCU for close monitoring , hemodynamically stable at the time of admission to the CCU").toDF("text")
val result = pipeline.fit(data).transform(data)
```
</div>

{:.h2_title}
## Results

```bash
|   | coords      | chunk                       | entity    | icd10cm_opts                                                                              |
|---|-------------|-----------------------------|-----------|-------------------------------------------------------------------------------------------|
| 0 | 2::499::506 | insomnia                    | Diagnosis | [(G4700, Insomnia, unspecified), (G4709, Other insomnia), (F5102, Adjustment insomnia)...]|
| 1 | 4::83::109  | chronic renal insufficiency | Diagnosis | [(N185, Chronic kidney disease, stage 5), (N181, Chronic kidney disease, stage 1), (N1...]|
| 2 | 4::120::128 | gastritis                   | Diagnosis | [(K2970, Gastritis, unspecified, without bleeding), (B9681, Helicobacter pylori [H. py...]|
```
{:.model-param}
## Model Information

{:.table-model}
|----------------|----------------------------------------|
| Name:           | chunkresolve_icd10cm_diseases_clinical |
| Type:    | ChunkEntityResolverModel               |
| Compatibility:  | Spark NLP 2.4.5+                                  |
| License:        | Licensed                               |
|Edition:|Official|                             |
|Input labels:         | [token, chunk_embeddings]                |
|Output labels:        | [entity]                                 |
| Language:       | en                                     |
| Case sensitive: | True                                   |
| Dependencies:  | embeddings_clinical                    |

{:.h2_title}
## Data Source
Trained on ICD10CM Dataset Range: A000-N989 Except Neoplasms and Musculoskeletal
https://www.icd10data.com/ICD10CM/Codes/
