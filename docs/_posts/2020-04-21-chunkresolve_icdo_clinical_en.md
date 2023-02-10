---
layout: model
title: ICDO Entity Resolver
author: John Snow Labs
name: chunkresolve_icdo_clinical
class: ChunkEntityResolverModel
language: en
repository: clinical/models
date: 2020-04-21
task: Entity Resolution
edition: Healthcare NLP 2.4.2
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
Entity Resolution model Based on KNN using Word Embeddings + Word Movers Distance

## Predicted Entities
ICD-O Codes and their normalized definition with `clinical_embeddings`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_ICDO/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ER_ICDO.ipynb#scrollTo=Qdh2BQaLI7tU){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_icdo_clinical_en_2.4.5_2.4_1587491354644.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/chunkresolve_icdo_clinical_en_2.4.5_2.4_1587491354644.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}
{:.h2_title}
## How to use
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
...

model = ChunkEntityResolverModel.pretrained("chunkresolve_icdo_clinical","en","clinical/models")
	.setInputCols("token","chunk_embeddings")
	.setOutputCol("entity")

pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings, clinical_ner_model, clinical_ner_chunker, chunk_embeddings, model])

data = ["""DIAGNOSIS: Left breast adenocarcinoma stage T3 N1b M0, stage IIIA.
She has been found more recently to have stage IV disease with metastatic deposits and recurrence involving the chest wall and lower left neck lymph nodes.
PHYSICAL EXAMINATION
NECK: On physical examination palpable lymphadenopathy is present in the left lower neck and supraclavicular area. No other cervical lymphadenopathy or supraclavicular lymphadenopathy is present.
RESPIRATORY: Good air entry bilaterally. Examination of the chest wall reveals a small lesion where the chest wall recurrence was resected. No lumps, bumps or evidence of disease involving the right breast is present.
ABDOMEN: Normal bowel sounds, no hepatomegaly. No tenderness on deep palpation. She has just started her last cycle of chemotherapy today, and she wishes to visit her daughter in Brooklyn, New York. After this she will return in approximately 3 to 4 weeks and begin her radiotherapy treatment at that time."""]

pipeline_model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

light_pipeline = LightPipeline(pipeline_model)
result = light_pipeline.annotate(data)

```

```scala
...
val model = ChunkEntityResolverModel.pretrained("chunkresolve_icdo_clinical","en","clinical/models")
	.setInputCols("token","chunk_embeddings")
	.setOutputCol("entity")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings, clinical_ner_model, clinical_ner_chunker, chunk_embeddings, model))

val data = Seq("DIAGNOSIS: Left breast adenocarcinoma stage T3 N1b M0, stage IIIA. She has been found more recently to have stage IV disease with metastatic deposits and recurrence involving the chest wall and lower left neck lymph nodes. PHYSICAL EXAMINATION NECK: On physical examination palpable lymphadenopathy is present in the left lower neck and supraclavicular area. No other cervical lymphadenopathy or supraclavicular lymphadenopathy is present. RESPIRATORY: Good air entry bilaterally. Examination of the chest wall reveals a small lesion where the chest wall recurrence was resected. No lumps, bumps or evidence of disease involving the right breast is present. ABDOMEN: Normal bowel sounds, no hepatomegaly. No tenderness on deep palpation. She has just started her last cycle of chemotherapy today, and she wishes to visit her daughter in Brooklyn, New York. After this she will return in approximately 3 to 4 weeks and begin her radiotherapy treatment at that time.").toDF("text")
val result = pipeline.fit(data).transform(data)
```
</div>

{:.h2_title}
## Results

```bash
|   | chunk                      | begin | end | entity | idco_description                            | icdo_code |
|---|----------------------------|-------|-----|--------|---------------------------------------------|-----------|
| 0 | Left breast adenocarcinoma | 11    | 36  | Cancer | Intraductal carcinoma, noninfiltrating, NOS | 8500/2    |
| 1 | T3 N1b M0                  | 44    | 52  | Cancer | Kaposi sarcoma                              | 9140/3    |
```

{:.model-param}
## Model Information

{:.table-model}
|----------------|----------------------------|
| Name:           | chunkresolve_icdo_clinical |
| Type:    | ChunkEntityResolverModel   |
| Compatibility:  | Spark NLP 2.4.2+                     |
| License:        | Licensed                   |
|Edition:|Official|                 |
|Input labels:         | token, chunk_embeddings    |
|Output labels:        | entity                     |
| Language:       | en                         |
| Case sensitive: | True                       |
| Dependencies:  | embeddings_clinical        |

{:.h2_title}
## Data Source
Trained on ICD-O Histology Behaviour dataset
https://apps.who.int/iris/bitstream/handle/10665/96612/9789241548496_eng.pdf
