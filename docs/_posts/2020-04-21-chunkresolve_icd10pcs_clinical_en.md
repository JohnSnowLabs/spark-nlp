---
layout: model
title: ICD10PCS Entity Resolver
author: John Snow Labs
name: chunkresolve_icd10pcs_clinical
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
Entity Resolution model Based on KNN using Word Embeddings + Word Movers Distance.


## Predicted Entities
ICD10-PCS Codes and their normalized definition with `clinical_embeddings`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_icd10pcs_clinical_en_2.4.5_2.4_1587491320087.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/chunkresolve_icd10pcs_clinical_en_2.4.5_2.4_1587491320087.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}
{:.h2_title}
## How to use
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
...
model = ChunkEntityResolverModel.pretrained("chunkresolve_icd10pcs_clinical","en","clinical/models")
	.setInputCols("token","chunk_embeddings")
	.setOutputCol("entity")

pipeline_icd10pcs = Pipeline(stages = [documentAssembler, sentenceDetector, tokenizer, stopwords, word_embeddings, ner, chunk_embeddings, model])

data = ["""He has a starvation ketosis but nothing found for significant for dry oral mucosa"""]

pipeline_model = pipeline_icd10pcs.fit(spark.createDataFrame([[""]]).toDF("text"))

light_pipeline = LightPipeline(pipeline_model)

result = light_pipeline.annotate(data)
```

```scala
...
val model = ChunkEntityResolverModel.pretrained("chunkresolve_icd10pcs_clinical","en","clinical/models")
	.setInputCols("token","chunk_embeddings")
	.setOutputCol("entity")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, stopwords, word_embeddings, ner, chunk_embeddings, model))

val data = Seq("He has a starvation ketosis but nothing found for significant for dry oral mucosa").toDF("text")
val result = pipeline.fit(data).transform(data)
```
</div>

{:.h2_title}
## Results

```bash
|   | chunks               | begin | end | code    | resolutions                                      |
|---|----------------------|-------|-----|---------|--------------------------------------------------|
| 0 | a starvation ketosis | 7     | 26  | 6A3Z1ZZ | Hyperthermia, Multiple:::Narcosynthesis:::Hype...|
| 1 | dry oral mucosa      | 66    | 80  | 8E0ZXY4 | Yoga Therapy:::Release Cecum, Open Approach:::...|
```

{:.model-param}
## Model Information

{:.table-model}
|----------------|--------------------------------|
| Name:           | chunkresolve_icd10pcs_clinical |
| Type:    | ChunkEntityResolverModel       |
| Compatibility:  | Spark NLP 2.4.2+                         |
| License:        | Licensed                       |
|Edition:|Official|                     |
|Input labels:         | token, chunk_embeddings        |
|Output labels:        | entity                         |
| Language:       | en                             |
| Case sensitive: | True                           |
| Dependencies:  | embeddings_clinical            |

{:.h2_title}
## Data Source
Trained on ICD10 Procedure Coding System dataset
https://www.icd10data.com/ICD10PCS/Codes
