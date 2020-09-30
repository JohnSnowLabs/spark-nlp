---
layout: model
title: ChunkResolver Icd10cm Injuries Clinical
author: John Snow Labs
name: chunkresolve_icd10cm_injuries_clinical
class: ChunkEntityResolverModel
language: en
repository: clinical/models
date: 2020-04-28
tags: [clinical,entity_resolution,icd10,icd10cm,injuries,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Entity Resolution model Based on KNN using Word Embeddings + Word Movers Distance  


{:.h2_title}
## Prediction Labels
ICD10-CM Codes and their normalized definition with `clinical_embeddings`

[https://www.icd10data.com/ICD10CM/Codes/S00-T88](https://www.icd10data.com/ICD10CM/Codes/S00-T88)

{:.h2_title}
## Data Source
Trained on ICD10CM Dataset Range: S0000XA-S98929S 

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_ICD10_CM/){:.button.button-orange}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_icd10cm_injuries_clinical_en_2.4.5_2.4_1588103825347.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = ChunkEntityResolverModel.pretrained("chunkresolve_icd10cm_injuries_clinical","en","clinical/models") \
	.setInputCols("token","chunk_embeddings") \
	.setOutputCol("entity")
```

```scala
val model = ChunkEntityResolverModel.pretrained("chunkresolve_icd10cm_injuries_clinical","en","clinical/models")
	.setInputCols("token","chunk_embeddings")
	.setOutputCol("entity")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|----------------|----------------------------------------|
| Model Name     | chunkresolve_icd10cm_injuries_clinical |
| Type           | ChunkEntityResolverModel               |
| Compatibility  | 2.4.5                                  |
| License        | Licensed                               |
| Edition        | Healthcare                             |
| Inputs         | token, chunk_embeddings                |
| Output         | entity                                 |
| Language       | en                                     |
| Case Sensitive | True                                   |
| Dependencies   | embeddings_clinical                    |

