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
## Prediction Domain
ICD10-CM Codes and their normalized definition with `clinical_embeddings`

{:.h2_title}
## Data Source
Trained on ICD10CM Dataset Range: S0000XA-S98929S 
https://www.icd10data.com/ICD10CM/Codes/S00-T88

{:.btn-box}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_icd10cm_injuries_clinical_en_2.4.5_2.4_1588103825347.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = ChunkEntityResolverModel.pretrained("chunkresolve_icd10cm_injuries_clinical","en","clinical/models")
	.setInputCols("token","chunk_embeddings")
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
| name           | chunkresolve_icd10cm_injuries_clinical |
| model_class    | ChunkEntityResolverModel               |
| compatibility  | 2.4.5                                  |
| license        | Licensed                               |
| edition        | Healthcare                             |
| inputs         | token, chunk_embeddings                |
| output         | entity                                 |
| language       | en                                     |
| case_sensitive | True                                   |
| upstream_deps  | embeddings_clinical                    |

