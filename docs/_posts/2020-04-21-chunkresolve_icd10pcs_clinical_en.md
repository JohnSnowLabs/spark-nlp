---
layout: model
title: ChunkResolver Icd10pcs Clinical
author: John Snow Labs
name: chunkresolve_icd10pcs_clinical
class: ChunkEntityResolverModel
language: en
repository: clinical/models
date: 2020-04-21
tags: [clinical,entity_resolution,icd10,icd10pcs,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Entity Resolution model Based on KNN using Word Embeddings + Word Movers Distance  


{:.h2_title}
## Prediction Domain
ICD10-PCS Codes and their normalized definition with `clinical_embeddings`

[https://www.icd10data.com/ICD10PCS/Codes](https://www.icd10data.com/ICD10PCS/Codes)

{:.h2_title}
## Data Source
Trained on ICD10 Procedure Coding System dataset

{:.btn-box}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_icd10pcs_clinical_en_2.4.5_2.4_1587491320087.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = ChunkEntityResolverModel.pretrained("chunkresolve_icd10pcs_clinical","en","clinical/models")\
	.setInputCols("token","chunk_embeddings")\
	.setOutputCol("entity")
```

```scala
val model = ChunkEntityResolverModel.pretrained("chunkresolve_icd10pcs_clinical","en","clinical/models")
	.setInputCols("token","chunk_embeddings")
	.setOutputCol("entity")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|----------------|--------------------------------|
| name           | chunkresolve_icd10pcs_clinical |
| model_class    | ChunkEntityResolverModel       |
| compatibility  | 2.4.2                          |
| license        | Licensed                       |
| edition        | Healthcare                     |
| inputs         | token, chunk_embeddings        |
| output         | entity                         |
| language       | en                             |
| case_sensitive | True                           |
| upstream_deps  | embeddings_clinical            |

