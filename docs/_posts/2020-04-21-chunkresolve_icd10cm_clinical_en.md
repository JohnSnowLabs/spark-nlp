---
layout: model
title: ChunkResolver Icd10cm Clinical
author: John Snow Labs
name: chunkresolve_icd10cm_clinical
class: ChunkEntityResolverModel
language: en
repository: clinical/models
date: 2020-04-21
tags: [clinical,entity_resolution,icd10,icd10cm,en]
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

[https://www.icd10data.com/ICD10CM/Codes/](https://www.icd10data.com/ICD10CM/Codes/)

{:.h2_title}
## Data Source
Trained on ICD10 Clinical Modification datasetwith tenths of variations per code

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_ICD10_CM/){:.button.button-orange}[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/enterprise/healthcare/EntityResolution_ICD10_RxNorm_Detailed.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_icd10cm_clinical_en_2.4.5_2.4_1587491222166.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = ChunkEntityResolverModel.pretrained("chunkresolve_icd10cm_clinical","en","clinical/models") \
	.setInputCols("token","chunk_embeddings") \
	.setOutputCol("entity")
```

```scala
val model = ChunkEntityResolverModel.pretrained("chunkresolve_icd10cm_clinical","en","clinical/models")
	.setInputCols("token","chunk_embeddings")
	.setOutputCol("entity")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|----------------|-------------------------------|
| Model Name     | chunkresolve_icd10cm_clinical |
| Type           | ChunkEntityResolverModel      |
| Compatibility  | 2.4.2                         |
| License        | Licensed                      |
| Edition        | Healthcare                    |
| Inputs         | token, chunk_embeddings       |
| Output         | entity                        |
| Language       | en                            |
| Case Sensitive | True                          |
| Dependencies   | embeddings_clinical           |

