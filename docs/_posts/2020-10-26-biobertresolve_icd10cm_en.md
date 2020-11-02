---
layout: model
title: Sentence Entity Resolver ICD10CM BioBert
author: John Snow Labs
name: biobertresolve_icd10cm
class: SentenceEntityResolverModel
language: en
repository: clinical/models
date: 2020-10-26
tags: [clinical,entity_resolver,icd10]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 


 {:.h2_title}
## Predicted Entities
ICD10-CM Codes and their normalized definition with BertSentenceEmbeddings: `sent_biobert_pubmed_base_cased` 

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_ICD10_CM/){:.button.button-orange}<br/><button class="button button-orange" disabled>Open in Colab</button><br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/biobertresolve_icd10cm_en_2.6.3_2.4_1603673704767.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = SentenceEntityResolverModel.pretrained("biobertresolve_icd10cm","en","clinical/models")\
	.setInputCols("sentence_embeddings")\
	.setOutputCol("entity")
```

```scala
val model = SentenceEntityResolverModel.pretrained("biobertresolve_icd10cm","en","clinical/models")
	.setInputCols("sentence_embeddings")
	.setOutputCol("entity")
```
</div>



{:.model-param}
## Model Information
{:.table-model}
|----------------|--------------------------------|
| Model Name     | biobertresolve_icd10cm         |
| Model Class    | SentenceEntityResolverModel    |
| Dimension      | 2.4                            |
| Compatibility  | 2.6.3                          |
| License        | Licensed                       |
| Edition        | Healthcare                     |
| Inputs         | sentence_embeddings            |
| Output         | entity                         |
| Language       | en                             |
| Case Sensitive | True                           |
| Dependencies   | sent_biobert_pubmed_base_cased |




{:.h2_title}
## Data Source
Trained on ICD10CM Dataset  
Visit [this](https://www.icd10data.com/ICD10CM/Codes/) link to get more information

