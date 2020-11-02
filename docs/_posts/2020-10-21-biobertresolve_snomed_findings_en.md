---
layout: model
title: Sentence Entity Resolver SNOMED Findings BioBert
author: John Snow Labs
name: biobertresolve_snomed_findings
class: SentenceEntityResolverModel
language: en
repository: clinical/models
date: 2020-10-21
tags: [clinical,entity_resolver]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description 


 {:.h2_title}
## Predicted Entities
Snomed Codes and their normalized definition with BertSentenceEmbeddings: `sent_biobert_pubmed_base_cased` 

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_SNOMED/){:.button.button-orange}<br/>[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/13.Snomed_Entity_Resolver_Model_Training.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}<br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/biobertresolve_snomed_findings_en_2.6.3_2.4_1603209744068.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = SentenceEntityResolverModel.pretrained("biobertresolve_snomed_findings","en","clinical/models")\
	.setInputCols("sentence_embeddings")\
	.setOutputCol("entity")
```

```scala
val model = SentenceEntityResolverModel.pretrained("biobertresolve_snomed_findings","en","clinical/models")
	.setInputCols("sentence_embeddings")
	.setOutputCol("entity")
```
</div>



{:.model-param}
## Model Information
{:.table-model}
|----------------|--------------------------------|
| Model Name     | biobertresolve_snomed_findings |
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
Trained on SNOMED CT Findings  
Visit [this](http://www.snomed.org/) link to get more information

