---
layout: model
title: SentenceResolver SNOMED Findings BioBert
author: John Snow Labs
name: biobertresolve_snomed_findings_clinical
class: 
language: en
repository: clinical/models
date: 21/10/2020
tags: [clinical,entity_resolver]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Map recognized clinical concepts to Snomed codes.

 {:.h2_title}
## Predicted Entities
Snomed Codes and their normalized definition with BertSentenceEmbeddings: `sent_biobert_pubmed_base_cased` 

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_SNOMED/){:.button.button-orange}<br/>[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/13.Snomed_Entity_Resolver_Model_Training.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}<br/>[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/biobertresolve_snomed_findings_en_2.6.3_2.4_1603209744068.zip){:.button.button-orange.button-orange-trans.arr.button-icon}<br/>

## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = SentenceEntityResolverModel.pretrained("biobertresolve_snomed_findings_clinical","en","clinical/models")\
	.setInputCols("sentence_embeddings")\
	.setOutputCol("entity")
```

```scala

```
</div>



{:.model-param}
## Model Information

{:.table-model}
|-------------------------|-----------------------------------------|
| Model Name              | biobertresolve_snomed_findings_clinical |
| Model Class             | SentenceEntityResolverModel             |
| Spark Compatibility     | 2.6.3                                   |
| Spark NLP Compatibility | 2.4                                     |
| License                 | Licensed                                |
| Edition                 | Healthcare                              |
| Input Labels            | sentence_embeddings                     |
| Output Labels           | entity                                  |
| Language                | en                                      |
| Case Sensitive          | True                                    |
| Upstream Dependencies   | sent_biobert_pubmed_base_cased          |




{:.h2_title}
## Data Source
Trained on SNOMED CT Findings

