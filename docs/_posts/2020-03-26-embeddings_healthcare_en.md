---
layout: model
title: Embeddings Healthcare
author: John Snow Labs
name: embeddings_healthcare
class: WordEmbeddingsModel
language: en
repository: clinical/models
date: 2020-03-26
tags: [clinical,embeddings,pubmed,umls,mimic,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Word Embeddings lookup annotator that maps tokens to vectors  


{:.h2_title}
## Prediction Domain
Word2Vec feature vectors based on embeddings_healthcare

[https://www.nlm.nih.gov/databases/download/pubmed_medline.html](https://www.nlm.nih.gov/databases/download/pubmed_medline.html)

{:.h2_title}
## Data Source
Trained on PubMed + ICD10 + UMLS + MIMIC III corpora

{:.btn-box}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/embeddings_healthcare_en_2.4.4_2.4_1585188313964.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = WordEmbeddingsModel.pretrained("embeddings_healthcare","en","clinical/models")\
	.setInputCols("document","token")\
	.setOutputCol("word_embeddings")
```

```scala
val model = WordEmbeddingsModel.pretrained("embeddings_healthcare","en","clinical/models")
	.setInputCols("document","token")
	.setOutputCol("word_embeddings")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|---------------|-----------------------|
| name          | embeddings_healthcare |
| model_class   | WordEmbeddingsModel   |
| compatibility | 2.4.4                 |
| license       | Licensed              |
| edition       | Healthcare            |
| inputs        | document, token       |
| output        | word_embeddings       |
| language      | en                    |
| dimension     | 400                   |

