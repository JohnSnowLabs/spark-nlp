---
layout: model
title: Embeddings Healthcare
author: John Snow Labs
name: embeddings_healthcare
class: WordEmbeddingsModel
language: en
repository: clinical/models
date: 2020-03-26
tags: [licensed,clinical,embeddings,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Word Embeddings lookup annotator that maps tokens to vectors.

## Predicted Entities 
Word2Vec feature vectors based on embeddings_healthcare.

{:.h2_title}
## Data Source
Trained on PubMed + ICD10 + UMLS + MIMIC III corpora
https://www.nlm.nih.gov/databases/download/pubmed_medline.html

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
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
| Name:          | embeddings_healthcare |
| Type:   | WordEmbeddingsModel   |
| Compatibility: | Spark NLP 2.4.4+                 |
| License:       | Licensed              |
| Edition:       | Official            |
|Input labels:        | [document, token]       |
|Output labels:       | [word_embeddings]       |
| Language:      | en                    |
| Dimension:    | 400.0                 |

