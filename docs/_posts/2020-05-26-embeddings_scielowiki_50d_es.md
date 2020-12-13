---
layout: model
title: Embeddings Scielowiki 50 dims
author: John Snow Labs
name: embeddings_scielowiki_50d
class: WordEmbeddingsModel
language: es
repository: clinical/models
date: 2020-05-26
tags: [clinical,embeddings,es]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Word Embeddings lookup annotator that maps tokens to vectors

## Predicted Entities 
Word2Vec feature vectors based on embeddings_scielowiki_50d

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/embeddings_scielowiki_50d_es_2.5.0_2.4_1590467602230.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = WordEmbeddingsModel.pretrained("embeddings_scielowiki_50d","es","clinical/models")\
	.setInputCols("document","token")\
	.setOutputCol("word_embeddings")
```

```scala
val model = WordEmbeddingsModel.pretrained("embeddings_scielowiki_50d","es","clinical/models")
	.setInputCols("document","token")
	.setOutputCol("word_embeddings")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---------------|---------------------------|
| Name:          | embeddings_scielowiki_50d |
| Type:   | WordEmbeddingsModel       |
| Compatibility: | Spark NLP 2.5.0+                    |
| License:       | Licensed                  |
| Edition:       | Official                |
|Input labels:        | [document, token]           |
|Output labels:       | [word_embeddings]           |
| Language:      | es                        |
| Dimension:    | 50.0                      |

{:.h2_title}
## Data Source
Trained on Scielo Articles + Clinical Wikipedia Articles
https://zenodo.org/record/3744326#.XtViinVKh_U
