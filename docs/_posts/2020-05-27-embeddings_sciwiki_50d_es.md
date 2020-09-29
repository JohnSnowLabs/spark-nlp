---
layout: model
title: Embeddings Sciwiki 50 dims
author: John Snow Labs
name: embeddings_sciwiki_50d
class: WordEmbeddingsModel
language: es
repository: clinical/models
date: 2020-05-27
tags: [clinical,embeddings,wikipedia,es]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Word Embeddings lookup annotator that maps tokens to vectors  


{:.h2_title}
## Prediction Domain
Word2Vec feature vectors based on embeddings_sciwiki_50d

[https://zenodo.org/record/3744326#.XtViinVKh_U](https://zenodo.org/record/3744326#.XtViinVKh_U)

{:.h2_title}
## Data Source
Trained on Clinical Wikipedia Articles

{:.btn-box}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/embeddings_sciwiki_50d_es_2.5.0_2.4_1590609287349.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = WordEmbeddingsModel.pretrained("embeddings_sciwiki_50d","es","clinical/models")\
	.setInputCols("document","token")\
	.setOutputCol("word_embeddings")
```

```scala
val model = WordEmbeddingsModel.pretrained("embeddings_sciwiki_50d","es","clinical/models")
	.setInputCols("document","token")
	.setOutputCol("word_embeddings")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|---------------|------------------------|
| name          | embeddings_sciwiki_50d |
| model_class   | WordEmbeddingsModel    |
| compatibility | 2.5.0                  |
| license       | Licensed               |
| edition       | Healthcare             |
| inputs        | document, token        |
| output        | word_embeddings        |
| language      | es                     |
| dimension     | 50                     |

