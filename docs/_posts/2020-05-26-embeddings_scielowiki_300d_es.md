---
layout: model
title: Embeddings Scielowiki 300 dims
author: John Snow Labs
name: embeddings_scielowiki_300d
class: WordEmbeddingsModel
language: es
repository: clinical/models
date: 2020-05-26
tags: [clinical,embeddings,scielo,wikipedia,es]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Word Embeddings lookup annotator that maps tokens to vectors  


{:.h2_title}
## Prediction Labels
Word2Vec feature vectors based on embeddings_scielowiki_300d

[https://zenodo.org/record/3744326#.XtViinVKh_U](https://zenodo.org/record/3744326#.XtViinVKh_U)

{:.h2_title}
## Data Source
Trained on Scielo Articles + Clinical Wikipedia Articles

{:.btn-box}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/embeddings_scielowiki_300d_es_2.5.0_2.4_1590467643391.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = WordEmbeddingsModel.pretrained("embeddings_scielowiki_300d","es","clinical/models") \
	.setInputCols("document","token") \
	.setOutputCol("word_embeddings")
```

```scala
val model = WordEmbeddingsModel.pretrained("embeddings_scielowiki_300d","es","clinical/models")
	.setInputCols("document","token")
	.setOutputCol("word_embeddings")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|---------------|----------------------------|
| Model Name    | embeddings_scielowiki_300d |
| Type          | WordEmbeddingsModel        |
| Compatibility | 2.5.0                      |
| License       | Licensed                   |
| Edition       | Healthcare                 |
| Inputs        | document, token            |
| Output        | word_embeddings            |
| Language      | es                         |
| Dimension     | 300                        |

