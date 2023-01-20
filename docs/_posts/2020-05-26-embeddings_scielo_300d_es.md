---
layout: model
title: Embeddings Scielo 300 dims
author: John Snow Labs
name: embeddings_scielo_300d
class: WordEmbeddingsModel
language: es
repository: clinical/models
date: 2020-05-26
task: Embeddings
edition: Healthcare NLP 2.5.0
spark_version: 2.4
tags: [clinical,embeddings,es]
supported: true
annotator: WordEmbeddingsModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Word Embeddings lookup annotator that maps tokens to vectors.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/embeddings_scielo_300d_es_2.5.0_2.4_1590467138742.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/embeddings_scielo_300d_es_2.5.0_2.4_1590467138742.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
model = WordEmbeddingsModel.pretrained("embeddings_scielo_300d","es","clinical/models")\
	.setInputCols(["document","token"])\
	.setOutputCol("word_embeddings")
```

```scala
val model = WordEmbeddingsModel.pretrained("embeddings_scielo_300d","es","clinical/models")
	.setInputCols("document","token")
	.setOutputCol("word_embeddings")
```


{:.nlu-block}
```python
import nlu
nlu.load("es.embed.scielo.300d").predict("""Put your text here.""")
```

</div>

{:.h2_title}
## Results 
Word2Vec feature vectors based on ``embeddings_scielo_300d``.

{:.model-param}
## Model Information

{:.table-model}
|---------------|------------------------|
| Name:          | embeddings_scielo_300d |
| Type:   | WordEmbeddingsModel    |
| Compatibility: | Spark NLP 2.5.0+                 |
| License:       | Licensed               |
| Edition:       | Official             |
|Input labels:        | [document, token]        |
|Output labels:       | [word_embeddings]        |
| Language:      | es                     |
| Dimension:    | 300.0                  |

{:.h2_title}
## Data Source
Trained on Scielo Articles
https://zenodo.org/record/3744326#.XtViinVKh_U