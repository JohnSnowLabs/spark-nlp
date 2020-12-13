---
layout: model
title: Embeddings BioVec
author: John Snow Labs
name: embeddings_biovec
class: WordEmbeddingsModel
language: en
repository: clinical/models
date: 2020-06-02
tags: [clinical,embeddings,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Word Embeddings lookup annotator that maps tokens to vectors

## Predicted Entities 
Word2Vec feature vectors based on embeddings_biovec

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/embeddings_biovec_en_2.5.0_2.4_1591068211397.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = WordEmbeddingsModel.pretrained("embeddings_biovec","en","clinical/models")\
	.setInputCols("document","token")\
	.setOutputCol("word_embeddings")
```

```scala
val model = WordEmbeddingsModel.pretrained("embeddings_biovec","en","clinical/models")
	.setInputCols("document","token")
	.setOutputCol("word_embeddings")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---------------|---------------------|
| Name:          | embeddings_biovec   |
| Type:   | WordEmbeddingsModel |
| Compatibility: | Spark NLP 2.5.0+              |
| License:       | Licensed            |
| Edition:       | Official          |
|Input labels:        | [document, token]     |
|Output labels:       | [word_embeddings]     |
| Language:      | en                  |
| Dimension:    | 300.0               |

{:.h2_title}
## Data Source
Trained on PubMed corpora
https://github.com/ncbi-nlp/BioSentVec