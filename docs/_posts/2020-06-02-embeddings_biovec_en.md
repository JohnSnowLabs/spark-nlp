---
layout: model
title: Embeddings BioVec
author: John Snow Labs
name: embeddings_biovec
class: WordEmbeddingsModel
language: en
repository: clinical/models
date: 2020-06-02
tags: [clinical,embeddings,pubmed,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Word Embeddings lookup annotator that maps tokens to vectors  


{:.h2_title}
## Prediction Labels
Word2Vec feature vectors based on embeddings_biovec

[https://github.com/ncbi-nlp/BioSentVec](https://github.com/ncbi-nlp/BioSentVec)

{:.h2_title}
## Data Source
Trained on PubMed corpora

{:.btn-box}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/embeddings_biovec_en_2.5.0_2.4_1591068211397.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = WordEmbeddingsModel.pretrained("embeddings_biovec","en","clinical/models") \
	.setInputCols("document","token") \
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
| Model Name    | embeddings_biovec   |
| Type          | WordEmbeddingsModel |
| Compatibility | 2.5.0               |
| License       | Licensed            |
| Edition       | Healthcare          |
| Inputs        | document, token     |
| Output        | word_embeddings     |
| Language      | en                  |
| Dimension     | 300                 |

