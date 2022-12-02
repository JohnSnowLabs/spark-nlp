---
layout: model
title: Embeddings Clinical
author: John Snow Labs
name: embeddings_clinical
class: WordEmbeddingsModel
language: en
repository: clinical/models
date: 2020-01-28
task: Embeddings
edition: Healthcare NLP 2.4.0
spark_version: 2.4
tags: [clinical,licensed,embeddings,en]
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/embeddings_clinical_en_2.4.0_2.4_1580237286004.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
model = WordEmbeddingsModel.pretrained("embeddings_clinical","en","clinical/models")\
	.setInputCols("document","token")\
	.setOutputCol("word_embeddings")
```

```scala
val model = WordEmbeddingsModel.pretrained("embeddings_clinical","en","clinical/models")
	.setInputCols("document","token")
	.setOutputCol("word_embeddings")
```


{:.nlu-block}
```python
import nlu
nlu.load("en.embed.glove.clinical").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---------------|---------------------|
| Name:          | embeddings_clinical |
| Type:   | WordEmbeddingsModel |
| Compatibility: | Spark NLP 2.4.0+               |
| License:       | Licensed            |
| Edition:       | Official          |
|Input labels:        | [document, token]     |
|Output labels:       | [word_embeddings]     |
| Language:      | en                  |
| Dimension:    | 200.0               |

{:.h2_title}
## Data Source
Trained on PubMed corpora
https://www.nlm.nih.gov/databases/download/pubmed_medline.html