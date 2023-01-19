---
layout: model
title: Embeddings Healthcare 100 dims
author: John Snow Labs
name: embeddings_healthcare_100d
class: WordEmbeddingsModel
language: en
repository: clinical/models
date: 2020-05-29
task: Embeddings
edition: Healthcare NLP 2.5.0
spark_version: 2.4
tags: [clinical,embeddings,en]
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/embeddings_healthcare_100d_en_2.5.0_2.4_1590794626292.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/embeddings_healthcare_100d_en_2.5.0_2.4_1590794626292.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
model = WordEmbeddingsModel.pretrained("embeddings_healthcare_100d","en","clinical/models")\
	.setInputCols(["document","token"])\
	.setOutputCol("word_embeddings")
```

```scala
val model = WordEmbeddingsModel.pretrained("embeddings_healthcare_100d","en","clinical/models")
	.setInputCols("document","token")
	.setOutputCol("word_embeddings")
```


{:.nlu-block}
```python
import nlu
nlu.load("en.embed.glove.healthcare_100d").predict("""Put your text here.""")
```

</div>

{:.h2_title}
## Results 
Word2Vec feature vectors based on ``embeddings_healthcare_100d``.

{:.model-param}
## Model Information

{:.table-model}
|---------------|----------------------------|
| Name:          | embeddings_healthcare_100d |
| Type:   | WordEmbeddingsModel        |
| Compatibility: | Spark NLP 2.5.0+                     |
| License:       | Licensed                   |
| Edition:       | Official                 |
|Input labels:        | [document, token]            |
|Output labels:       | [word_embeddings]            |
| Language:      | en                         |
| Dimension:    | 100.0                      |

{:.h2_title}
## Data Source
Trained on PubMed + ICD10 + UMLS + MIMIC III corpora
https://www.nlm.nih.gov/databases/download/pubmed_medline.html