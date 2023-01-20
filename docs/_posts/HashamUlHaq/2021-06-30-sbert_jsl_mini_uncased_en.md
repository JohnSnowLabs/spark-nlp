---
layout: model
title: Sentence Embeddings - sbert mini (tuned)
author: John Snow Labs
name: sbert_jsl_mini_uncased
date: 2021-06-30
tags: [embeddings, clinical, licensed, en]
task: Embeddings
language: en
edition: Healthcare NLP 3.1.0
spark_version: 2.4
supported: true
annotator: BertSentenceEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


This model is trained to generate contextual sentence embeddings of input sentences.


{:.btn-box}
[Live Demo](https://nlp.johnsnowlabs.com/demo){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbert_jsl_mini_uncased_en_3.1.0_2.4_1625050221194.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/sbert_jsl_mini_uncased_en_3.1.0_2.4_1625050221194.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
sbiobert_embeddings = BertSentenceEmbeddings.pretrained("sbert_jsl_mini_uncased","en","clinical/models").setInputCols(["sentence"]).setOutputCol("sbert_embeddings")
```
```scala
val sbiobert_embeddings = BertSentenceEmbeddings
.pretrained("sbert_jsl_mini_uncased","en","clinical/models")
.setInputCols(Array("sentence"))
.setOutputCol("sbert_embeddings")
```


{:.nlu-block}
```python
import nlu
nlu.load("en.embed_sentence.bert.jsl_mini_uncased").predict("""Put your text here.""")
```

</div>


## Results


```bash
Gives a 768 dimensional vector representation of the sentence.
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|sbert_jsl_mini_uncased|
|Compatibility:|Healthcare NLP 3.1.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Case sensitive:|false|


## Data Source


Tuned on MedNLI dataset


## Benchmarking

```bash
MedNLI    Score
Acc       0.663
STS(cos)  0.701
```

<!--stackedit_data:
eyJoaXN0b3J5IjpbOTI1ODE2NjIwXX0=
-->