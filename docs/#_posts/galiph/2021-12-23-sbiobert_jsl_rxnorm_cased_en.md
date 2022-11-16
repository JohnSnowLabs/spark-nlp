---
layout: model
title: Sentence Embeddings - sbert medium (tuned)
author: John Snow Labs
name: sbiobert_jsl_rxnorm_cased
date: 2021-12-23
tags: [licensed, embeddings, clinical, en]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.3.4
spark_version: 2.4
supported: true
annotator: BertSentenceEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps sentences & documents to a 768 dimensional dense vector space by using average pooling on top of BioBert model. It's also fine-tuned on RxNorm dataset to help generalization over medication-related datasets.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobert_jsl_rxnorm_cased_en_3.3.4_2.4_1640271525048.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
sentence_embeddings = BertSentenceEmbeddings.pretrained("sbiobert_jsl_rxnorm_cased", "en", "clinical/models")\
.setInputCols(["sentence"])\
.setOutputCol("sbioert_embeddings")
```
```scala
val sentence_embeddings = BertSentenceEmbeddings.pretrained('sbiobert_jsl_rxnorm_cased', 'en','clinical/models')\
.setInputCols("sentence")\
.setOutputCol("sbioert_embeddings")
```


{:.nlu-block}
```python
import nlu
nlu.load("en.embed_sentence.biobert.rxnorm").predict("""Put your text here.""")
```

</div>

## Results

```bash
Gives a 768-dimensional vector representation of the sentence.
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobert_jsl_rxnorm_cased|
|Compatibility:|Healthcare NLP 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|402.0 MB|