---
layout: model
title: Sentence Embeddings - Bluebert uncased (MedNLI)
author: John Snow Labs
name: sbluebert_base_uncased_mli
date: 2020-11-27
task: Embeddings
language: en
edition: Healthcare NLP 2.6.4
spark_version: 2.4
tags: [embeddings, en, licensed]
supported: true
annotator: BertSentenceEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description

This model is trained to generate contextual sentence embeddings of input sentences. It has been fine-tuned on MedNLI dataset to provide sota performance on STS and SentEval Benchmarks.


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbluebert_base_uncased_mli_en_2.6.4_2.4_1606228596089.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use

Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, BertSentenceEmbeddings. The output of this model can be used in tasks like NER, Classification, Entity Resolution etc.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
sbiobert_embeddings = BertSentenceEmbeddings\
.pretrained("sbluebert_base_uncased_mli","en","clinical/models")\
.setInputCols(["ner_chunk_doc"])\
.setOutputCol("sbert_embeddings")

```

```scala

val sbiobert_embeddings = BertSentenceEmbeddings.pretrained("sbluebert_base_uncased_mli","en","clinical/models")
.setInputCols(Array("ner_chunk_doc"))
.setOutputCol("sbert_embeddings")

```



{:.nlu-block}
```python
import nlu
nlu.load("en.embed_sentence.bluebert.mli").predict("""Put your text here.""")
```

</div>

{:.h2_title}
## Results
Gives a 768 dimensional vector representation of the sentence.

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbluebert_base_uncased_mli|
|Type:|BertSentenceEmbeddings|
|Compatibility:|Spark NLP 2.6.4 +|
|Edition:|Official|
|License:|Licensed|
|Input Labels:|[ner_chunk]|
|Output Labels:|[sentence_embeddings]|
|Language:|[en]|
|Case sensitive:|false|

{:.h2_title}
## Data Source
Tuned on MedNLI dataset using Bluebert weights.

