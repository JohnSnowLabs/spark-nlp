---
layout: model
title: Sentence Embeddings - Biobert cased (MedNLI)
author: John Snow Labs
name: sbiobert_base_cased_mli
date: 2020-11-27
tags: [embeddings, en, licensed]
article_header:
    type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is trained to generate contextual sentence embeddings of input sentences. It has been fine-tuned on MedNLI dataset to provide sota performance on STS and SentEval Benchmarks.

## Predicted Entities 

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobert_base_cased_mli_en_2.6.4_2.4_1606225728763.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, BertSentenceEmbeddings. The output of this model can be used in tasks like NER, Classification, Entity Resolution etc.
    
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
sbiobert_embeddins = BertSentenceEmbeddings\
     .pretrained("sbiobert_base_cased_mli",'en','clinical/models')\
     .setInputCols(["ner_chunk_doc"])\
     .setOutputCol("sbert_embeddings")

```

```scala

val ner = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli",'en','clinical/models')
    .setInputCols(["ner_chunk_doc"])
    .setOutputCol("sbert_embeddings")

```

</div>

{:.h2_title}
## Results
Gives a 768 dimensional vector representation of the sentence.

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobert_base_cased_mli|
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
Tuned on MedNLI dataset using Biobert weights.

