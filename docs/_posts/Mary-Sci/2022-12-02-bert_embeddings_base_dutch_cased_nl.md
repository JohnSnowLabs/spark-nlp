---
layout: model
title: Dutch BertForMaskedLM Base Cased model (from GroNLP)
author: John Snow Labs
name: bert_embeddings_base_dutch_cased
date: 2022-12-02
tags: [nl, open_source, bert_embeddings, bertformaskedlm]
task: Embeddings
language: nl
edition: Spark NLP 4.2.4
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForMaskedLM model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-base-dutch-cased` is a Dutch model originally trained by `GroNLP`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_base_dutch_cased_nl_4.2.4_3.0_1670016541889.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols(["text"]) \
    .setOutputCols("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

bert_loaded = BertEmbeddings.pretrained("bert_embeddings_base_dutch_cased","nl") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings") \
    .setCaseSensitive(True)
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, bert_loaded])

data = spark.createDataFrame([["I love Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
    .setInputCols(Array("text")) 
    .setOutputCols(Array("document"))
      
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")
 
val bert_loaded = BertEmbeddings.pretrained("bert_embeddings_base_dutch_cased","nl") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("embeddings")
    .setCaseSensitive(True)    
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, bert_loaded))

val data = Seq("I love Spark NLP").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_embeddings_base_dutch_cased|
|Compatibility:|Spark NLP 4.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|nl|
|Size:|409.5 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/GroNLP/bert-base-dutch-cased
- https://www.semanticscholar.org/author/Wietse-de-Vries/144611157
- https://www.semanticscholar.org/author/Andreas-van-Cranenburgh/2791585
- https://www.semanticscholar.org/author/Arianna-Bisazza/3242253
- https://www.semanticscholar.org/author/Tommaso-Caselli/1864635
- https://www.semanticscholar.org/author/Gertjan-van-Noord/143715131
- https://www.semanticscholar.org/author/M.-Nissim/2742475
- https://arxiv.org/abs/1912.09582
- https://github.com/wietsedv/bertje
- https://www.semanticscholar.org/paper/BERTje%3A-A-Dutch-BERT-Model-Vries-Cranenburgh/a4d5e425cac0bf84c86c0c9f720b6339d6288ffa
- https://www.clips.uantwerpen.be/conll2002/ner/
- https://ivdnt.org/downloads/taalmaterialen/tstc-sonar-corpus
- https://github.com/google-research/bert/blob/master/multilingual.md
- http://textdata.nl
- https://github.com/iPieter/RobBERT
- https://universaldependencies.org/treebanks/nl_lassysmall/index.html
- https://github.com/google-research/bert/blob/master/multilingual.md
- http://textdata.nl
- https://github.com/iPieter/RobBERT