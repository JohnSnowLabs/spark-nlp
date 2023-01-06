---
layout: model
title: Arabic BertForMaskedLM Large Cased model (from aubmindlab)
author: John Snow Labs
name: bert_embeddings_large_arabertv02
date: 2022-12-02
tags: [ar, open_source, bert_embeddings, bertformaskedlm]
task: Embeddings
language: ar
edition: Spark NLP 4.2.4
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForMaskedLM model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-large-arabertv02` is a Arabic model originally trained by `aubmindlab`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_large_arabertv02_ar_4.2.4_3.0_1670019689670.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

bert_loaded = BertEmbeddings.pretrained("bert_embeddings_large_arabertv02","ar") \
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
 
val bert_loaded = BertEmbeddings.pretrained("bert_embeddings_large_arabertv02","ar") 
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
|Model Name:|bert_embeddings_large_arabertv02|
|Compatibility:|Spark NLP 4.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|ar|
|Size:|1.4 GB|
|Case sensitive:|true|

## References

- https://huggingface.co/aubmindlab/bert-large-arabertv02
- https://github.com/google-research/bert
- https://arxiv.org/abs/2003.00104
- https://github.com/WissamAntoun/pydata_khobar_meetup
- http://alt.qcri.org/farasa/segmenter.html
- /aubmindlab/bert-large-arabertv02/blob/main/(https://github.com/google-research/bert/blob/master/multilingual.md)
- https://github.com/elnagara/HARD-Arabic-Dataset
- https://www.aclweb.org/anthology/D15-1299
- https://staff.aub.edu.lb/~we07/Publications/ArSentD-LEV_Sentiment_Corpus.pdf
- https://github.com/mohamedadaly/LABR
- http://curtis.ml.cmu.edu/w/courses/index.php/ANERcorp
- https://github.com/husseinmozannar/SOQAL
- https://github.com/aub-mind/arabert/blob/master/AraBERT/README.md
- https://arxiv.org/abs/2003.00104v2
- https://archive.org/details/arwiki-20190201
- https://www.semanticscholar.org/paper/1.5-billion-words-Arabic-Corpus-El-Khair/f3eeef4afb81223df96575adadf808fe7fe440b4
- https://www.aclweb.org/anthology/W19-4619
- https://sites.aub.edu.lb/mindlab/
- https://www.yakshof.com/#/
- https://www.behance.net/rahalhabib
- https://www.linkedin.com/in/wissam-antoun-622142b4/
- https://twitter.com/wissam_antoun
- https://github.com/WissamAntoun
- https://www.linkedin.com/in/fadybaly/
- https://twitter.com/fadybaly
- https://github.com/fadybaly