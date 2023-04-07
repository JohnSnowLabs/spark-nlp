---
layout: model
title: English DistilBertForTokenClassification Cased model (from whispAI)
author: John Snow Labs
name: distilbert_token_classifier_directquote_sentlevel_distilbert
date: 2023-03-19
tags: [en, open_source, distilbert, token_classification, ner, tensorflow]
task: Named Entity Recognition
language: en
edition: Spark NLP 4.3.1
spark_version: 3.0
supported: true
engine: tensorflow
annotator: DistilBertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `DirectQuote-SentLevel-DistilBERT` is a English model originally trained by `whispAI`.

## Predicted Entities

`LeftSpeaker`, `Out`, `Speaker`, `RightSpeaker`, `Unknown`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_token_classifier_directquote_sentlevel_distilbert_en_4.3.1_3.0_1679228598544.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_token_classifier_directquote_sentlevel_distilbert_en_4.3.1_3.0_1679228598544.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier = DistilBertForTokenClassification.pretrained("distilbert_token_classifier_directquote_sentlevel_distilbert","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, tokenClassifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
    .setInputCols(Array("text")) 
    .setOutputCols(Array("document"))
      
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")
 
val tokenClassifier = DistilBertForTokenClassification.pretrained("distilbert_token_classifier_directquote_sentlevel_distilbert","en") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("ner")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_token_classifier_directquote_sentlevel_distilbert|
|Compatibility:|Spark NLP 4.3.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|244.1 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/whispAI/DirectQuote-SentLevel-DistilBERT
- #quote-extraction--attribution-on-directquotehttpsarxivorgabs211007827-dataset-with-bert-based-token-classification-💬
- https://arxiv.org/abs/2110.07827
- https://arxiv.org/abs/2110.07827
- https://arxiv.org/abs/2110.07827
- https://www.theguardian.com/info/2021/nov/25/talking-sense-using-machine-learning-to-understand-quotes
- https://arxiv.org/abs/2110.07827
- https://stanfordnlp.github.io/CoreNLP/quote.html
- https://textacy.readthedocs.io/en/latest/api_reference/extract.html#textacy.extract.triples.direct_quotations
- https://stanfordnlp.github.io/CoreNLP/quote.html
- https://arxiv.org/abs/2110.07827
- https://textacy.readthedocs.io/en/latest/api_reference/extract.html#textacy.extract.triples.direct_quotations