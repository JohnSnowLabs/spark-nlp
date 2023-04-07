---
layout: model
title: English BertForSequenceClassification Base Uncased model (from bhadresh-savani)
author: John Snow Labs
name: bert_sequence_classifier_base_uncased_emotion
date: 2023-03-16
tags: [en, open_source, bert, sequence_classification, ner, tensorflow]
task: Named Entity Recognition
language: en
edition: Spark NLP 4.3.1
spark_version: 3.0
supported: true
engine: tensorflow
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-base-uncased-emotion` is a English model originally trained by `bhadresh-savani`.

## Predicted Entities

`love`, `joy`, `fear`, `sadness`, `anger`, `surprise`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_base_uncased_emotion_en_4.3.1_3.0_1678949243504.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_base_uncased_emotion_en_4.3.1_3.0_1678949243504.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_base_uncased_emotion","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, sequenceClassifier])

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
 
val sequenceClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_base_uncased_emotion","en") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("ner")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, sequenceClassifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_base_uncased_emotion|
|Compatibility:|Spark NLP 4.3.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|410.0 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/bhadresh-savani/bert-base-uncased-emotion
- https://arxiv.org/abs/1810.04805
- https://github.com/bhadreshpsavani/ExploringSentimentalAnalysis/blob/main/SentimentalAnalysisWithDistilbert.ipynb
- https://learning.oreilly.com/library/view/natural-language-processing/9781098103231/
- https://paperswithcode.com/sota?task=Text+Classification&dataset=emotion