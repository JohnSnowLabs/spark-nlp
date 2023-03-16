---
layout: model
title: Chinese BertForSequenceClassification Cased model (from IDEA-CCNL)
author: John Snow Labs
name: bert_sequence_classifier_erlangshen_roberta_110m_similarity
date: 2023-03-16
tags: [zh, open_source, bert, sequence_classification, ner, tensorflow]
task: Named Entity Recognition
language: zh
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

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `Erlangshen-Roberta-110M-Similarity` is a Chinese model originally trained by `IDEA-CCNL`.

## Predicted Entities

`not similar`, `similar`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_erlangshen_roberta_110m_similarity_zh_4.3.1_3.0_1678948652034.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_erlangshen_roberta_110m_similarity_zh_4.3.1_3.0_1678948652034.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_erlangshen_roberta_110m_similarity","zh") \
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
 
val sequenceClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_erlangshen_roberta_110m_similarity","zh") 
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
|Model Name:|bert_sequence_classifier_erlangshen_roberta_110m_similarity|
|Compatibility:|Spark NLP 4.3.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|zh|
|Size:|383.9 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-110M-Similarity
- https://github.com/IDEA-CCNL/Fengshenbang-LM
- https://fengshenbang-doc.readthedocs.io/
- https://arxiv.org/abs/2209.02970
- https://arxiv.org/abs/2209.02970
- https://github.com/IDEA-CCNL/Fengshenbang-LM/
- https://github.com/IDEA-CCNL/Fengshenbang-LM/