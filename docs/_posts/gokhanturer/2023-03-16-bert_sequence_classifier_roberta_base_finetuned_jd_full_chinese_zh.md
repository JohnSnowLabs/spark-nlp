---
layout: model
title: Chinese BertForSequenceClassification Base Cased model (from uer)
author: John Snow Labs
name: bert_sequence_classifier_roberta_base_finetuned_jd_full_chinese
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

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `roberta-base-finetuned-jd-full-chinese` is a Chinese model originally trained by `uer`.

## Predicted Entities

`star 4`, `star 5`, `star 1`, `star 2`, `star 3`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_roberta_base_finetuned_jd_full_chinese_zh_4.3.1_3.0_1678984192277.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_roberta_base_finetuned_jd_full_chinese_zh_4.3.1_3.0_1678984192277.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_roberta_base_finetuned_jd_full_chinese","zh") \
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
 
val sequenceClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_roberta_base_finetuned_jd_full_chinese","zh") 
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
|Model Name:|bert_sequence_classifier_roberta_base_finetuned_jd_full_chinese|
|Compatibility:|Spark NLP 4.3.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|zh|
|Size:|383.6 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/uer/roberta-base-finetuned-jd-full-chinese
- https://arxiv.org/abs/1909.05658
- https://github.com/dbiir/UER-py/wiki/Modelzoo
- https://github.com/zhangxiangxiao/glyph
- https://arxiv.org/abs/1708.02657
- https://github.com/dbiir/UER-py/
- https://cloud.tencent.com/