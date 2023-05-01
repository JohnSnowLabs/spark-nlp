---
layout: model
title: Chinese Named Entity Recognition (from uer)
author: John Snow Labs
name: bert_ner_roberta_base_finetuned_cluener2020_chinese
date: 2022-05-09
tags: [bert, ner, token_classification, zh, open_source]
task: Named Entity Recognition
language: zh
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Named Entity Recognition model, uploaded to Hugging Face, adapted and imported into Spark NLP. `roberta-base-finetuned-cluener2020-chinese` is a Chinese model orginally trained by `uer`.

## Predicted Entities

`position`, `company`, `address`, `movie`, `organization`, `game`, `name`, `book`, `government`, `scene`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_ner_roberta_base_finetuned_cluener2020_chinese_zh_3.4.2_3.0_1652096419329.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_ner_roberta_base_finetuned_cluener2020_chinese_zh_3.4.2_3.0_1652096419329.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
       .setInputCols(["document"])\
       .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols("sentence") \
    .setOutputCol("token")

tokenClassifier = BertForTokenClassification.pretrained("bert_ner_roberta_base_finetuned_cluener2020_chinese","zh") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
          .setInputCol("text") 
          .setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
       .setInputCols(Array("document"))
       .setOutputCol("sentence")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("sentence"))
    .setOutputCol("token")

val tokenClassifier = BertForTokenClassification.pretrained("bert_ner_roberta_base_finetuned_cluener2020_chinese","zh") 
    .setInputCols(Array("sentence", "token")) 
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector, tokenizer, tokenClassifier))

val data = Seq("PUT YOUR STRING HERE").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("zh.ner.bert.base_finetuned").predict("""PUT YOUR STRING HERE""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_ner_roberta_base_finetuned_cluener2020_chinese|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|zh|
|Size:|381.5 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/uer/roberta-base-finetuned-cluener2020-chinese
- https://github.com/dbiir/UER-py/wiki/Modelzoo
- https://github.com/CLUEbenchmark/CLUENER2020
- https://github.com/dbiir/UER-py/
- https://cloud.tencent.com/