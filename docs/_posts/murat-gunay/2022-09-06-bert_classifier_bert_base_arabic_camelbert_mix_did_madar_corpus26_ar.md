---
layout: model
title: Arabic BertForSequenceClassification Base Cased model (from CAMeL-Lab)
author: John Snow Labs
name: bert_classifier_bert_base_arabic_camelbert_mix_did_madar_corpus26
date: 2022-09-06
tags: [ar, open_source, bert, sequence_classification, classification]
task: Text Classification
language: ar
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-base-arabic-camelbert-mix-did-madar-corpus26` is a Arabic model originally trained by `CAMeL-Lab`.

## Predicted Entities

`MUS`, `BEI`, `DAM`, `AMM`, `SFX`, `TUN`, `RAB`, `ALX`, `RIY`, `ALE`, `TRI`, `CAI`, `JER`, `ASW`, `SAN`, `ALG`, `BAG`, `SAL`, `MOS`, `FES`, `BAS`, `DOH`, `JED`, `KHA`, `MSA`, `BEN`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_bert_base_arabic_camelbert_mix_did_madar_corpus26_ar_4.1.0_3.0_1662507353689.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_classifier_bert_base_arabic_camelbert_mix_did_madar_corpus26_ar_4.1.0_3.0_1662507353689.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_bert_base_arabic_camelbert_mix_did_madar_corpus26","ar") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, seq_classifier])

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
 
val seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_bert_base_arabic_camelbert_mix_did_madar_corpus26","ar") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, seq_classifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("ar.classify.mix_did_madar_corpus26.bert.base.by_camel_lab").predict("""PUT YOUR STRING HERE""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_classifier_bert_base_arabic_camelbert_mix_did_madar_corpus26|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|ar|
|Size:|409.6 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-mix-did-madar-corpus26
- https://camel.abudhabi.nyu.edu/madar-shared-task-2019/
- https://arxiv.org/abs/2103.06678
- https://github.com/CAMeL-Lab/CAMeLBERT
- https://github.com/CAMeL-Lab/camel_tools