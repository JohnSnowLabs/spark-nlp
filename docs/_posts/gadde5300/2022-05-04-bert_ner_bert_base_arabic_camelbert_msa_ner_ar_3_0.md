---
layout: model
title: Arabic Named Entity Recognition (Modern Standard Arabic-MSA)
author: John Snow Labs
name: bert_ner_bert_base_arabic_camelbert_msa_ner
date: 2022-05-04
tags: [bert, ner, token_classification, ar, open_source]
task: Named Entity Recognition
language: ar
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: BertForTokenClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Named Entity Recognition model, uploaded to Hugging Face, adapted and imported into Spark NLP. `bert-base-arabic-camelbert-msa-ner` is a Arabic model orginally trained by `CAMeL-Lab`.

## Predicted Entities

`ORG`, `LOC`, `PERS`, `MISC`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_ner_bert_base_arabic_camelbert_msa_ner_ar_3.4.2_3.0_1651630233283.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

tokenClassifier = BertForTokenClassification.pretrained("bert_ner_bert_base_arabic_camelbert_msa_ner","ar") \
.setInputCols(["sentence", "token"]) \
.setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier])

data = spark.createDataFrame([["أنا أحب الشرارة NLP"]]).toDF("text")

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

val tokenClassifier = BertForTokenClassification.pretrained("bert_ner_bert_base_arabic_camelbert_msa_ner","ar") 
.setInputCols(Array("sentence", "token")) 
.setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector, tokenizer, tokenClassifier))

val data = Seq("أنا أحب الشرارة NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("ar.ner.arabic_camelbert_msa_ner").predict("""أنا أحب الشرارة NLP""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_ner_bert_base_arabic_camelbert_msa_ner|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|ar|
|Size:|407.1 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-msa-ner
- https://camel.abudhabi.nyu.edu/anercorp/
- https://arxiv.org/abs/2103.06678
- https://github.com/CAMeL-Lab/CAMeLBERT
- https://github.com/CAMeL-Lab/camel_tools
- https://github.com/CAMeL-Lab/camel_tools