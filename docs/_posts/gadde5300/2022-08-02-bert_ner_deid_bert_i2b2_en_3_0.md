---
layout: model
title: English BertForTokenClassification Cased model (from obi)
author: John Snow Labs
name: bert_ner_deid_bert_i2b2
date: 2022-08-02
tags: [bert, ner, open_source, en]
task: Named Entity Recognition
language: en
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `deid_bert_i2b2` is a English model originally trained by `obi`.

## Predicted Entities

`L-HOSP`, `L-DATE`, `L-AGE`, `HOSP`, `DATE`, `PATIENT`, `U-DATE`, `PHONE`, `U-HOSP`, `ID`, `U-LOC`, `U-OTHERPHI`, `U-ID`, `U-PATIENT`, `U-EMAIL`, `U-PHONE`, `LOC`, `L-EMAIL`, `U-PATORG`, `L-PHONE`, `EMAIL`, `AGE`, `L-PATIENT`, `L-OTHERPHI`, `L-LOC`, `U-STAFF`, `L-PATORG`, `L-STAFF`, `PATORG`, `U-AGE`, `L-ID`, `OTHERPHI`, `STAFF`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_ner_deid_bert_i2b2_en_4.1.0_3.0_1659456176674.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

tokenClassifier = BertForTokenClassification.pretrained("bert_ner_deid_bert_i2b2","en") \
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

val tokenClassifier = BertForTokenClassification.pretrained("bert_ner_deid_bert_i2b2","en") 
    .setInputCols(Array("sentence", "token")) 
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector, tokenizer, tokenClassifier))

val data = Seq("PUT YOUR STRING HERE").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_ner_deid_bert_i2b2|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|404.0 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/obi/deid_bert_i2b2
- https://github.com/obi-ml-public/ehr_deidentification/tree/master/steps/forward_pass
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4978170/
- https://arxiv.org/pdf/1904.03323.pdf
- https://github.com/obi-ml-public/ehr_deidentification/tree/master/steps/train
- https://github.com/obi-ml-public/ehr_deidentification/blob/master/AnnotationGuidelines.md
- https://www.hhs.gov/hipaa/for-professionals/privacy/laws-regulations/index.html
- https://github.com/obi-ml-public/ehr_deidentification