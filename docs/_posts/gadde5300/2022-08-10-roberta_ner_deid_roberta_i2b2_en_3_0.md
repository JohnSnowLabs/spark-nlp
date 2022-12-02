---
layout: model
title: English RobertaForTokenClassification Cased model (from obi)
author: John Snow Labs
name: roberta_ner_deid_roberta_i2b2
date: 2022-08-10
tags: [bert, ner, open_source, en]
task: Named Entity Recognition
language: en
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: RoBertaForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RobertaForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `deid_roberta_i2b2` is a English model originally trained by `obi`.

## Predicted Entities

`DATE`, `L-AGE`, `U-PATIENT`, `L-STAFF`, `U-OTHERPHI`, `U-ID`, `EMAIL`, `U-LOC`, `L-HOSP`, `L-PATIENT`, `PATIENT`, `PHONE`, `U-PHONE`, `L-OTHERPHI`, `HOSP`, `L-PATORG`, `AGE`, `U-EMAIL`, `L-ID`, `U-HOSP`, `U-AGE`, `OTHERPHI`, `LOC`, `ID`, `U-DATE`, `L-DATE`, `U-PATORG`, `L-PHONE`, `STAFF`, `L-EMAIL`, `PATORG`, `U-STAFF`, `L-LOC`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_ner_deid_roberta_i2b2_en_4.1.0_3.0_1660139968678.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

tokenClassifier = BertForTokenClassification.pretrained("roberta_ner_deid_roberta_i2b2","en") \
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

val tokenClassifier = BertForTokenClassification.pretrained("roberta_ner_deid_roberta_i2b2","en") 
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
|Model Name:|roberta_ner_deid_roberta_i2b2|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|1.3 GB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/obi/deid_roberta_i2b2
- https://arxiv.org/pdf/1907.11692.pdf
- https://github.com/obi-ml-public/ehr_deidentification/tree/master/steps/train
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4978170/
- https://www.hhs.gov/hipaa/for-professionals/privacy/laws-regulations/index.html
- https://github.com/obi-ml-public/ehr_deidentification
- https://github.com/obi-ml-public/ehr_deidentification/tree/master/steps/forward_pass
- https://github.com/obi-ml-public/ehr_deidentification/blob/master/AnnotationGuidelines.md