---
layout: model
title: English RobertaForSequenceClassification Large Cased model
author: John Snow Labs
name: roberta_classifier_large_mnli
date: 2022-09-09
tags: [en, open_source, roberta, sequence_classification, classification]
task: Text Classification
language: en
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: RoBertaForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RobertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `roberta-large-mnli` is a English model originally trained by HuggingFace.

## Predicted Entities

`ENTAILMENT`, `NEUTRAL`, `CONTRADICTION`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_classifier_large_mnli_en_4.1.0_3.0_1662766951816.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_classifier_large_mnli_en_4.1.0_3.0_1662766951816.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_large_mnli","en") \
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
 
val seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_large_mnli","en") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("class")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, seq_classifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_classifier_large_mnli|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|846.8 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/roberta-large-mnli
- https://github.com/facebookresearch/fairseq/tree/main/examples/roberta
- https://arxiv.org/abs/1907.11692
- https://github.com/facebookresearch/fairseq/tree/main/examples/roberta
- https://github.com/facebookresearch/fairseq/tree/main/examples/roberta
- https://aclanthology.org/2021.acl-long.330.pdf
- https://dl.acm.org/doi/pdf/10.1145/3442188.3445922
- https://cims.nyu.edu/~sbowman/multinli/
- https://yknzhu.wixsite.com/mbweb
- https://en.wikipedia.org/wiki/English_Wikipedia
- https://commoncrawl.org/2016/10/news-dataset-available/
- https://github.com/jcpeterson/openwebtext
- https://arxiv.org/abs/1806.02847
- https://github.com/facebookresearch/fairseq/tree/main/examples/roberta
- https://arxiv.org/pdf/1804.07461.pdf
- https://cims.nyu.edu/~sbowman/multinli/
- https://arxiv.org/pdf/1804.07461.pdf
- https://arxiv.org/pdf/1804.07461.pdf
- https://arxiv.org/abs/1704.05426
- https://arxiv.org/abs/1508.05326
- https://arxiv.org/pdf/1809.05053.pdf
- https://cims.nyu.edu/~sbowman/multinli/
- https://arxiv.org/pdf/1809.05053.pdf
- https://mlco2.github.io/impact#compute
- https://arxiv.org/abs/1910.09700
- https://arxiv.org/pdf/1907.11692.pdf
- https://arxiv.org/pdf/1907.11692.pdf