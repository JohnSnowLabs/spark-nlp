---
layout: model
title: Multilingual full_5epoch_bert_base_multilingual_finetuned_cefr_ner_60000news BertForTokenClassification from DioBot2000
author: John Snow Labs
name: full_5epoch_bert_base_multilingual_finetuned_cefr_ner_60000news
date: 2025-01-25
tags: [xx, open_source, onnx, token_classification, bert, ner]
task: Named Entity Recognition
language: xx
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: onnx
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`full_5epoch_bert_base_multilingual_finetuned_cefr_ner_60000news` is a Multilingual model originally trained by DioBot2000.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/full_5epoch_bert_base_multilingual_finetuned_cefr_ner_60000news_xx_5.5.1_3.0_1737844353473.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/full_5epoch_bert_base_multilingual_finetuned_cefr_ner_60000news_xx_5.5.1_3.0_1737844353473.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
     
documentAssembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')
    
tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

tokenClassifier  = BertForTokenClassification.pretrained("full_5epoch_bert_base_multilingual_finetuned_cefr_ner_60000news","xx") \
     .setInputCols(["documents","token"]) \
     .setOutputCol("ner")

pipeline = Pipeline().setStages([documentAssembler, tokenizer, tokenClassifier])
data = spark.createDataFrame([["I love spark-nlp"]]).toDF("text")
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)

```
```scala

val documentAssembler = new DocumentAssembler()
    .setInputCols("text")
    .setOutputCols("document")
    
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val tokenClassifier = BertForTokenClassification.pretrained("full_5epoch_bert_base_multilingual_finetuned_cefr_ner_60000news", "xx")
    .setInputCols(Array("documents","token")) 
    .setOutputCol("ner") 
    
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier))
val data = Seq("I love spark-nlp").toDS.toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|full_5epoch_bert_base_multilingual_finetuned_cefr_ner_60000news|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|xx|
|Size:|665.1 MB|

## References

https://huggingface.co/DioBot2000/Full-5epoch-BERT-base-multilingual-finetuned-CEFR_ner-60000news