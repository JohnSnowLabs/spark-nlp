---
layout: model
title: English cold_fusion_bert_base_uncased_itr23_seed0_finetuned_convincingness_ibm BertForSequenceClassification from jakub014
author: John Snow Labs
name: cold_fusion_bert_base_uncased_itr23_seed0_finetuned_convincingness_ibm
date: 2023-10-27
tags: [bert, en, open_source, sequence_classification, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.1.4
spark_version: 3.4
supported: true
engine: onnx
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`cold_fusion_bert_base_uncased_itr23_seed0_finetuned_convincingness_ibm` is a English model originally trained by jakub014.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/cold_fusion_bert_base_uncased_itr23_seed0_finetuned_convincingness_ibm_en_5.1.4_3.4_1698388674044.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/cold_fusion_bert_base_uncased_itr23_seed0_finetuned_convincingness_ibm_en_5.1.4_3.4_1698388674044.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

tokenizer = Tokenizer()\
    .setInputCols("document")\
    .setOutputCol("token")  
    
sequenceClassifier = BertForSequenceClassification.pretrained("cold_fusion_bert_base_uncased_itr23_seed0_finetuned_convincingness_ibm","en")\
            .setInputCols(["document","token"])\
            .setOutputCol("class")

pipeline = Pipeline().setStages([document_assembler, tokenizer, sequenceClassifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)

```
```scala

val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols("document") 
    .setOutputCol("token")  
    
val sequenceClassifier = BertForSequenceClassification.pretrained("cold_fusion_bert_base_uncased_itr23_seed0_finetuned_convincingness_ibm","en")
            .setInputCols(Array("document","token"))
            .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, sequenceClassifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)


```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|cold_fusion_bert_base_uncased_itr23_seed0_finetuned_convincingness_ibm|
|Compatibility:|Spark NLP 5.1.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|409.4 MB|

## References

https://huggingface.co/jakub014/ColD-Fusion-bert-base-uncased-itr23-seed0-finetuned-convincingness-IBM