---
layout: model
title: English vg055_roberta_base_bne_finetuned_analisis_sentimiento_textos_turisticos_mx_polaridad RoBertaForSequenceClassification from hackathon-somos-nlp-2023
author: John Snow Labs
name: vg055_roberta_base_bne_finetuned_analisis_sentimiento_textos_turisticos_mx_polaridad
date: 2024-09-11
tags: [roberta, en, open_source, sequence_classification, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
engine: onnx
annotator: RoBertaForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`vg055_roberta_base_bne_finetuned_analisis_sentimiento_textos_turisticos_mx_polaridad` is a English model originally trained by hackathon-somos-nlp-2023.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/vg055_roberta_base_bne_finetuned_analisis_sentimiento_textos_turisticos_mx_polaridad_en_5.5.0_3.0_1726063520080.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/vg055_roberta_base_bne_finetuned_analisis_sentimiento_textos_turisticos_mx_polaridad_en_5.5.0_3.0_1726063520080.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
    
sequenceClassifier = RoBertaForSequenceClassification.pretrained("vg055_roberta_base_bne_finetuned_analisis_sentimiento_textos_turisticos_mx_polaridad","en")\
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
    
val sequenceClassifier = RoBertaForSequenceClassification.pretrained("vg055_roberta_base_bne_finetuned_analisis_sentimiento_textos_turisticos_mx_polaridad","en")
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
|Model Name:|vg055_roberta_base_bne_finetuned_analisis_sentimiento_textos_turisticos_mx_polaridad|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|450.4 MB|

## References

References

https://huggingface.co/hackathon-somos-nlp-2023/vg055-roberta-base-bne-finetuned-analisis-sentimiento-textos-turisticos-mx-polaridad