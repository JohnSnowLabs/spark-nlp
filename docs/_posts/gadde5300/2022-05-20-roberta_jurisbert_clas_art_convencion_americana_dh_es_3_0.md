---
layout: model
title: Spanish Text Classification (from `hackathon-pln-es`)
author: John Snow Labs
name: roberta_jurisbert_clas_art_convencion_americana_dh
date: 2022-05-20
tags: [roberta,  text_classification, es, open_source]
task: Text Classification
language: es
edition: Spark NLP 3.4.4
spark_version: 3.0
supported: true
annotator: RoBertaForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `jurisbert-clas-art-convencion-americana-dh` is a Spanish model orginally trained by `hackathon-pln-es`.

## Predicted Entities

`Artículo 63.1 Reparaciones`, `Artículo 15. Derecho de Reunión`, `Artículo 4. Derecho a la Vida`, `Artículo 1. Obligación de Respetar los Derechos`, `Artículo 5. Derecho a la Integridad Personal`, `Artículo 8. Garantías Judiciales`, `Artículo 19. Derechos del Niño`, `Artículo 17. Protección a la Familia`, `Artículo 2. Deber de Adoptar Disposiciones de Derecho Interno`, `Artículo 16. Libertad de Asociación`, `Artículo 25. Protección Judicial`, `Artículo 11. Protección de la Honra y de la Dignidad`, `Artículo 12. Libertad de Conciencia y de Religión`, `Artículo 9. Principio de Legalidad y de Retroactividad`, `Artículo 7. Derecho a la Libertad Personal`, `Artículo 24. Igualdad ante la Ley`, `Artículo 6. Prohibición de la Esclavitud y Servidumbre`, `Artículo 22. Derecho de Circulación y de Residencia`, `Artículo 28. Cláusula Federal`, `Artículo 21. Derecho a la Propiedad Privada`, `Artículo_29_Normas_de_Interpretación`, `Artículo 23. Derechos Políticos`, `Artículo 13. Libertad de Pensamiento y de Expresión`, `Artículo 26. Desarrollo Progresivo`, `Artículo 30. Alcance de las Restricciones`, `Artículo 14. Derecho de Rectificación o Respuesta`, `Artículo 3. Derecho al Reconocimiento de la Personalidad Jurídica`, `Artículo 27. Suspensión de Garantías`, `Artículo 20. Derecho a la Nacionalidad`, `Artículo 18. Derecho al Nombre`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_jurisbert_clas_art_convencion_americana_dh_es_3.4.4_3.0_1653049484318.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_jurisbert_clas_art_convencion_americana_dh_es_3.4.4_3.0_1653049484318.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier = RoBertaForSequenceClassification.pretrained("roberta_jurisbert_clas_art_convencion_americana_dh","es") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("class")

pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier])

data = spark.createDataFrame([["Me encanta Spark NLP"]]).toDF("text")

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

val tokenClassifier = RoBertaForSequenceClassification.pretrained("roberta_jurisbert_clas_art_convencion_americana_dh","es") 
    .setInputCols(Array("sentence", "token")) 
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector, tokenizer, tokenClassifier))

val data = Seq("Me encanta Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_jurisbert_clas_art_convencion_americana_dh|
|Compatibility:|Spark NLP 3.4.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|es|
|Size:|466.6 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

https://huggingface.co/hackathon-pln-es/jurisbert-clas-art-convencion-americana-dh
