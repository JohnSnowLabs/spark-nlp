---
layout: model
title: Conventions Classification
author: John Snow Labs
name: legclf_conventions
date: 2022-08-09
tags: [es, legal, conventions, classification, licensed]
task: Text Classification
language: es
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a Roberta-based Legal Sequence Classifier NLP model to label texts as one of the following categories:

- Convención Internacional sobre la Protección de los Derechos de todos los Trabajadores Migratorios y de sus Familias
- Convención de los Derechos del Niño
- Convención sobre la Eliminación de todas las formas de Discriminación contra la Mujer
- Pacto Internacional de Derechos Civiles y Políticos
- Convención Internacional Sobre la Eliminación de Todas las Formas de Discriminación Racial
- Convención contra la Tortura y otros Tratos o Penas Crueles, Inhumanos o Degradantes
- Convención sobre los Derechos de las Personas con Discapacidad
- Pacto Internacional de Derechos Económicos, Sociales y Culturales

This model was originally trained with 3799 legal texts (see the original work [here](https://huggingface.co/hackathon-pln-es/jurisbert-class-tratados-internacionales-sistema-universal), and has been finetuned by JSL on more scrapped texts from the internet with weak labelling (as, for example, https://www.un.org/es/events/childrenday/pdf/derechos.pdf for `Convencion de los Derechos del Niño`).

## Predicted Entities

`Convención sobre la Eliminación de todas las formas de Discriminación contra la Mujer`, `Convención sobre los Derechos de las Personas con Discapacidad`, `Convención Internacional Sobre la Eliminación de Todas las Formas de Discriminación Racial`, `Convención contra la Tortura y otros Tratos o Penas Crueles, Inhumanos o Degradantes`, `Convención Internacional sobre la Protección de los Derechos de todos los Trabajadores Migratorios y de sus Familias`, `Convención de los Derechos del Niño`, `Pacto Internacional de Derechos Económicos, Sociales y Culturales`, `Pacto Internacional de Derechos Civiles y Políticos`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_conventions_es_1.0.0_3.2_1660056648122.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = nlp.DocumentAssembler() \
       .setInputCol("text") \
       .setOutputCol("document")

sentenceDetector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
       .setInputCols(["document"])\
       .setOutputCol("sentence")

tokenizer = nlp.Tokenizer() \
    .setInputCols("sentence") \
    .setOutputCol("token")

tokenClassifier = nlp.RoBertaForSequenceClassification.pretrained("legclf_conventions","es", "legal/models") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("class")

pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier])

text = """La Convención, a lo largo de sus 54 artículos, reconoce que los niños (seres humanos menores de 18 años) son individuos con derecho de pleno desarrollo físico, mental y social, y con derecho a expresar libremente sus opiniones. Además la Convención es también un modelo para la salud, la supervivencia y el progreso de toda la sociedad humana. """

data = spark.createDataFrame([[text]]).toDF("text")

result = pipeline.fit(data).transform(data)
```

</div>

## Results

```bash
+--------------------+-------------------------------------+
|                text|                               result|
+--------------------+-------------------------------------+
|La Convención, a ...|[Convención de los Derechos del Niño]|
+--------------------+-------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_conventions|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|es|
|Size:|466.6 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

This model was originally trained with 3799 legal texts (see the original work [here](https://huggingface.co/hackathon-pln-es/jurisbert-class-tratados-internacionales-sistema-universal), and has been finetuned by JSL on more scrapped texts from the internet with weak labelling (as, for example, https://www.un.org/es/events/childrenday/pdf/derechos.pdf for `Convencion de los Derechos del Niño`).

## Benchmarking

```bash
label            precision  recall   f1-score  support
accuracy            -          -     0.90      120
macro-avg        0.90       0.91     0.90      120
weighted-avg     0.90       0.90     0.90      120
```     
