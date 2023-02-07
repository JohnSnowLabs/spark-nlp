---
layout: model
title: Detect Clinical Entities (ner_eu_clinical_case - es)
author: John Snow Labs
name: ner_eu_clinical_case
date: 2023-02-01
tags: [es, clinical, licensed, ner]
task: Named Entity Recognition
language: es
edition: Healthcare NLP 4.2.8
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained named entity recognition (NER) deep learning model for extracting clinical entities from Spanish texts. The SparkNLP deep learning model (MedicalNerModel) is inspired by a former state of the art model for NER: Chiu & Nichols, Named Entity Recognition with Bidirectional LSTM-CNN.

The corpus used for model training is provided by European Clinical Case Corpus (E3C), a project aimed at offering a freely available multilingual corpus of semantically annotated clinical narratives.

## Predicted Entities

`clinical_event`, `bodypart`, `clinical_condition`, `units_measurements`, `patient`, `date_time`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_eu_clinical_case_es_4.2.8_3.0_1675285093855.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_eu_clinical_case_es_4.2.8_3.0_1675285093855.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
	.setInputCol("text")\
	.setOutputCol("document")
 
sentenceDetectorDL = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
	.setInputCols(["document"])\
	.setOutputCol("sentence")

tokenizer = Tokenizer()\
	.setInputCols(["sentence"])\
	.setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d","es")\
	.setInputCols(["sentence","token"])\
	.setOutputCol("embeddings")

ner = MedicalNerModel.pretrained("ner_eu_clinical_case", "es", "clinical/models") \
	.setInputCols(["sentence", "token", "embeddings"]) \
	.setOutputCol("ner")
 
ner_converter = NerConverterInternal()\
	.setInputCols(["sentence", "token", "ner"])\
	.setOutputCol("ner_chunk")

pipeline = Pipeline(stages=[
	document_assembler,
	sentenceDetectorDL,
	tokenizer,
	word_embeddings,
	ner,
	ner_converter])

data = spark.createDataFrame([["""Un niño de 3 años con trastorno autista en el hospital de la sala pediátrica A del hospital universitario. No tiene antecedentes familiares de enfermedad o trastorno del espectro autista. El niño fue diagnosticado con un trastorno de comunicación severo, con dificultades de interacción social y retraso en el procesamiento sensorial. Los análisis de sangre fueron normales (hormona estimulante de la tiroides (TSH), hemoglobina, volumen corpuscular medio (MCV) y ferritina). La endoscopia alta también mostró un tumor submucoso que causaba una obstrucción subtotal de la salida gástrica. Ante la sospecha de tumor del estroma gastrointestinal, se realizó gastrectomía distal. El examen histopatológico reveló proliferación de células fusiformes en la capa submucosa."""]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
	.setInputCol("text")
	.setOutputCol("document")
 
val sentenceDetectorDL = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
	.setInputCols("document")
	.setOutputCol("sentence")

val tokenizer = new Tokenizer()
	.setInputCols("sentence")
	.setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d","es")
	.setInputCols(Array("sentence","token"))
	.setOutputCol("embeddings")

val ner = MedicalNerModel.pretrained("ner_eu_clinical_case", "es", "clinical/models")
	.setInputCols(Array("sentence", "token", "embeddings"))
	.setOutputCol("ner")
 
val ner_converter = new NerConverterInternal()
	.setInputCols(Array("sentence", "token", "ner"))
	.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(
	document_assembler,
	sentenceDetectorDL,
	tokenizer,
	word_embeddings,
	ner,
	ner_converter))

val data =  Seq("""Un niño de 3 años con trastorno autista en el hospital de la sala pediátrica A del hospital universitario. No tiene antecedentes familiares de enfermedad o trastorno del espectro autista. El niño fue diagnosticado con un trastorno de comunicación severo, con dificultades de interacción social y retraso en el procesamiento sensorial. Los análisis de sangre fueron normales (hormona estimulante de la tiroides (TSH), hemoglobina, volumen corpuscular medio (MCV) y ferritina). La endoscopia alta también mostró un tumor submucoso que causaba una obstrucción subtotal de la salida gástrica. Ante la sospecha de tumor del estroma gastrointestinal, se realizó gastrectomía distal. El examen histopatológico reveló proliferación de células fusiformes en la capa submucosa.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+--------------------------------+------------------+
|chunk                           |ner_label         |
+--------------------------------+------------------+
|Un niño de 3 años               |patient           |
|trastorno autista               |clinical_event    |
|antecedentes                    |clinical_event    |
|enfermedad                      |clinical_event    |
|trastorno del espectro autista  |clinical_event    |
|El niño                         |patient           |
|diagnosticado                   |clinical_event    |
|trastorno de comunicación severo|clinical_event    |
|dificultades                    |clinical_event    |
|retraso                         |clinical_event    |
|análisis                        |clinical_event    |
|sangre                          |bodypart          |
|normales                        |units_measurements|
|hormona                         |clinical_event    |
|la tiroides                     |bodypart          |
|TSH                             |clinical_event    |
|hemoglobina                     |clinical_event    |
|volumen                         |clinical_event    |
|MCV                             |clinical_event    |
|ferritina                       |clinical_event    |
|endoscopia                      |clinical_event    |
|mostró                          |clinical_event    |
|tumor submucoso                 |clinical_event    |
|obstrucción                     |clinical_event    |
|tumor                           |clinical_event    |
|del estroma gastrointestinal    |bodypart          |
|gastrectomía                    |clinical_event    |
|examen                          |clinical_event    |
|reveló                          |clinical_event    |
|proliferación                   |clinical_event    |
|células fusiformes              |bodypart          |
|la capa submucosa               |bodypart          |
+--------------------------------+------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_eu_clinical_case|
|Compatibility:|Healthcare NLP 4.2.8+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|895.1 KB|

## References

The corpus used for model training is provided by European Clinical Case Corpus (E3C), a project aimed at offering a freely available multilingual corpus of semantically annotated clinical narratives.

## Benchmarking

```bash
             label     tp     fp    fn  total  precision  recall      f1
         date_time   87.0   10.0  17.0  104.0     0.8969  0.8365  0.8657
units_measurements   37.0    5.0  11.0   48.0     0.8810  0.7708  0.8222
clinical_condition   50.0   34.0  70.0  120.0     0.5952  0.4167  0.4902
           patient   76.0    8.0  11.0   87.0     0.9048  0.8736  0.8889
    clinical_event  399.0   44.0  79.0  478.0     0.9007  0.8347  0.8664
          bodypart  153.0   56.0  13.0  166.0     0.7321  0.9217  0.8160
            macro     -      -      -     -         -       -     0.7916
            micro     -      -      -     -         -       -     0.8128
```
