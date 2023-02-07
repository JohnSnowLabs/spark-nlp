---
layout: model
title: Detect Clinical Conditions (ner_eu_clinical_case - fr)
author: John Snow Labs
name: ner_eu_clinical_condition
date: 2023-02-06
tags: [fr, clinical, licensed, ner, clinical_condition]
task: Named Entity Recognition
language: fr
edition: Healthcare NLP 4.2.8
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained named entity recognition (NER) deep learning model for extracting clinical conditions from French texts. The SparkNLP deep learning model (MedicalNerModel) is inspired by a former state of the art model for NER: Chiu & Nichols, Named Entity Recognition with Bidirectional LSTM-CNN.

The corpus used for model training is provided by European Clinical Case Corpus (E3C), a project aimed at offering a freely available multilingual corpus of semantically annotated clinical narratives.

## Predicted Entities

`clinical_condition`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_eu_clinical_condition_fr_4.2.8_3.0_1675725809666.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_eu_clinical_condition_fr_4.2.8_3.0_1675725809666.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

word_embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d","fr")\
	.setInputCols(["sentence","token"])\
	.setOutputCol("embeddings")

ner = MedicalNerModel.pretrained('ner_eu_clinical_condition', "fr", "clinical/models") \
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

data = spark.createDataFrame([["""Il aurait présenté il y’ a environ 30 ans des ulcérations génitales non traitées spontanément guéries. L’interrogatoire retrouvait une toux sèche depuis trois mois, des douleurs rétro-sternales constrictives, une dyspnée stade III de la NYHA et un contexte d’ apyrexie. Sur ce tableau s’ est greffé des œdèmes des membres inférieurs puis un tableau d’ anasarque d’ où son hospitalisation en cardiologie pour décompensation cardiaque globale."""]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documenter = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
  .setInputCols("document")
  .setOutputCol("sentence")

val tokenizer = new Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d","fr")
	.setInputCols(Array("sentence","token"))
	.setOutputCol("embeddings")

val ner_model = MedicalNerModel.pretrained("ner_eu_clinical_condition", "fr", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverterInternal()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(documenter, sentenceDetector, tokenizer, word_embeddings, ner_model, ner_converter))

val data = Seq(Array("""Il aurait présenté il y’ a environ 30 ans des ulcérations génitales non traitées spontanément guéries. L’interrogatoire retrouvait une toux sèche depuis trois mois, des douleurs rétro-sternales constrictives, une dyspnée stade III de la NYHA et un contexte d’ apyrexie. Sur ce tableau s’ est greffé des œdèmes des membres inférieurs puis un tableau d’ anasarque d’ où son hospitalisation en cardiologie pour décompensation cardiaque globale.""")).toDS().toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+------------------------+------------------+
|chunk                   |ner_label         |
+------------------------+------------------+
|ulcérations             |clinical_condition|
|toux sèche              |clinical_condition|
|douleurs                |clinical_condition|
|dyspnée                 |clinical_condition|
|apyrexie                |clinical_condition|
|anasarque               |clinical_condition|
|décompensation cardiaque|clinical_condition|
+------------------------+------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_eu_clinical_condition|
|Compatibility:|Healthcare NLP 4.2.8+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|fr|
|Size:|899.9 KB|

## References

The corpus used for model training is provided by European Clinical Case Corpus (E3C), a project aimed at offering a freely available multilingual corpus of semantically annotated clinical narratives.

## Benchmarking

```bash
             label     tp     fp    fn  total  precision  recall      f1
    clinical_event  269.0   51.0  52.0  321.0     0.8406  0.8380  0.8393
            macro     -      -      -     -         -       -     0.8393
            micro     -      -      -     -         -       -     0.8393
```