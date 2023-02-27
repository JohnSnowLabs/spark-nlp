---
layout: model
title: Detect Clinical Conditions (ner_eu_clinical_condition - it)
author: John Snow Labs
name: ner_eu_clinical_condition
date: 2023-02-06
tags: [it, clinical, licensed, ner, clinical_condition]
task: Named Entity Recognition
language: it
edition: Healthcare NLP 4.2.8
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained named entity recognition (NER) deep learning model for extracting clinical conditions from Italian texts. The SparkNLP deep learning model (MedicalNerModel) is inspired by a former state of the art model for NER: Chiu & Nichols, Named Entity Recognition with Bidirectional LSTM-CNN.

The corpus used for model training is provided by European Clinical Case Corpus (E3C), a project aimed at offering a freely available multilingual corpus of semantically annotated clinical narratives.

## Predicted Entities

`clinical_condition`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_eu_clinical_condition_it_4.2.8_3.0_1675726754516.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_eu_clinical_condition_it_4.2.8_3.0_1675726754516.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

word_embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d","it")\
	.setInputCols(["sentence","token"])\
	.setOutputCol("embeddings")

ner = MedicalNerModel.pretrained('ner_eu_clinical_condition', "it", "clinical/models") \
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

data = spark.createDataFrame([["""Donna, 64 anni, ricovero per dolore epigastrico persistente, irradiato a barra e posteriormente, associato a dispesia e anoressia. Poche settimane dopo compaiono, però, iperemia, intenso edema vulvare ed una esione ulcerativa sul lato sinistro della parete rettale che la RM mostra essere una fistola transfinterica. Questi trattamenti determinano un miglioramento dell’ infiammazione ed una riduzione dell’ ulcera, ma i condilomi permangono inalterati."""]]).toDF("text")

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

val word_embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d","it")
	.setInputCols(Array("sentence","token"))
	.setOutputCol("embeddings")

val ner_model = MedicalNerModel.pretrained("ner_eu_clinical_condition", "it", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverterInternal()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(documenter, sentenceDetector, tokenizer, word_embeddings, ner_model, ner_converter))

val data = Seq(Array("""Donna, 64 anni, ricovero per dolore epigastrico persistente, irradiato a barra e posteriormente, associato a dispesia e anoressia. Poche settimane dopo compaiono, però, iperemia, intenso edema vulvare ed una esione ulcerativa sul lato sinistro della parete rettale che la RM mostra essere una fistola transfinterica. Questi trattamenti determinano un miglioramento dell’ infiammazione ed una riduzione dell’ ulcera, ma i condilomi permangono inalterati.""")).toDS().toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+----------------------+------------------+
|chunk                 |ner_label         |
+----------------------+------------------+
|dolore epigastrico    |clinical_condition|
|anoressia             |clinical_condition|
|iperemia              |clinical_condition|
|edema                 |clinical_condition|
|fistola transfinterica|clinical_condition|
|infiammazione         |clinical_condition|
+----------------------+------------------+




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
|Language:|it|
|Size:|903.5 KB|

## References

The corpus used for model training is provided by European Clinical Case Corpus (E3C), a project aimed at offering a freely available multilingual corpus of semantically annotated clinical narratives.

## Benchmarking

```bash
             label     tp     fp    fn  total  precision  recall      f1
clinical_condition  208.0   35.0  46.0  254.0     0.8560  0.8189  0.8370
            macro     -      -      -     -         -       -     0.8370
            micro     -      -      -     -         -       -     0.8370
```