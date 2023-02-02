---
layout: model
title: Detect Clinical Entities (ner_eu_clinical_case - eu)
author: John Snow Labs
name: ner_eu_clinical_case
date: 2023-02-02
tags: [eu, clinical, licensed, ner]
task: Named Entity Recognition
language: eu
edition: Healthcare NLP 4.2.8
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained named entity recognition (NER) deep learning model for extracting clinical entities from Basque texts. The SparkNLP deep learning model (MedicalNerModel) is inspired by a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM-CNN.

The corpus used for model training is provided by European Clinical Case Corpus (E3C), a project aimed at offering a freely available multilingual corpus of semantically annotated clinical narratives.

## Predicted Entities

`clinical_event`, `bodypart`, `clinical_condition`, `units_measurements`, `patient`, `date_time`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_eu_clinical_case_eu_4.2.8_3.0_1675359410041.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_eu_clinical_case_eu_4.2.8_3.0_1675359410041.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

word_embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d","eu")\
	.setInputCols(["sentence","token"])\
	.setOutputCol("embeddings")

ner = MedicalNerModel.pretrained('ner_eu_clinical_case', "eu", "clinical/models") \
	.setInputCols(["sentence", "token", "embeddings"]) \
	.setOutputCol("ner")
 
ner_converter = NerConverterInternal()\
	.setInputCols(["sentence", "token", "ner"])\
	.setOutputCol("ner_chunk")

pipeline = pipeline(stages=[
	document_assembler,
	sentenceDetectorDL,
	tokenizer,
	word_embeddings,
	ner,
	ner_converter])

data = spark.createDataFrame([["""3 urteko mutiko bat nahasmendu autistarekin unibertsitateko ospitaleko A pediatriako ospitalean. Ez du autismoaren espektroaren nahaste edo gaixotasun familiaren aurrekaririk. Mutilari komunikazio-nahaste larria diagnostikatu zioten, elkarrekintza sozialeko zailtasunak eta prozesamendu sentsorial atzeratua. Odol-analisiak normalak izan ziren (tiroidearen hormona estimulatzailea (TSH), hemoglobina, batez besteko bolumen corpuskularra (MCV) eta ferritina). Goiko endoskopiak mukosaren azpiko tumore bat ere erakutsi zuen, urdail-irteeren guztizko oztopoa eragiten zuena. Estroma gastrointestinalaren tumore baten susmoa ikusita, distaleko gastrektomia egin zen. Azterketa histopatologikoak agerian utzi zuen mukosaren azpiko zelulen ugaltzea."""]]).toDF("text")

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

val word_embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d","eu")
	.setInputCols(Array("sentence","token"))
	.setOutputCol("embeddings")

val ner_model = MedicalNerModel.pretrained("ner_eu_clinical_case", "eu", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverterInternal()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(documenter, sentenceDetector, tokenizer, word_embeddings, ner_model, ner_converter))

val data = Seq(Array("""3 urteko mutiko bat nahasmendu autistarekin unibertsitateko ospitaleko A pediatriako ospitalean. Ez du autismoaren espektroaren nahaste edo gaixotasun familiaren aurrekaririk. Mutilari komunikazio-nahaste larria diagnostikatu zioten, elkarrekintza sozialeko zailtasunak eta prozesamendu sentsorial atzeratua. Odol-analisiak normalak izan ziren (tiroidearen hormona estimulatzailea (TSH), hemoglobina, batez besteko bolumen corpuskularra (MCV) eta ferritina). Goiko endoskopiak mukosaren azpiko tumore bat ere erakutsi zuen, urdail-irteeren guztizko oztopoa eragiten zuena. Estroma gastrointestinalaren tumore baten susmoa ikusita, distaleko gastrektomia egin zen. Azterketa histopatologikoak agerian utzi zuen mukosaren azpiko zelulen ugaltzea.""")).toDS().toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+----------------------------+------------------+
|chunk                       |ner_label         |
+----------------------------+------------------+
|3 urteko mutiko bat         |patient           |
|nahasmendu                  |clinical_event    |
|autismoaren espektroaren    |clinical_condition|
|nahaste                     |clinical_event    |
|gaixotasun                  |clinical_event    |
|familiaren                  |patient           |
|aurrekaririk                |clinical_event    |
|Mutilari                    |patient           |
|komunikazio-nahaste         |clinical_event    |
|diagnostikatu               |clinical_event    |
|elkarrekintza               |clinical_event    |
|zailtasunak                 |clinical_event    |
|prozesamendu sentsorial     |clinical_event    |
|Odol-analisiak              |clinical_event    |
|normalak                    |units_measurements|
|tiroidearen                 |bodypart          |
|hormona estimulatzailea     |clinical_event    |
|TSH                         |clinical_event    |
|hemoglobina                 |clinical_event    |
|bolumen                     |clinical_event    |
|MCV                         |clinical_event    |
|ferritina                   |clinical_event    |
|Goiko                       |bodypart          |
|endoskopiak                 |clinical_event    |
|mukosaren azpiko            |bodypart          |
|tumore                      |clinical_event    |
|erakutsi                    |clinical_event    |
|oztopoa                     |clinical_event    |
|Estroma gastrointestinalaren|clinical_event    |
|tumore                      |clinical_event    |
|ikusita                     |clinical_event    |
|distaleko                   |bodypart          |
|gastrektomia                |clinical_event    |
|Azterketa                   |clinical_event    |
|agerian                     |clinical_event    |
|utzi                        |clinical_event    |
|mukosaren azpiko zelulen    |bodypart          |
|ugaltzea                    |clinical_event    |
+----------------------------+------------------+


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
|Language:|eu|
|Size:|896.1 KB|

## References

The corpus used for model training is provided by European Clinical Case Corpus (E3C), a project aimed at offering a freely available multilingual corpus of semantically annotated clinical narratives.

## Sample text from the training dataset

3 urteko mutiko bat nahasmendu autistarekin unibertsitateko ospitaleko A pediatriako ospitalean. Ez du autismoaren espektroaren nahaste edo gaixotasun familiaren aurrekaririk. Mutilari komunikazio-nahaste larria diagnostikatu zioten, elkarrekintza sozialeko zailtasunak eta prozesamendu sentsorial atzeratua. Odol-analisiak normalak izan ziren (tiroidearen hormona estimulatzailea (TSH), hemoglobina, batez besteko bolumen corpuskularra (MCV) eta ferritina). Goiko endoskopiak mukosaren azpiko tumore bat ere erakutsi zuen, urdail-irteeren guztizko oztopoa eragiten zuena. Estroma gastrointestinalaren tumore baten susmoa ikusita, distaleko gastrektomia egin zen. Azterketa histopatologikoak agerian utzi zuen mukosaren azpiko zelulen ugaltzea.

## Benchmarking

```bash
             label     tp     fp    fn  total  precision  recall      f1
         date_time  103.0   13.0  26.0  129.0     0.8879  0.7984  0.8408
units_measurements  257.0   37.0   9.0  266.0     0.8741  0.9662  0.9179
clinical_condition   20.0   22.0  33.0   53.0     0.4782  0.3774  0.4211
           patient   69.0    3.0   8.0   77.0     0.9583  0.8961  0.9262
    clinical_event  712.0  121.0  95.0  807.0     0.8547  0.8823  0.8683
          bodypart  182.0   33.0  15.0  197.0     0.8465  0.9239  0.8835
            macro     -      -      -     -         -       -     0.8096
            micro     -      -      -     -         -       -     0.8640
```