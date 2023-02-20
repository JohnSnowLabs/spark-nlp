---
layout: model
title: Detect Clinical Entities (ner_eu_clinical_case - fr)
author: John Snow Labs
name: ner_eu_clinical_case
date: 2023-02-01
tags: [fr, clinical, licensed, ner]
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

Pretrained named entity recognition (NER) deep learning model for extracting clinical entities from French texts. The SparkNLP deep learning model (MedicalNerModel) is inspired by a former state of the art model for NER: Chiu & Nichols, Named Entity Recognition with Bidirectional LSTM-CNN.

The corpus used for model training is provided by European Clinical Case Corpus (E3C), a project aimed at offering a freely available multilingual corpus of semantically annotated clinical narratives.

## Predicted Entities

`clinical_event`, `bodypart`, `clinical_condition`, `units_measurements`, `patient`, `date_time`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_eu_clinical_case_fr_4.2.8_3.0_1675293960896.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_eu_clinical_case_fr_4.2.8_3.0_1675293960896.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

ner = MedicalNerModel.pretrained('ner_eu_clinical_case', "fr", "clinical/models") \
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

data = spark.createDataFrame([["""Un garçon de 3 ans atteint d'un trouble autistique à l'hôpital du service pédiatrique A de l'hôpital universitaire. Il n'a pas d'antécédents familiaux de troubles ou de maladies du spectre autistique. Le garçon a été diagnostiqué avec un trouble de communication sévère, avec des difficultés d'interaction sociale et un traitement sensoriel retardé. Les tests sanguins étaient normaux (thyréostimuline (TSH), hémoglobine, volume globulaire moyen (MCV) et ferritine). L'endoscopie haute a également montré une tumeur sous-muqueuse provoquant une obstruction subtotale de la sortie gastrique. Devant la suspicion d'une tumeur stromale gastro-intestinale, une gastrectomie distale a été réalisée. L'examen histopathologique a révélé une prolifération de cellules fusiformes dans la couche sous-muqueuse."""]]).toDF("text")

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

val ner_model = MedicalNerModel.pretrained("ner_eu_clinical_case", "fr", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverterInternal()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(documenter, sentenceDetector, tokenizer, word_embeddings, ner_model, ner_converter))

val data = Seq(Array("""Un garçon de 3 ans atteint d'un trouble autistique à l'hôpital du service pédiatrique A de l'hôpital universitaire. Il n'a pas d'antécédents familiaux de troubles ou de maladies du spectre autistique. Le garçon a été diagnostiqué avec un trouble de communication sévère, avec des difficultés d'interaction sociale et un traitement sensoriel retardé. Les tests sanguins étaient normaux (thyréostimuline (TSH), hémoglobine, volume globulaire moyen (MCV) et ferritine). L'endoscopie haute a également montré une tumeur sous-muqueuse provoquant une obstruction subtotale de la sortie gastrique. Devant la suspicion d'une tumeur stromale gastro-intestinale, une gastrectomie distale a été réalisée. L'examen histopathologique a révélé une prolifération de cellules fusiformes dans la couche sous-muqueuse.""")).toDS().toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-----------------------------------------------------+------------------+
|chunk                                                |ner_label         |
+-----------------------------------------------------+------------------+
|Un garçon de 3 ans                                   |patient           |
|trouble autistique à l'hôpital du service pédiatrique|clinical_condition|
|l'hôpital                                            |clinical_event    |
|Il n'a                                               |patient           |
|d'antécédents                                        |clinical_event    |
|troubles                                             |clinical_condition|
|maladies                                             |clinical_condition|
|du spectre autistique                                |bodypart          |
|Le garçon                                            |patient           |
|diagnostiqué                                         |clinical_event    |
|trouble                                              |clinical_condition|
|difficultés                                          |clinical_event    |
|traitement                                           |clinical_event    |
|tests                                                |clinical_event    |
|normaux                                              |units_measurements|
|thyréostimuline                                      |clinical_event    |
|TSH                                                  |clinical_event    |
|ferritine                                            |clinical_event    |
|L'endoscopie                                         |clinical_event    |
|montré                                               |clinical_event    |
|tumeur sous-muqueuse                                 |clinical_condition|
|provoquant                                           |clinical_event    |
|obstruction                                          |clinical_condition|
|la sortie gastrique                                  |bodypart          |
|suspicion                                            |clinical_event    |
|tumeur stromale gastro-intestinale                   |clinical_condition|
|gastrectomie                                         |clinical_event    |
|L'examen                                             |clinical_event    |
|révélé                                               |clinical_event    |
|prolifération                                        |clinical_event    |
|cellules fusiformes                                  |bodypart          |
|la couche sous-muqueuse                              |bodypart          |
+-----------------------------------------------------+------------------+
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
|Language:|fr|
|Size:|895.0 KB|

## References

The corpus used for model training is provided by European Clinical Case Corpus (E3C), a project aimed at offering a freely available multilingual corpus of semantically annotated clinical narratives.

## Benchmarking

```bash
             label     tp     fp    fn  total  precision  recall      f1
         date_time   49.0   14.0  70.0  104.0     0.7778  0.7000  0.7368
units_measurements   92.0   19.0   6.0   48.0     0.8288  0.9388  0.8804
clinical_condition  178.0   74.0  73.0  120.0     0.7063  0.7092  0.7078
           patient  114.0    6.0  15.0   87.0     0.9500  0.8837  0.9157
    clinical_event  265.0   81.0  71.0  478.0     0.7659  0.7887  0.7771
          bodypart  243.0   34.0  64.0  166.0     0.8773  0.7915  0.8322
            macro     -      -      -     -         -       -     0.8083
            micro     -      -      -     -         -       -     0.7978
```
