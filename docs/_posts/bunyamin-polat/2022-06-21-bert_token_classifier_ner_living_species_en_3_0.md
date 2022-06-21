---
layout: model
title: Detect Living Species
author: John Snow Labs
name: bert_token_classifier_ner_living_species
date: 2022-06-21
tags: [en, ner, clinical, licensed, bertfortokenclassification]
task: Named Entity Recognition
language: en
edition: Spark NLP for Healthcare 4.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Extract living species from clinical texts which is critical to scientific disciplines like medicine, biology, ecology/biodiversity, nutrition and agriculture. This model is trained with the `BertForTokenClassification` method from the `transformers` library and imported into Spark NLP.

It is trained on the [LivingNER corpus] (https://zenodo.org/record/6642852#.YrH3_XZBy01) that is composed of clinical case reports extracted from miscellaneous medical specialties including COVID, oncology, infectious diseases, tropical medicine, urology, pediatrics, and others.

## Predicted Entities

`HUMAN`, `SPECIES`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_living_species_en_4.0.0_3.0_1655830020322.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")\

sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

ner_model = BertForTokenClassification.pretrained("bert_token_classifier_ner_living_species", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("ner")\
    .setCaseSensitive(True)\
    .setMaxSentenceLength(512)

ner_converter = NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

pipeline = Pipeline(stages=[
    document_assembler, 
    sentence_detector,
    tokenizer,
    ner_model,
    ner_converter   
    ])

model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

data = spark.createDataFrame([["""On 1 January 2020, a 42-year-old man was admitted to Union Hospital (Tongji Medical School, Wuhan, Hubei Province) with hyperthermia (39.6°C), cough and fatigue of one week's duration. On auscultation, bilateral breath sounds with moist rales were heard at the bases of both lungs. Laboratory tests showed leukocytopenia (leukocyte count: 2.88 3 109/L) and lymphocytosis (lymphocyte count: 0.90 3 109/L). The leukocyte count showed 56.6% neutrophils, 32.1% lymphocytes and 10.2% monocytes. Several additional analytical tests gave abnormal results, such as C-reactive protein (158.95 mg/L; normal range: 0-10 mg/L), erythrocyte sedimentation rate (38 mm/h; normal value: 20 mm/h), serum amyloid A protein (607.1 mg/L; normal value: 10 mg/L), aspartate aminotransferase (53 U/L; normal range: 8-40 U/L) and alanine aminotransferase (60 U/L; normal range: 5-40 U/L). Real-time fluorescence PCR of the patient's sputum was positive for 2019-nCoV nucleic acid. The patient was treated with antivirals (ganciclovir, oseltamivir) and anti-inflammatory drugs (meropenem, linezolid), with symptomatic treatment, from 1 January 2020 until his discharge on 25 January 2020. The consecutive imaging tests shown in the figure illustrate the patient's improvement after therapy."""]]).toDF("text")

result = model.transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")
    .setInputCols("document")
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val ner_model = BertForTokenClassification.pretrained("bert_token_classifier_ner_living_species", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("ner")
    .setCaseSensitive(True)
    .setMaxSentenceLength(512)

val ner_converter = new NerConverter()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")

val pipeline = new PipelineModel().setStages(Array(
    document_assembler, 
    sentence_detector,
    tokenizer,
    ner_model,
    ner_converter))

val data = Seq("On 1 January 2020, a 42-year-old man was admitted to Union Hospital (Tongji Medical School, Wuhan, Hubei Province) with hyperthermia (39.6°C), cough and fatigue of one week's duration. On auscultation, bilateral breath sounds with moist rales were heard at the bases of both lungs. Laboratory tests showed leukocytopenia (leukocyte count: 2.88 3 109/L) and lymphocytosis (lymphocyte count: 0.90 3 109/L). The leukocyte count showed 56.6% neutrophils, 32.1% lymphocytes and 10.2% monocytes. Several additional analytical tests gave abnormal results, such as C-reactive protein (158.95 mg/L; normal range: 0-10 mg/L), erythrocyte sedimentation rate (38 mm/h; normal value: 20 mm/h), serum amyloid A protein (607.1 mg/L; normal value: 10 mg/L), aspartate aminotransferase (53 U/L; normal range: 8-40 U/L) and alanine aminotransferase (60 U/L; normal range: 5-40 U/L). Real-time fluorescence PCR of the patient's sputum was positive for 2019-nCoV nucleic acid. The patient was treated with antivirals (ganciclovir, oseltamivir) and anti-inflammatory drugs (meropenem, linezolid), with symptomatic treatment, from 1 January 2020 until his discharge on 25 January 2020. The consecutive imaging tests shown in the figure illustrate the patient's improvement after therapy.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+----------+-------+
| ner_chunk|  label|
+----------+-------+
|       man|  HUMAN|
| patient's|  HUMAN|
| 2019-nCoV|SPECIES|
|   patient|  HUMAN|
|antivirals|SPECIES|
| patient's|  HUMAN|
+----------+-------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_living_species|
|Compatibility:|Spark NLP for Healthcare 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|404.2 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

[LivingNER dataset](https://zenodo.org/record/6642852#.YrH3_XZBy01)

## Benchmarking

```bash
 label         precision  recall  f1-score  support 
 B-HUMAN       0.83       0.96    0.89      2950    
 B-SPECIES     0.70       0.93    0.80      3129    
 I-HUMAN       0.73       0.39    0.51      145     
 I-SPECIES     0.67       0.81    0.74      1166    
 micro-avg     0.74       0.91    0.82      7390    
 macro-avg     0.73       0.77    0.73      7390    
 weighted-avg  0.75       0.91    0.82      7390  
```