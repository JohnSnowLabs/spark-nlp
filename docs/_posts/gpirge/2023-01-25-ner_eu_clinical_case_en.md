---
layout: model
title: Detect Clinical Entities (ner_eu_clinical_case)
author: John Snow Labs
name: ner_eu_clinical_case
date: 2023-01-25
tags: [clinical, licensed, ner, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 4.2.7
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained named entity recognition (NER) deep learning model for clinical entities. The SparkNLP deep learning model (MedicalNerModel) is inspired by a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM-CNN.

The corpus used for model training is provided by European Clinical Case Corpus (E3C), a project aimed at offering a freely available multilingual corpus of semantically annotated clinical narratives.

## Predicted Entities

`clinical_event`, `bodypart`, `clinical_condition`, `units_measurements`, `patient`, `date_time`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_eu_clinical_case_en_4.2.7_3.2_1674657662344.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_eu_clinical_case_en_4.2.7_3.2_1674657662344.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
	.setInputCol("text")\
	.setOutputCol("document")
 
sentenceDetectorDL = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "en")\
	.setInputCols(["document"])\
	.setOutputCol("sentence")

tokenizer = Tokenizer()\
	.setInputCols(["sentence"])\
	.setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical","en","clinical/models")\
	.setInputCols(["sentence","token"])\
	.setOutputCol("embeddings")

ner = MedicalNerModel.pretrained('ner_eu_clinical_case', "en", "clinical/models") \
	.setInputCols(["sentence", "token", "embeddings"]) \
	.setOutputCol("ner")
 
ner_converter = NerConverter()\
	.setInputCols(["sentence", "token", "ner"])\
	.setOutputCol("ner_chunk")

pipeline = pipeline(stages=[
	document_assembler,
	sentenceDetectorDL,
	tokenizer,
	word_embeddings,
	ner,
	ner_converter])

data = spark.createDataFrame([["""A 3-year-old boy with autistic disorder on hospital of pediatric ward A at university hospital. He has no family history of illness or autistic spectrum disorder. The child was diagnosed with a severe communication disorder, with social interaction difficulties and sensory processing delay. Blood work was normal (thyroid-stimulating hormone (TSH), hemoglobin, mean corpuscular volume (MCV), and ferritin). Upper endoscopy also showed a submucosal tumor causing subtotal obstruction of the gastric outlet. Because a gastrointestinal stromal tumor was suspected, distal gastrectomy was performed. Histopathological examination revealed spindle cell proliferation in the submucosal layer."""]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documenter = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained()
  .setInputCols("document")
  .setOutputCol("sentence")

val tokenizer = new Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical","en","clinical/models")
	.setInputCols(Array("sentence","token"))
	.setOutputCol("embeddings")

val ner_model = MedicalNerModel.pretrained("ner_eu_clinical_case", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(documenter, sentenceDetector, tokenizer, word_embeddings, ner_model, ner_converter))

val data = Seq(Array("""A 3-year-old boy with autistic disorder on hospital of pediatric ward A at university hospital. He has no family history of illness or autistic spectrum disorder. The child was diagnosed with a severe communication disorder, with social interaction difficulties and sensory processing delay. Blood work was normal (thyroid-stimulating hormone (TSH), hemoglobin, mean corpuscular volume (MCV), and ferritin). Upper endoscopy also showed a submucosal tumor causing subtotal obstruction of the gastric outlet. Because a gastrointestinal stromal tumor was suspected, distal gastrectomy was performed. Histopathological examination revealed spindle cell proliferation in the submucosal layer.""")).toDS().toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+------------------------------+------------------+
|chunk                         |ner_label         |
+------------------------------+------------------+
|A 3-year-old boy              |patient           |
|autistic disorder             |clinical_condition|
|He                            |patient           |
|illness                       |clinical_event    |
|autistic spectrum disorder    |clinical_condition|
|The child                     |patient           |
|diagnosed                     |clinical_event    |
|disorder                      |clinical_event    |
|difficulties                  |clinical_event    |
|Blood                         |bodypart          |
|work                          |clinical_event    |
|normal                        |units_measurements|
|hormone                       |clinical_event    |
|hemoglobin                    |clinical_event    |
|volume                        |clinical_event    |
|endoscopy                     |clinical_event    |
|showed                        |clinical_event    |
|tumor                         |clinical_condition|
|causing                       |clinical_event    |
|obstruction                   |clinical_event    |
|the gastric outlet            |bodypart          |
|gastrointestinal stromal tumor|clinical_condition|
|suspected                     |clinical_event    |
|gastrectomy                   |clinical_event    |
|examination                   |clinical_event    |
|revealed                      |clinical_event    |
|spindle cell proliferation    |clinical_condition|
|the submucosal layer          |bodypart          |
+------------------------------+------------------+


```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_eu_clinical_case|
|Compatibility:|Healthcare NLP 4.2.7+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|849.0 KB|

## References

The corpus used for model training is provided by European Clinical Case Corpus (E3C), a project aimed at offering a freely available multilingual corpus of semantically annotated clinical narratives.

## Benchmarking

```bash
             label     tp     fp    fn  total  precision  recall      f1
         date_time   54.0    7.0  15.0   69.0     0.8852  0.7826  0.8308
units_measurements  111.0   48.0  12.0  123.0     0.6981  0.9024  0.7872
clinical_condition   93.0   47.0  81.0  174.0     0.6643  0.5345  0.5924
           patient  119.0   16.0   5.0  124.0     0.8815  0.9597  0.9189
    clinical_event  331.0  126.0  89.0  420.0     0.7243  0.7881  0.7548
          bodypart  171.0   58.0  84.0  255.0     0.7467  0.6706  0.7066
            macro     -      -      -     -         -       -     0.7651
            micro     -      -      -     -         -       -     0.7454
```
