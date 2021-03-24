---
layout: model
title: Detect Radiology Related Entities
author: John Snow Labs
name: ner_radiology
date: 2021-01-18
task: Named Entity Recognition
language: en
edition: Spark NLP for Healthcare 2.7.0
tags: [en, ner, licensed, clinical]
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained named entity recognition deep learning model for radiology related texts and reports.

## Predicted Entities

 `ImagingTest`, `Imaging_Technique`, `ImagingFindings`, `OtherFindings`, `BodyPart`, `Direction`, `Test`, `Symptom`, `Disease_Syndrome_Disorder`,  `Medical_Device`, `Procedure`, `Measurements`, `Units`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_radiology_en_2.7.0_2.4_1610995075088.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

Use as part of an NLP pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel. Add the NerConverter to the end of the pipeline to convert entity tokens into full entity chunks.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
radiology_ner = NerDLModel.pretrained("ner_radiology", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")
...
nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, radiology_ner, ner_converter])

model = nlpPipeline.fit(spark.createDataFrame([['''Bilateral breast ultrasound was subsequently performed, which demonstrated an ovoid mass measuring approximately 0.5 x 0.5 x 0.4 cm in diameter located within the anteromedial aspect of the left shoulder. This mass demonstrates isoechoic echotexture to the adjacent muscle, with no evidence of internal color flow. This may represent benign fibrous tissue or a lipoma.''']]).toDF("text"))

results = model.transform(data)
```


</div>

## Results

```bash
|    | chunks                | entities                  |
|----|-----------------------|---------------------------|
| 0  | Bilateral             | Direction                 |
| 1  | breast                | BodyPart                  |
| 2  | ultrasound            | ImagingTest               |
| 3  | ovoid mass            | ImagingFindings           |
| 4  | 0.5 x 0.5 x 0.4       | Measurements              |
| 5  | cm                    | Units                     |
| 6  | anteromedial aspect   | Direction                 |
| 7  | left                  | Direction                 |
| 8  | shoulder              | BodyPart                  |
| 9  | mass                  | ImagingFindings           |
| 10 | isoechoic echotexture | ImagingFindings           |
| 11 | muscle                | BodyPart                  |
| 12 | internal color flow   | ImagingFindings           |
| 13 | benign fibrous tissue | ImagingFindings           |
| 14 | lipoma                | Disease_Syndrome_Disorder |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_radiology|
|Type:|ner|
|Compatibility:|Spark NLP 2.7.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Dependencies:|embeddings_clinical|

## Data Source

Trained on a custom dataset comprising of  MIMIC-CXR and MT Radiology  texts

## Benchmarking

```bash
+--------------------+------+-----+-----+------+---------+------+------+
|              entity|    tp|   fp|   fn| total|precision|recall|    f1|
+--------------------+------+-----+-----+------+---------+------+------+
|       OtherFindings|   8.0| 15.0| 63.0|  71.0|   0.3478|0.1127|0.1702|
|        Measurements| 481.0| 30.0| 15.0| 496.0|   0.9413|0.9698|0.9553|
|           Direction| 650.0|137.0| 94.0| 744.0|   0.8259|0.8737|0.8491|
|     ImagingFindings|1345.0|355.0|324.0|1669.0|   0.7912|0.8059|0.7985|
|            BodyPart|1942.0|335.0|290.0|2232.0|   0.8529|0.8701|0.8614|
|      Medical_Device| 236.0| 75.0| 64.0| 300.0|   0.7588|0.7867|0.7725|
|                Test| 222.0| 41.0| 48.0| 270.0|   0.8441|0.8222| 0.833|
|           Procedure| 269.0|117.0|116.0| 385.0|   0.6969|0.6987|0.6978|
|         ImagingTest| 263.0| 50.0| 43.0| 306.0|   0.8403|0.8595|0.8498|
|             Symptom| 498.0|101.0|132.0| 630.0|   0.8314|0.7905|0.8104|
|Disease_Syndrome_...|1180.0|258.0|200.0|1380.0|   0.8206|0.8551|0.8375|
|               Units| 269.0| 10.0|  2.0| 271.0|   0.9642|0.9926|0.9782|
|   Imaging_Technique| 140.0| 38.0| 25.0| 165.0|   0.7865|0.8485|0.8163|
+--------------------+------+-----+-----+------+---------+------+------+

                                                    +------------------+
                                                    |             macro|
                                                    +------------------+
                                                    |0.7524248724038437|
                                                    +------------------+

                                                    +------------------+
                                                    |             micro|
                                                    +------------------+
                                                    |0.8315240382681794|
                                                    +------------------+
```
