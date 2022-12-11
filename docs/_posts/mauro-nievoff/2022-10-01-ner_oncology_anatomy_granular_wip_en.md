---
layout: model
title: Extract Granular Anatomical Entities from Oncology Texts
author: John Snow Labs
name: ner_oncology_anatomy_granular_wip
date: 2022-10-01
tags: [licensed, clinical, oncology, en, ner, anatomy]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 4.0.0
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model extractions mentions of anatomical entities using granular labels.

Definitions of Predicted Entities:

- `Direction`: Directional and laterality terms, such as "left", "right", "bilateral", "upper" and "lower".
- `Site_Bone`: Anatomical terms that refer to the human skeleton.
- `Site_Brain`: Anatomical terms that refer to the central nervous system (including the brain stem and the cerebellum).
- `Site_Breast`: Anatomical terms that refer to the breasts.
- `Site_Liver`: Anatomical terms that refer to the liver.
- `Site_Lung`: Anatomical terms that refer to the lungs.
- `Site_Lymph_Node`: Anatomical terms that refer to lymph nodes, excluding adenopathies.
- `Site_Other_Body_Part`: Relevant anatomical terms that are not included in the rest of the anatomical entities.


## Predicted Entities

`Direction`, `Site_Bone`, `Site_Brain`, `Site_Breast`, `Site_Liver`, `Site_Lung`, `Site_Lymph_Node`, `Site_Other_Body_Part`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_ONCOLOGY_CLINICAL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/27.Oncology_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_oncology_anatomy_granular_wip_en_4.0.0_3.0_1664584284877.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use


<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel().pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")                

ner = MedicalNerModel.pretrained("ner_oncology_anatomy_granular_wip", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk")
pipeline = Pipeline(stages=[document_assembler,
                            sentence_detector,
                            tokenizer,
                            word_embeddings,
                            ner,
                            ner_converter])

data = spark.createDataFrame([["The patient presented a mass in her left breast, and a possible metastasis in her lungs and in her liver."]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
    
val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")
    .setInputCols(Array("document"))
    .setOutputCol("sentence")
    
val tokenizer = new Tokenizer()
    .setInputCols(Array("sentence"))
    .setOutputCol("token")
    
val word_embeddings = WordEmbeddingsModel().pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")                
    
val ner = MedicalNerModel.pretrained("ner_oncology_anatomy_granular_wip", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")
    
val ner_converter = new NerConverter()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")

        
val pipeline = new Pipeline().setStages(Array(document_assembler,
                            sentence_detector,
                            tokenizer,
                            word_embeddings,
                            ner,
                            ner_converter))    

val data = Seq("The patient presented a mass in her left breast, and a possible metastasis in her lungs and in her liver.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
| chunk   | ner_label   |
|:--------|:------------|
| left    | Direction   |
| breast  | Site_Breast |
| lungs   | Site_Lung   |
| liver   | Site_Liver  |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_oncology_anatomy_granular_wip|
|Compatibility:|Healthcare NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|859.9 KB|

## References

In-house annotated oncology case reports.

## Benchmarking

```bash
               label     tp    fp    fn  total  precision  recall   f1
           Direction  601.0 150.0 133.0  734.0       0.80    0.82 0.81
     Site_Lymph_Node  415.0  31.0  51.0  466.0       0.93    0.89 0.91
         Site_Breast   98.0   6.0  20.0  118.0       0.94    0.83 0.88
Site_Other_Body_Part  713.0 277.0 388.0 1101.0       0.72    0.65 0.68
           Site_Bone  176.0  30.0  56.0  232.0       0.85    0.76 0.80
          Site_Liver  134.0  77.0  36.0  170.0       0.64    0.79 0.70
           Site_Lung  337.0  70.0 106.0  443.0       0.83    0.76 0.79
          Site_Brain  164.0  58.0  36.0  200.0       0.74    0.82 0.78
           macro_avg 2638.0 699.0 826.0 3464.0       0.81    0.79 0.80
           micro_avg    NaN   NaN   NaN    NaN       0.79    0.76 0.78
```