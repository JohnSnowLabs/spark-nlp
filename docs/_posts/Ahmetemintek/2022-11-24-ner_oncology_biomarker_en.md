---
layout: model
title: Extract Biomarkers and their Results
author: John Snow Labs
name: ner_oncology_biomarker
date: 2022-11-24
tags: [licensed, clinical, en, ner, oncology, biomarker]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 4.2.2
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model extracts mentions of biomarkers and biomarker results from oncology texts.

Definitions of Predicted Entities:

- `Biomarker`: Biological molecules that indicate the presence or absence of cancer, or the type of cancer (including oncogenes).
- `Biomarker_Result`: Terms or values that are identified as the result of a biomarkers.


## Predicted Entities

`Biomarker`, `Biomarker_Result`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_ONCOLOGY_CLINICAL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/27.Oncology_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_oncology_biomarker_en_4.2.2_3.0_1669299787628.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

ner = MedicalNerModel.pretrained("ner_oncology_biomarker", "en", "clinical/models") \
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

data = spark.createDataFrame([["The results of immunohistochemical examination showed that she tested negative for CK7, synaptophysin (Syn), chromogranin A (CgA), Muc5AC, human epidermal growth factor receptor-2 (HER2), and Muc6; positive for CK20, Muc1, Muc2, E-cadherin, and p53; the Ki-67 index was about 87%."]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
    
val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")
    .setInputCols("document")
    .setOutputCol("sentence")
    
val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")
    
val word_embeddings = WordEmbeddingsModel().pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")                
    
val ner = MedicalNerModel.pretrained("ner_oncology_biomarker", "en", "clinical/models")
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

val data = Seq("The results of immunohistochemical examination showed that she tested negative for CK7, synaptophysin (Syn), chromogranin A (CgA), Muc5AC, human epidermal growth factor receptor-2 (HER2), and Muc6; positive for CK20, Muc1, Muc2, E-cadherin, and p53; the Ki-67 index was about 87%.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
| chunk                                    | ner_label        |
|:-----------------------------------------|:-----------------|
| negative                                 | Biomarker_Result |
| CK7                                      | Biomarker        |
| synaptophysin                            | Biomarker        |
| Syn                                      | Biomarker        |
| chromogranin A                           | Biomarker        |
| CgA                                      | Biomarker        |
| Muc5AC                                   | Biomarker        |
| human epidermal growth factor receptor-2 | Biomarker        |
| HER2                                     | Biomarker        |
| Muc6                                     | Biomarker        |
| positive                                 | Biomarker_Result |
| CK20                                     | Biomarker        |
| Muc1                                     | Biomarker        |
| Muc2                                     | Biomarker        |
| E-cadherin                               | Biomarker        |
| p53                                      | Biomarker        |
| Ki-67 index                              | Biomarker        |
| 87%                                      | Biomarker_Result |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_oncology_biomarker|
|Compatibility:|Healthcare NLP 4.2.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|34.3 MB|

## References

In-house annotated oncology case reports.

## Benchmarking

```bash
           label   tp  fp  fn  total  precision  recall   f1
Biomarker_Result 1030 148 415   1445       0.87    0.71 0.79
       Biomarker 1685 272 279   1964       0.86    0.86 0.86
       macro_avg 2715 420 694   3409       0.87    0.79 0.82
       micro_avg 2715 420 694   3409       0.87    0.80 0.83
```