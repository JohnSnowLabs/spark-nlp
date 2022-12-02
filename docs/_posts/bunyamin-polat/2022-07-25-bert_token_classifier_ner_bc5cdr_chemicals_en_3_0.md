---
layout: model
title: Detect Chemicals in Medical Text
author: John Snow Labs
name: bert_token_classifier_ner_bc5cdr_chemicals
date: 2022-07-25
tags: [en, ner, clinical, licensed, bertfortokenclasification]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 4.0.0
spark_version: 3.0
supported: true
annotator: MedicalBertForTokenClassifier
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Chemicals, diseases, and their relations are among the most searched topics by PubMed users worldwide as they play central roles in many areas of biomedical research and healthcare, such as drug discovery and safety surveillance. In addition, identifying chemicals as biomarkers can be helpful in informing potential relationships between chemicals and pathologies.

This model is trained with the `BertForTokenClassification` method from the `transformers` library and imported into Spark NLP. The model detects chemicals from a medical text.

## Predicted Entities

`CHEM`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_bc5cdr_chemicals_en_4.0.0_3.0_1658752308692.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")\

sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

ner_model = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_bc5cdr_chemicals", "en", "clinical/models")\
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

data = spark.createDataFrame([["""The possibilities that these cardiovascular findings might be the result of non-selective inhibition of monoamine oxidase or of amphetamine and metamphetamine are discussed. The results have shown that the degradation product p-choloroaniline is not a significant factor in chlorhexidine-digluconate associated erosive cystitis. A high percentage of kanamycin - colistin and povidone-iodine irrigations were associated with erosive cystitis and suggested a possible complication with human usage."""]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")
    .setInputCols(Array("document"))
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols(Array("sentence"))
    .setOutputCol("token")

val ner_model = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_bc5cdr_chemicals", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("ner")
    .setCaseSensitive(True)
    .setMaxSentenceLength(512)

val ner_converter = new NerConverter()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(document_assembler, 
                                                   sentence_detector,
                                                   tokenizer,
                                                   ner_model,
                                                   ner_converter))

val data = Seq("""The possibilities that these cardiovascular findings might be the result of non-selective inhibition of monoamine oxidase or of amphetamine and metamphetamine are discussed. The results have shown that the degradation product p-choloroaniline is not a significant factor in chlorhexidine-digluconate associated erosive cystitis. A high percentage of kanamycin - colistin and povidone-iodine irrigations were associated with erosive cystitis and suggested a possible complication with human usage.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-------------------------+-----+
|ner_chunk                |label|
+-------------------------+-----+
|amphetamine              |CHEM |
|metamphetamine           |CHEM |
|p-choloroaniline         |CHEM |
|chlorhexidine-digluconate|CHEM |
|kanamycin                |CHEM |
|colistin                 |CHEM |
|povidone-iodine          |CHEM |
+-------------------------+-----+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_bc5cdr_chemicals|
|Compatibility:|Healthcare NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|404.2 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

[https://github.com/cambridgeltl/MTL-Bioinformatics-2016](https://github.com/cambridgeltl/MTL-Bioinformatics-2016)

## Benchmarking

```bash
 label         precision  recall  f1-score  support 
 B-CHEM        0.8920     0.9734  0.9309    5385    
 I-CHEM        0.8129     0.8993  0.8539    1628    
 micro-avg     0.8734     0.9562  0.9129    7013    
 macro-avg     0.8524     0.9364  0.8924    7013    
 weighted-avg  0.8736     0.9562  0.9130    7013      
```
