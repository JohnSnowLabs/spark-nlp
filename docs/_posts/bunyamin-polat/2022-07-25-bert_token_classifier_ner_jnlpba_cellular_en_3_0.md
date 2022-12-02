---
layout: model
title: Detect Cellular/Molecular Biology Entities
author: John Snow Labs
name: bert_token_classifier_ner_jnlpba_cellular
date: 2022-07-25
tags: [en, ner, clinical, licensed, bertfortokenclassification]
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

This model detects molecular biology-related terms in medical texts. The model is trained with the BertForTokenClassification method from the transformers library and imported into Spark NLP.

## Predicted Entities

`cell_line`, `cell_type`, `protein`, `DNA`, `RNA`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_jnlpba_cellular_en_4.0.0_3.0_1658754953153.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

ner_model = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_jnlpba_cellular", "en", "clinical/models")\
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

data = spark.createDataFrame([["""The results suggest that activation of protein kinase C, but not new protein synthesis, is required for IL-2 induction of IFN-gamma and GM-CSF cytoplasmic mRNA. It also was observed that suppression of cytokine gene expression by these agents was independent of the inhibition of proliferation. These data indicate that IL-2 and IL-12 may have distinct signaling pathways leading to the induction of IFN-gammaand GM-CSFgene expression, andthatthe NK3.3 cell line may serve as a novel model for dissecting the biochemical and molecular events involved in these pathways. A functional T-cell receptor signaling pathway is required for p95vav activity. Stimulation of the T-cell antigen receptor ( TCR ) induces activation of multiple tyrosine kinases, resulting in phosphorylation of numerous intracellular substrates. One substrate is p95vav, which is expressed exclusively in hematopoietic and trophoblast cells."""]]).toDF("text")

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

val ner_model = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_jnlpba_cellular", "en", "clinical/models")
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

val data = Seq("""The results suggest that activation of protein kinase C, but not new protein synthesis, is required for IL-2 induction of IFN-gamma and GM-CSF cytoplasmic mRNA. It also was observed that suppression of cytokine gene expression by these agents was independent of the inhibition of proliferation. These data indicate that IL-2 and IL-12 may have distinct signaling pathways leading to the induction of IFN-gammaand GM-CSFgene expression, andthatthe NK3.3 cell line may serve as a novel model for dissecting the biochemical and molecular events involved in these pathways. A functional T-cell receptor signaling pathway is required for p95vav activity. Stimulation of the T-cell antigen receptor ( TCR ) induces activation of multiple tyrosine kinases, resulting in phosphorylation of numerous intracellular substrates. One substrate is p95vav, which is expressed exclusively in hematopoietic and trophoblast cells.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-------------------------------------+---------+
|ner_chunk                            |label    |
+-------------------------------------+---------+
|protein kinase C                     |protein  |
|IL-2                                 |protein  |
|IFN-gamma and GM-CSF cytoplasmic mRNA|RNA      |
|cytokine gene                        |DNA      |
|IL-2                                 |protein  |
|IL-12                                |protein  |
|IFN-gammaand GM-CSFgene              |protein  |
|NK3.3 cell line                      |cell_line|
|T-cell receptor                      |protein  |
|p95vav                               |protein  |
|T-cell antigen receptor              |protein  |
|TCR                                  |protein  |
|tyrosine kinases                     |protein  |
|p95vav                               |protein  |
|hematopoietic and trophoblast cells  |cell_type|
+-------------------------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_jnlpba_cellular|
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
 B-cell_line   0.5850     0.6880  0.6324    500     
 I-cell_line   0.6374     0.7644  0.6952    989     
 B-DNA         0.7187     0.7453  0.7318    1056    
 I-DNA         0.8134     0.8603  0.8362    1789    
 B-protein     0.7286     0.8429  0.7816    5067    
 I-protein     0.8020     0.8129  0.8074    4774    
 B-RNA         0.6812     0.7966  0.7344    118     
 I-RNA         0.8358     0.8984  0.8660    187     
 B-cell_type   0.7768     0.7501  0.7632    1921    
 I-cell_type   0.8654     0.7887  0.8253    2991    
 micro-avg     0.7673     0.8065  0.7864    19392   
 macro-avg     0.7444     0.7948  0.7673    19392   
 weighted-avg  0.7722     0.8065  0.7875    19392  
```
