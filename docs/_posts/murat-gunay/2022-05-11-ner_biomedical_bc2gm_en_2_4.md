---
layout: model
title: Detect Genes/Proteins (BC2GM) in Medical Texts
author: John Snow Labs
name: ner_biomedical_bc2gm
date: 2022-05-11
tags: [bc2gm, ner, biomedical, gene_protein, gene, protein, en, licensed, clinical]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.5.1
spark_version: 2.4
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Named Entity recognition annotator allows for a generic model to be trained by utilizing a deep learning algorithm (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN.


This model has been trained to extract genes/proteins from a medical text for PySpark 2.4.x users.


## Predicted Entities


`GENE_PROTEIN`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_biomedical_bc2gm_en_3.5.1_2.4_1652262009994.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
	.setInputCol("text")\
	.setOutputCol("document")


sentenceDetectorDL = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")\
	.setInputCols(["document"])\
	.setOutputCol("sentence")


tokenizer = Tokenizer()\
	.setInputCols(["sentence"])\
	.setOutputCol("token")


word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical" ,"en", "clinical/models")\
	.setInputCols(["sentence","token"])\
	.setOutputCol("embeddings")


ner = MedicalNerModel.pretrained("ner_biomedical_bc2gm", "en", "clinical/models") \
	.setInputCols(["sentence", "token", "embeddings"]) \
	.setOutputCol("ner")


ner_converter = NerConverter()\
	.setInputCols(["sentence", "token", "ner"])\
	.setOutputCol("ner_chunk")


nlpPipeline = Pipeline(stages=[
	document_assembler,
	sentenceDetectorDL,
	tokenizer,
	word_embeddings,
	ner,
	ner_converter])


data = spark.createDataFrame([["Immunohistochemical staining was positive for S-100 in all 9 cases stained, positive for HMB-45 in 9 (90%) of 10, and negative for cytokeratin in all 9 cases in which myxoid melanoma remained in the block after previous sections."]]).toDF("text")

result = nlpPipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
	.setInputCol("text")
	.setOutputCol("document")


val sentenceDetectorDL =
SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")
	.setInputCols("document")
	.setOutputCol("sentence")


val tokenizer = new Tokenizer()
	.setInputCols("sentence")
	.setOutputCol("token")


val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical" ,"en", "clinical/models")
	.setInputCols(Array("document","token"))
	.setOutputCol("word_embeddings")


val ner = MedicalNerModel.pretrained("ner_biomedical_bc2gm", "en", "clinical/models")
	.setInputCols(Array("sentence", "token", "word_embeddings"))
	.setOutputCol("ner")


val ner_converter = new NerConverter()
	.setInputCols(Array("sentence", "token", "ner"))
.setOutputCol("ner_chunk")


val pipeline = new Pipeline().setStages(Array(
	document_assembler, 
	sentenceDetectorDL, 
	tokenizer, 
	word_embeddings, 
	ner, 
	ner_converter))


val data = Seq("Immunohistochemical staining was positive for S-100 in all 9 cases stained, positive for HMB-45 in 9 (90%) of 10, and negative for cytokeratin in all 9 cases in which myxoid melanoma remained in the block after previous sections.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.biomedical_bc2gm").predict("""Immunohistochemical staining was positive for S-100 in all 9 cases stained, positive for HMB-45 in 9 (90%) of 10, and negative for cytokeratin in all 9 cases in which myxoid melanoma remained in the block after previous sections.""")
```

</div>


## Results


```bash
+-----------+------------+
|chunk      |ner_label   |
+-----------+------------+
|S-100      |GENE_PROTEIN|
|HMB-45     |GENE_PROTEIN|
|cytokeratin|GENE_PROTEIN|
+-----------+------------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_biomedical_bc2gm|
|Compatibility:|Healthcare NLP 3.5.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|14.6 MB|


## References


Created by Smith et al. at 2008, the BioCreative II Gene Mention Recognition ([BC2GM](https://metatext.io/datasets/biocreative-ii-gene-mention-recognition-(bc2gm))) Dataset contains data where participants are asked to identify a gene mention in a sentence by giving its start and end characters. The training set consists of a set of sentences, and for each sentence a set of gene mentions (GENE annotations).


## Benchmarking


```bash
label 			precision recall f1-score support
GENE_PROTEIN    0.83   0.82     0.82    6325
micro-avg      	0.83   0.82     0.82    6325
macro-avg      	0.83   0.82     0.82    6325
weighted-avg    0.83   0.82     0.82    6325
```