---
layout: model
title: Detect Radiology Concepts - WIP (biobert)
author: John Snow Labs
name: jsl_rd_ner_wip_greedy_biobert
date: 2021-07-26
tags: [licensed, clinical, en, ner]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.1.3
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Extract clinical entities from Radiology reports using pretrained NER model.


## Predicted Entities


`Test_Result`, `OtherFindings`, `BodyPart`, `ImagingFindings`, `Disease_Syndrome_Disorder`, `ImagingTest`, `Measurements`, `Procedure`, `Score`, `Test`, `Medical_Device`, `Direction`, `Symptom`, `Imaging_Technique`, `ManualFix`, `Units`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/jsl_rd_ner_wip_greedy_biobert_en_3.1.3_3.0_1627305153541.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/jsl_rd_ner_wip_greedy_biobert_en_3.1.3_3.0_1627305153541.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
         
sentence_detector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

embeddings_clinical = BertEmbeddings.pretrained('biobert_pubmed_base_cased') \
    .setInputCols(['sentence', 'token']) \
    .setOutputCol('embeddings')

clinical_ner = MedicalNerModel.pretrained("jsl_rd_ner_wip_greedy_biobert", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverter()\
 	  .setInputCols(["sentence", "token", "ner"])\
 	  .setOutputCol("ner_chunk")
    
nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings_clinical,  clinical_ner, ner_converter])

model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame([["Bilateral breast ultrasound was subsequently performed, which demonstrated an ovoid mass measuring approximately 0.5 x 0.5 x 0.4 cm in diameter located within the anteromedial aspect of the left shoulder. This mass demonstrates isoechoic echotexture to the adjacent muscle, with no evidence of internal color flow. This may represent benign fibrous tissue or a lipoma."]], ["text"]))
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
         
val sentence_detector = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val embeddings_clinical = BertEmbeddings.pretrained("biobert_pubmed_base_cased")
    .setInputCols(["sentence", "token"])
    .setOutputCol("embeddings")

val ner = MedicalNerModel.pretrained("jsl_rd_ner_wip_greedy_biobert", "en", "clinical/models") 
    .setInputCols("sentence", "token", "embeddings")
    .setOutputCol("ner")

val ner_converter = new NerConverter()
 	.setInputCols(Array("sentence", "token", "ner"))
 	.setOutputCol("ner_chunk")
    
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings_clinical, ner, ner_converter))

val data = Seq("""Bilateral breast ultrasound was subsequently performed, which demonstrated an ovoid mass measuring approximately 0.5 x 0.5 x 0.4 cm in diameter located within the anteromedial aspect of the left shoulder. This mass demonstrates isoechoic echotexture to the adjacent muscle, with no evidence of internal color flow. This may represent benign fibrous tissue or a lipoma.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.radiology.wip_greedy_biobert").predict("""Bilateral breast ultrasound was subsequently performed, which demonstrated an ovoid mass measuring approximately 0.5 x 0.5 x 0.4 cm in diameter located within the anteromedial aspect of the left shoulder. This mass demonstrates isoechoic echotexture to the adjacent muscle, with no evidence of internal color flow. This may represent benign fibrous tissue or a lipoma.""")
```

</div>


## Results


```bash
|    | chunk                 | entity                    |
|---:|:----------------------|:--------------------------|
|  0 | Bilateral             | Direction                 |
|  1 | breast                | BodyPart                  |
|  2 | ultrasound            | ImagingTest               |
|  3 | ovoid mass            | ImagingFindings           |
|  4 | 0.5 x 0.5 x 0.4       | Measurements              |
|  5 | cm                    | Units                     |
|  6 | left                  | Direction                 |
|  7 | shoulder              | BodyPart                  |
|  8 | mass                  | ImagingFindings           |
|  9 | isoechoic echotexture | ImagingFindings           |
| 10 | muscle                | BodyPart                  |
| 11 | internal color flow   | ImagingFindings           |
| 12 | benign fibrous tissue | ImagingFindings           |
| 13 | lipoma                | Disease_Syndrome_Disorder |
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|jsl_rd_ner_wip_greedy_biobert|
|Compatibility:|Healthcare NLP 3.1.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|


## Data Source


Trained on Dataset annotated by John Snow Labs


## Benchmarking


```bash
label                        tp     fp    fn    prec        rec        f1       
B-Units                      253    7     11    0.9730769   0.9583333  0.9656488
B-Medical_Device             382    109   74    0.7780040   0.8377193  0.8067581
B-BodyPart                   2645   347   276   0.8840241   0.9055118  0.8946389
I-BodyPart                   645    142   135   0.819568    0.8269231  0.8232291
B-Imaging_Technique          137    36    33    0.7919075   0.8058823  0.7988338
B-Procedure                  260    93    130   0.7365439   0.6666667  0.6998653
B-Direction                  1573   136   123   0.9204213   0.9274764  0.9239353
I-ImagingTest                30     9     32    0.7692308   0.4838709  0.5940594
I-Test_Result                2      0     0     1           1          1        
B-Measurements               452    24    30    0.9495798   0.9377593  0.9436326
B-ImagingFindings            1929   679   542   0.7396472   0.7806556  0.7595984
B-Test                       146    17    49    0.8957055   0.7487179  0.8156425
B-ManualFix                  2      0     2     1           0.5        0.6666667
I-Procedure                  147    91    106   0.6176470   0.5810277  0.598778 
I-Imaging_Technique          75     63    26    0.5434782   0.7425743  0.6276151
I-Measurements               45     3     6     0.9375      0.8823529  0.9090909
B-ImagingTest                328    36    85    0.9010989   0.7941888  0.8442728
I-Test                       26     9     34    0.7428571   0.4333333  0.5473684
I-Symptom                    138    62    142   0.69        0.4928571  0.575    
I-ImagingFindings            1348   617   662   0.6860051   0.6706468  0.678239 
B-Disease_Syndrome_Disorder  1068   298   243   0.7818448   0.8146453  0.7979080
B-Symptom                    523    110   190   0.8262243   0.7335203  0.7771174
I-Disease_Syndrome_Disorder  377    168   171   0.6917431   0.6879562  0.6898445
I-Medical_Device             369    72    62    0.8367347   0.8561485  0.8463302
I-Direction                  352    38    41    0.9025641   0.8956743  0.899106 
Macro-average	             13272  3200  3313  0.7195612   0.6489194  0.6824170
Micro-average	             13272  3200  3313  0.8057309   0.8002412  0.8029767
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTYxNDE3MDc3M119
-->