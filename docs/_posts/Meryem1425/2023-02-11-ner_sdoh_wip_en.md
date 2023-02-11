---
layout: model
title: Social Determinants of Health
author: John Snow Labs
name: ner_sdoh_wip
date: 2023-02-11
tags: [licensed, clinical, en, social_determinants, ner, public_health, sdoh]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 4.2.8
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model extracts terminology related to Social Determinants of Health from various kinds of biomedical documents.

## Predicted Entities

`Other_SDoH_Keywords`, `Education`, `Population_Group`, `Quality_Of_Life`, `Housing`, `Substance_Frequency`, `Smoking`, `Eating_Disorder`, `Obesity`, `Healthcare_Institution`, `Financial_Status`, `Age`, `Chidhood_Event`, `Exercise`, `Communicable_Disease`, `Hypertension`, `Other_Disease`, `Violence_Or_Abuse`, `Spiritual_Beliefs`, `Employment`, `Social_Exclusion`, `Access_To_Care`, `Marital_Status`, `Diet`, `Social_Support`, `Disability`, `Mental_Health`, `Alcohol`, `Insurance_Status`, `Substance_Quantity`, `Hyperlipidemia`, `Family_Member`, `Legal_Issues`, `Race_Ethnicity`, `Gender`, `Geographic_Entity`, `Sexual_Orientation`, `Transportation`, `Sexual_Activity`, `Language`, `Substance_Use`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_sdoh_wip_en_4.2.8_3.0_1676135569606.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_sdoh_wip_en_4.2.8_3.0_1676135569606.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "en")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

clinical_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

ner_model = MedicalNerModel.pretrained("ner_sdoh_wip", "en", "clinical/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")

ner_converter = NerConverterInternal()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

pipeline = Pipeline(stages=[
    document_assembler, 
    sentence_detector,
    tokenizer,
    clinical_embeddings,
    ner_model,
    ner_converter   
    ])

sample_texts = [["Smith is a 55 years old, divorced Mexcian American woman with financial problems. She speaks spanish. She lives in an apartment. She has been struggling with diabetes for the past 10 years and has recently been experiencing frequent hospitalizations due to uncontrolled blood sugar levels. Smith works as a cleaning assistant and does not have access to health insurance or paid sick leave. She has a son student at college. Pt with likely long-standing depression. She is aware she needs rehab. Pt reprots having her catholic faith as a means of support as well.  She has long history of etoh abuse, beginning in her teens. She reports she has been a daily drinker for 30 years, most recently drinking beer daily. She smokes a pack of cigarettes a day. She had DUI back in April and was due to be in court this week."]]
             
data = spark.createDataFrame(sample_texts).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "en")
    .setInputCols("document")
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val clinical_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val ner_model = MedicalNerModel.pretrained("ner_sdoh_wip", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverterInternal()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(
    document_assembler, 
    sentence_detector,
    tokenizer,
    clinical_embeddings,
    ner_model,
    ner_converter   
))

val data = Seq("He continues to smoke one pack of cigarettes daily, as he has for the past 28 years.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+------------------+-----+---+-------------------+
|chunk             |begin|end|ner_label          |
+------------------+-----+---+-------------------+
|55 years old      |11   |22 |Age                |
|divorced          |25   |32 |Marital_Status     |
|Mexcian American  |34   |49 |Race_Ethnicity     |
|woman             |51   |55 |Gender             |
|financial problems|62   |79 |Financial_Status   |
|She               |82   |84 |Gender             |
|spanish           |93   |99 |Language           |
|She               |102  |104|Gender             |
|apartment         |118  |126|Housing            |
|She               |129  |131|Gender             |
|diabetes          |158  |165|Other_Disease      |
|cleaning assistant|307  |324|Employment         |
|health insurance  |354  |369|Insurance_Status   |
|She               |391  |393|Gender             |
|son               |401  |403|Family_Member      |
|student           |405  |411|Education          |
|college           |416  |422|Education          |
|depression        |454  |463|Mental_Health      |
|She               |466  |468|Gender             |
|she               |479  |481|Gender             |
|rehab             |489  |493|Access_To_Care     |
|her               |514  |516|Gender             |
|catholic faith    |518  |531|Spiritual_Beliefs  |
|support           |547  |553|Social_Support     |
|She               |565  |567|Gender             |
|etoh abuse        |589  |598|Alcohol            |
|her               |614  |616|Gender             |
|teens             |618  |622|Age                |
|She               |625  |627|Gender             |
|she               |637  |639|Gender             |
|drinker           |658  |664|Alcohol            |
|drinking beer     |694  |706|Alcohol            |
|daily             |708  |712|Substance_Frequency|
|She               |715  |717|Gender             |
|smokes            |719  |724|Smoking            |
|a pack            |726  |731|Substance_Quantity |
|cigarettes        |736  |745|Smoking            |
|a day             |747  |751|Substance_Frequency|
|She               |754  |756|Gender             |
|DUI               |762  |764|Legal_Issues       |
+------------------+-----+---+-------------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_sdoh_wip|
|Compatibility:|Healthcare NLP 4.2.8+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|1.5 MB|

## Benchmarking

```bash
                 label	    tp	   fp	    fn	 total precision	  recall	      f1
   Other_SDoH_Keywords	 317.0	 84.0	 112.0	 429.0	0.790524	0.738928	0.763855
             Education	 116.0	 38.0	  31.0	 147.0	0.753247	0.789116	0.770764
      Population_Group	  27.0	  4.0	  11.0	  38.0	0.870968	0.710526	0.782609
       Quality_Of_Life	 146.0	 37.0	  57.0	 203.0	0.797814	0.719212	0.756477
               Housing	 809.0	 91.0	 118.0	 927.0	0.898889	0.872708	0.885605
   Substance_Frequency	  74.0	 19.0	  44.0	 118.0	0.795699	0.627119	0.701422
               Smoking	 136.0	  4.0	   2.0	 138.0	0.971429	0.985507	0.978417
       Eating_Disorder	  40.0	  2.0	   0.0	  40.0	0.952381	1.000000	0.975610
               Obesity	  16.0	  1.0	   5.0	  21.0	0.941176	0.761905	0.842105
Healthcare_Institution	 117.0	 36.0	  57.0	 174.0	0.764706	0.672414	0.715596
      Financial_Status	 222.0	 47.0	 128.0	 350.0	0.825279	0.634286	0.717286
                   Age	1328.0	109.0	  48.0	1376.0	0.924148	0.965116	0.944188
        Chidhood_Event	  30.0	  0.0	  24.0	  54.0	1.000000	0.555556	0.714286
              Exercise	  52.0	 17.0	  31.0	  83.0	0.753623	0.626506	0.684211
  Communicable_Disease	  61.0	  5.0	  10.0	  71.0	0.924242	0.859155	0.890511
          Hypertension	  45.0	  1.0	  12.0	  57.0	0.978261	0.789474	0.873786
         Other_Disease	1065.0	229.0	 119.0	1184.0	0.823029	0.899493	0.859564
     Violence_Or_Abuse	  98.0	 26.0	  53.0	 151.0	0.790323	0.649007	0.712727
     Spiritual_Beliefs	  94.0	  9.0	  21.0	 115.0	0.912621	0.817391	0.862385
            Employment	3797.0	272.0	 288.0	4085.0	0.933153	0.929498	0.931322
      Social_Exclusion	  38.0	  6.0	  14.0	  52.0	0.863636	0.730769	0.791667
        Access_To_Care	 810.0	 95.0	 160.0	 970.0	0.895028	0.835052	0.864000
        Marital_Status	 177.0	  4.0	   9.0	 186.0	0.977901	0.951613	0.964578
                  Diet	 110.0	 34.0	  30.0	 140.0	0.763889	0.785714	0.774648
        Social_Support	1243.0	197.0	  99.0	1342.0	0.863194	0.926230	0.893602
            Disability	  94.0	  4.0	   9.0	 103.0	0.959184	0.912621	0.935323
         Mental_Health	 817.0	 99.0	 216.0	1033.0	0.891921	0.790900	0.838379
               Alcohol	 592.0	 32.0	  28.0	 620.0	0.948718	0.954839	0.951768
      Insurance_Status	 145.0	 23.0	  32.0	 177.0	0.863095	0.819209	0.840580
    Substance_Quantity	 107.0	 42.0	  39.0	 146.0	0.718121	0.732877	0.725424
        Hyperlipidemia	  14.0	  1.0	   2.0	  16.0	0.933333	0.875000	0.903226
         Family_Member	4255.0	110.0	  73.0	4328.0	0.974800	0.983133	0.978949
          Legal_Issues	  71.0	 13.0	  20.0	  91.0	0.845238	0.780220	0.811429
        Race_Ethnicity	  81.0	  9.0	   7.0	  88.0	0.900000	0.920455	0.910112
                Gender	9698.0	183.0	 193.0	9891.0	0.981480	0.980487	0.980983
     Geographic_Entity	 189.0	 18.0	  22.0	 211.0	0.913043	0.895735	0.904306
    Sexual_Orientation	  21.0	  0.0	   3.0	  24.0	1.000000	0.875000	0.933333
        Transportation	  27.0	  2.0	  27.0	  54.0	0.931034	0.500000	0.650602
       Sexual_Activity	  56.0	  4.0	  24.0	  80.0	0.933333	0.700000	0.800000
              Language	  35.0	  6.0	   2.0	  37.0	0.853659	0.945946	0.897436
         Substance_Use	 400.0	 40.0	  24.0	 424.0	0.909091	0.943396	0.925926
```
