---
layout: model
title: Deidentification NER (Large)
author: John Snow Labs
name: ner_deid_large
class: NerDLModel
language: en
repository: clinical/models
date: 2020-07-22
tags: [clinical,ner,deidentify,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description

Deidentification NER (Large) is a Named Entity Recognition model that annotates text to find protected health information that may need to be deidentified. The entities it annotates are Age, Contact, Date, Id, Location, Name, and Profession. This model is trained with the 'embeddings_clinical' word embeddings model, so be sure to use the same embeddings in the pipeline.

## Predicted Entities 
Age, Contact, Date, Id, Location, Name, Profession

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_DEMOGRAPHICS){:.button.button-orange}[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_DEMOGRAPHICS.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_large_en_2.5.3_2.4_1595427435246.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = NerDLModel.pretrained("ner_deid_large","en","clinical/models")
	.setInputCols("sentence","token","word_embeddings")
	.setOutputCol("ner")

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, model, ner_converter])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

input_text = [
    """HISTORY OF PRESENT ILLNESS: Mr. Smith is a 60-year-old white male veteran with multiple comorbidities, who has a history of bladder cancer diagnosed approximately two years ago by the VA Hospital. He underwent a resection there. He was to be admitted to the Day Hospital for cystectomy. He was seen in Urology Clinic and Radiology Clinic on 02/04/2003.

HOSPITAL COURSE: Mr. Smith presented to the Day Hospital in anticipation for Urology surgery. On evaluation, EKG, echocardiogram was abnormal, a Cardiology consult was obtained. A cardiac adenosine stress MRI was then proceeded, same was positive for inducible ischemia, mild-to-moderate inferolateral subendocardial infarction with peri-infarct ischemia. In addition, inducible ischemia seen in the inferior lateral septum. Mr. Smith underwent a left heart catheterization, which revealed two vessel coronary artery disease. The RCA, proximal was 95% stenosed and the distal 80% stenosed. The mid LAD was 85% stenosed and the distal LAD was 85% stenosed. There was four Multi-Link Vision bare metal stents placed to decrease all four lesions to 0%. Following intervention, Mr. Smith was admitted to 7 Ardmore Tower under Cardiology Service under the direction of Dr. Hart. Mr. Smith had a noncomplicated post-intervention hospital course. He was stable for discharge home on 02/07/2003 with instructions to take Plavix daily for one month and Urology is aware of the same."""
]
result = pipeline_model.transform(spark.createDataFrame(pd.DataFrame({"text": input_text})))
```

```scala
val model = NerDLModel.pretrained("ner_deid_large","en","clinical/models")
	.setInputCols("sentence","token","word_embeddings")
	.setOutputCol("ner")
```
</div>

## Results
```bash
+---------------+---------+
|chunk          |ner_label|
+---------------+---------+
|Smith          |PATIENT  |
|Smith          |PATIENT  |
|VA Hospital    |HOSPITAL |
|Day Hospital   |HOSPITAL |
|02/04/2003     |DATE     |
|Smith          |PATIENT  |
|Day Hospital   |HOSPITAL |
|Smith          |PATIENT  |
|Smith          |PATIENT  |
|7 Ardmore Tower|STREET   |
|Hart           |DOCTOR   |
|Smith          |PATIENT  |
|02/07/2003     |DATE     |
+---------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|----------------|----------------------------------|
| Name:           | ner_deid_large                   |
| Type:    | NerDLModel                       |
| Compatibility:  | Spark NLP 2.4.2+                           |
| License:        | Licensed                         |
|Edition:|Official|                       |
|Input labels:         | [sentence, token, word_embeddings] |
|Output labels:        | [ner]                              |
| Language:       | en                               |
| Case sensitive: | False                            |
| Dependencies:  | embeddings_clinical              |

{:.h2_title}
## Data Source
Trained on plain n2c2 2014: De-identification and Heart Disease Risk Factors Challenge datasets with `embeddings_clinical`
https://portal.dbmi.hms.harvard.edu/projects/n2c2-2014/