{%- capture title -%}
DrugNormalizer
{%- endcapture -%}

{%- capture model -%}
model
{%- endcapture -%}

{%- capture model_description -%}
Annotator which normalizes raw text from clinical documents, e.g. scraped web pages or xml documents, from document type columns into Sentence.
Removes all dirty characters from text following one or more input regex patterns.
Can apply non wanted character removal which a specific policy.
Can apply lower case normalization.

See [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/23.Drug_Normalizer.ipynb) for more examples of usage.
{%- endcapture -%}

{%- capture model_input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture model_output_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture model_python_medical -%}
from johnsnowlabs import *
data = spark.createDataFrame([
  ["Sodium Chloride/Potassium Chloride 13bag"],
  ["interferon alfa-2b 10 million unit ( 1 ml ) injec"],
  ["aspirin 10 meq/ 5 ml oral sol"]
]).toDF("text")
document = nlp.DocumentAssembler().setInputCol("text").setOutputCol("document")
drugNormalizer = medical.DrugNormalizer().setInputCols(["document"]).setOutputCol("document_normalized")

trainingPipeline = Pipeline(stages=[document, drugNormalizer])
result = trainingPipeline.fit(data).transform(data)

result.selectExpr("explode(document_normalized.result) as normalized_text").show(truncate=False)
+----------------------------------------------------+
|normalized_text                                     |
+----------------------------------------------------+
|Sodium Chloride / Potassium Chloride 13 bag         |
|interferon alfa - 2b 10000000 unt ( 1 ml ) injection|
|aspirin 2 meq/ml oral solution                      |
+----------------------------------------------------+

{%- endcapture -%}

{%- capture model_python_legal -%}
from johnsnowlabs import *

document = nlp.DocumentAssembler().setInputCol("text").setOutputCol("document")
drugNormalizer = legal.DrugNormalizer().setInputCols(["document"]).setOutputCol("document_normalized")

trainingPipeline = Pipeline(stages=[document, drugNormalizer])
{%- endcapture -%}

{%- capture model_python_finance -%}
from johnsnowlabs import *

document = nlp.DocumentAssembler().setInputCol("text").setOutputCol("document")
drugNormalizer = finance.DrugNormalizer().setInputCols(["document"]).setOutputCol("document_normalized")

trainingPipeline = Pipeline(stages=[document, drugNormalizer])
{%- endcapture -%}



{%- capture model_scala_medical -%}
from johnsnowlabs import * 
val data = Seq(
  ("Sodium Chloride/Potassium Chloride 13bag"),
  ("interferon alfa-2b 10 million unit ( 1 ml ) injec"),
  ("aspirin 10 meq/ 5 ml oral sol")
).toDF("text")
val document = new nlp.DocumentAssembler().setInputCol("text").setOutputCol("document")
val drugNormalizer = new medical.DrugNormalizer().setInputCols("document").setOutputCol("document_normalized")

val trainingPipeline = new Pipeline().setStages(Array(document, drugNormalizer))
val result = trainingPipeline.fit(data).transform(data)

result.selectExpr("explode(document_normalized.result) as normalized_text").show(false)
+----------------------------------------------------+
|normalized_text                                     |
+----------------------------------------------------+
|Sodium Chloride / Potassium Chloride 13 bag         |
|interferon alfa - 2b 10000000 unt ( 1 ml ) injection|
|aspirin 2 meq/ml oral solution                      |
+----------------------------------------------------+

{%- endcapture -%}

{%- capture model_scala_legal -%}
from johnsnowlabs import * 

val document = new nlp.DocumentAssembler().setInputCol("text").setOutputCol("document")
val drugNormalizer = new legal.DrugNormalizer().setInputCols("document").setOutputCol("document_normalized")

val trainingPipeline = new Pipeline().setStages(Array(document, drugNormalizer))

{%- endcapture -%}

{%- capture model_scala_finance -%}
from johnsnowlabs import * 

val document = new nlp.DocumentAssembler().setInputCol("text").setOutputCol("document")
val drugNormalizer = new finance.DrugNormalizer().setInputCols("document").setOutputCol("document_normalized")

val trainingPipeline = new Pipeline().setStages(Array(document, drugNormalizer))

{%- endcapture -%}


{%- capture model_api_link -%}
[DrugNormalizer](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/DrugNormalizer)
{%- endcapture -%}

{% include templates/licensed_approach_model_medical_fin_leg_template.md
title=title
model=model
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_medical=model_python_medical
model_python_legal=model_python_legal
model_python_finance=model_python_finance
model_scala_medical=model_scala_medical
model_scala_legal=model_scala_legal
model_scala_finance=model_scala_finance
model_api_link=model_api_link%}