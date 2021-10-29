{%- capture title -%}
DrugNormalizer
{%- endcapture -%}

{%- capture description -%}
Annotator which normalizes raw text from clinical documents, e.g. scraped web pages or xml documents, from document type columns into Sentence.
Removes all dirty characters from text following one or more input regex patterns.
Can apply non wanted character removal which a specific policy.
Can apply lower case normalization.

See [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/23.Drug_Normalizer.ipynb) for more examples of usage.
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.common import *
from sparknlp.annotator import *
from sparknlp.training import *
import sparknlp_jsl
from sparknlp_jsl.base import *
from sparknlp_jsl.annotator import *
from pyspark.ml import Pipeline
data = spark.createDataFrame([
  ["Sodium Chloride/Potassium Chloride 13bag"],
  ["interferon alfa-2b 10 million unit ( 1 ml ) injec"],
  ["aspirin 10 meq/ 5 ml oral sol"]
]).toDF("text")
document = DocumentAssembler().setInputCol("text").setOutputCol("document")
drugNormalizer = DrugNormalizer().setInputCols(["document"]).setOutputCol("document_normalized")

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

{%- capture scala_example -%}
val data = Seq(
  ("Sodium Chloride/Potassium Chloride 13bag"),
  ("interferon alfa-2b 10 million unit ( 1 ml ) injec"),
  ("aspirin 10 meq/ 5 ml oral sol")
).toDF("text")
val document = new DocumentAssembler().setInputCol("text").setOutputCol("document")
val drugNormalizer = new DrugNormalizer().setInputCols("document").setOutputCol("document_normalized")

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

{%- capture api_link -%}
[DrugNormalizer](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/DrugNormalizer)
{%- endcapture -%}

{% include templates/licensed_anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
api_link=api_link%}