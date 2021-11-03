{%- capture title -%}
GenericClassifier
{%- endcapture -%}

{%- capture description -%}
Trains a TensorFlow model for generic classification of feature vectors. It takes FEATURE_VECTOR annotations from
`FeaturesAssembler` as input, classifies them and outputs CATEGORY annotations.
Please see the Parameters section for required training parameters.

For a more extensive example please see the
[Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/8.Generic_Classifier.ipynb).
{%- endcapture -%}

{%- capture input_anno -%}
FEATURE_VECTOR
{%- endcapture -%}

{%- capture output_anno -%}
CATEGORY
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
features_asm = FeaturesAssembler() \
    .setInputCols(["feature_1", "feature_2", "...", "feature_n"]) \
    .setOutputCol("features")

gen_clf = GenericClassifierApproach() \
    .setLabelColumn("target") \
    .setInputCols(["features"]) \
    .setOutputCol("prediction") \
    .setModelFile("/path/to/graph_file.pb") \
    .setEpochsNumber(50) \
    .setBatchSize(100) \
    .setFeatureScaling("zscore") \
    .setlearningRate(0.001) \
    .setFixImbalance(True) \
    .setOutputLogsPath("logs") \
    .setValidationSplit(0.2) # keep 20% of the data for validation purposes

pipeline = Pipeline().setStages([
    features_asm,
    gen_clf
])

clf_model = pipeline.fit(data)

{%- endcapture -%}

{%- capture approach_scala_example -%}
val features_asm = new FeaturesAssembler()
  .setInputCols(Array("feature_1", "feature_2", "...", "feature_n"))
  .setOutputCol("features")

val gen_clf = new GenericClassifierApproach()
  .setLabelColumn("target")
  .setInputCols("features")
  .setOutputCol("prediction")
  .setModelFile("/path/to/graph_file.pb")
  .setEpochsNumber(50)
  .setBatchSize(100)
  .setFeatureScaling("zscore")
  .setlearningRate(0.001f)
  .setFixImbalance(true)
  .setOutputLogsPath("logs")
  .setValidationSplit(0.2f) // keep 20% of the data for validation purposes

val pipeline = new Pipeline().setStages(Array(
  features_asm,
  gen_clf
))

val clf_model = pipeline.fit(data)

{%- endcapture -%}

{%- capture api_link -%}
[GenericClassifierApproach](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/generic_classifier/GenericClassifierApproach)
{%- endcapture -%}

{%- capture python_api_link -%}
[GenericClassifierApproach](https://nlp.johnsnowlabs.comlicensed/api/python/reference/autosummary/sparknlp_jsl.annotator.GenericClassifierApproach.html)

{%- endcapture -%}


{% include templates/licensed_training_anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
python_api_link=python_api_link
api_link=api_link
%}
