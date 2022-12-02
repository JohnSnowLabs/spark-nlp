{%- capture title -%}
Normalizer
{%- endcapture -%}

{%- capture model_description -%}
Instantiated Model of the Normalizer. For usage and examples, please see the documentation of that class.
{%- endcapture -%}

{%- capture model_input_anno -%}
TOKEN
{%- endcapture -%}

{%- capture model_output_anno -%}
TOKEN
{%- endcapture -%}

{%- capture model_api_link -%}
[NormalizerModel](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/NormalizerModel)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[NormalizerModel](/api/python/reference/autosummary/python/sparknlp/annotator/normalizer/index.html#sparknlp.annotator.normalizer.NormalizerModel)
{%- endcapture -%}

{%- capture model_source_link -%}
[NormalizerModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/NormalizerModel.scala)
{%- endcapture -%}

{%- capture approach_description -%}
Annotator that cleans out tokens. Requires stems, hence tokens.
Removes all dirty characters from text following a regex pattern and transforms words based on a provided dictionary

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb).
{%- endcapture -%}

{%- capture approach_input_anno -%}
TOKEN
{%- endcapture -%}

{%- capture approach_output_anno -%}
TOKEN
{%- endcapture -%}

{%- capture approach_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normalized") \
    .setLowercase(True) \
    .setCleanupPatterns(["""[^\w\d\s]"""]) # remove punctuations (keep alphanumeric chars)
# if we don't set CleanupPatterns, it will only keep alphabet letters ([^A-Za-z])

pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    normalizer
])

data = spark.createDataFrame([["John and Peter are brothers. However they don't support each other that much."]]) \
    .toDF("text")
result = pipeline.fit(data).transform(data)

result.selectExpr("normalized.result").show(truncate = False)
+----------------------------------------------------------------------------------------+
|result                                                                                  |
+----------------------------------------------------------------------------------------+
|[john, and, peter, are, brothers, however, they, dont, support, each, other, that, much]|
+----------------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture approach_scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.{Normalizer, Tokenizer}
import org.apache.spark.ml.Pipeline
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val normalizer = new Normalizer()
  .setInputCols("token")
  .setOutputCol("normalized")
  .setLowercase(true)
  .setCleanupPatterns(Array("""[^\w\d\s]""")) // remove punctuations (keep alphanumeric chars)
// if we don't set CleanupPatterns, it will only keep alphabet letters ([^A-Za-z])

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  normalizer
))

val data = Seq("John and Peter are brothers. However they don't support each other that much.")
  .toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("normalized.result").show(truncate = false)
+----------------------------------------------------------------------------------------+
|result                                                                                  |
+----------------------------------------------------------------------------------------+
|[john, and, peter, are, brothers, however, they, dont, support, each, other, that, much]|
+----------------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture approach_api_link -%}
[Normalizer](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/Normalizer)
{%- endcapture -%}

{%- capture approach_python_api_link -%}
[Normalizer](/api/python/reference/autosummary/python/sparknlp/annotator/normalizer/index.html#sparknlp.annotator.normalizer.Normalizer)
{%- endcapture -%}

{%- capture approach_source_link -%}
[Normalizer](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/Normalizer.scala)
{%- endcapture -%}


{% include templates/approach_model_template.md
title=title
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_api_link=model_python_api_link
model_api_link=model_api_link
model_source_link=model_source_link
approach_description=approach_description
approach_input_anno=approach_input_anno
approach_output_anno=approach_output_anno
approach_python_example=approach_python_example
approach_scala_example=approach_scala_example
approach_python_api_link=approach_python_api_link
approach_api_link=approach_api_link
approach_source_link=approach_source_link
%}
