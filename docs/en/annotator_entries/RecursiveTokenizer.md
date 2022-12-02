{%- capture title -%}
RecursiveTokenizer
{%- endcapture -%}

{%- capture model_description -%}
Instantiated model of the RecursiveTokenizer.
For usage and examples see the documentation of the main class.
{%- endcapture -%}

{%- capture model_input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture model_output_anno -%}
TOKEN
{%- endcapture -%}

{%- capture model_api_link -%}
[RecursiveTokenizerModel](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/RecursiveTokenizerModel)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[RecursiveTokenizerModel](/api/python/reference/autosummary/python/sparknlp/annotator/token/recursive_tokenizer/index.html#sparknlp.annotator.token.recursive_tokenizer.RecursiveTokenizerModel)
{%- endcapture -%}

{%- capture model_source_link -%}
[RecursiveTokenizerModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/RecursiveTokenizerModel.scala)
{%- endcapture -%}

{%- capture approach_description -%}
Tokenizes raw text recursively based on a handful of definable rules.

Unlike the Tokenizer, the RecursiveTokenizer operates based on these array string parameters only:
 - `prefixes`: Strings that will be split when found at the beginning of token.
 - `suffixes`: Strings that will be split when found at the end of token.
 - `infixes`: Strings that will be split when found at the middle of token.
 - `whitelist`: Whitelist of strings not to split

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/7.Context_Spell_Checker.ipynb)
and the [TokenizerTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/TokenizerTestSpec.scala).
{%- endcapture -%}

{%- capture approach_input_anno -%}
DOCUMENT
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

tokenizer = RecursiveTokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer
])

data = spark.createDataFrame([["One, after the Other, (and) again. PO, QAM,"]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.select("token.result").show(truncate=False)
+------------------------------------------------------------------+
|result                                                            |
+------------------------------------------------------------------+
|[One, ,, after, the, Other, ,, (, and, ), again, ., PO, ,, QAM, ,]|
+------------------------------------------------------------------+

{%- endcapture -%}

{%- capture approach_scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.RecursiveTokenizer
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new RecursiveTokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer
))

val data = Seq("One, after the Other, (and) again. PO, QAM,").toDF("text")
val result = pipeline.fit(data).transform(data)

result.select("token.result").show(false)
+------------------------------------------------------------------+
|result                                                            |
+------------------------------------------------------------------+
|[One, ,, after, the, Other, ,, (, and, ), again, ., PO, ,, QAM, ,]|
+------------------------------------------------------------------+

{%- endcapture -%}

{%- capture approach_api_link -%}
[RecursiveTokenizer](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/RecursiveTokenizer)
{%- endcapture -%}

{%- capture approach_python_api_link -%}
[RecursiveTokenizer](/api/python/reference/autosummary/python/sparknlp/annotator/token/recursive_tokenizer/index.html#sparknlp.annotator.token.recursive_tokenizer.RecursiveTokenizer)
{%- endcapture -%}

{%- capture approach_source_link -%}
[RecursiveTokenizer](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/RecursiveTokenizer.scala)
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
