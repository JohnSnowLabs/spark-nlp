{%- capture title -%}
Tokenizer
{%- endcapture -%}

{%- capture model_description -%}
Tokenizes raw text into word pieces, tokens. Identifies tokens with tokenization open standards. A few rules will help customizing it if defaults do not fit user needs.

This class represents an already fitted Tokenizer model.

See the main class Tokenizer for more examples of usage.
{%- endcapture -%}

{%- capture model_input_anno -%}
DOCUMENT //A Tokenizer could require only for now a SentenceDetector annotator
{%- endcapture -%}

{%- capture model_output_anno -%}
TOKEN
{%- endcapture -%}

{%- capture model_api_link -%}
[TokenizerModel](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/TokenizerModel)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[TokenizerModel](/api/python/reference/autosummary/python/sparknlp/annotator/token/tokenizer/index.html#sparknlp.annotator.token.tokenizer.TokenizerModel)
{%- endcapture -%}

{%- capture model_source_link -%}
[TokenizerModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/TokenizerModel.scala)
{%- endcapture -%}

{%- capture approach_description -%}
Tokenizes raw text in document type columns into TokenizedSentence .

This class represents a non fitted tokenizer. Fitting it will cause the internal RuleFactory to construct the rules for tokenizing from the input configuration.

Identifies tokens with tokenization open standards. A few rules will help customizing it if defaults do not fit user needs.

For extended examples of usage see the
[Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb)
and [Tokenizer test class](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/TokenizerTestSpec.scala)
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

data = spark.createDataFrame([["I'd like to say we didn't expect that. Jane's boyfriend."]]).toDF("text")
documentAssembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("token").fit(data)

pipeline = Pipeline().setStages([documentAssembler, tokenizer]).fit(data)
result = pipeline.transform(data)

result.selectExpr("token.result").show(truncate=False)
+-----------------------------------------------------------------------+
|output                                                                 |
+-----------------------------------------------------------------------+
|[I'd, like, to, say, we, didn't, expect, that, ., Jane's, boyfriend, .]|
+-----------------------------------------------------------------------+

{%- endcapture -%}

{%- capture approach_scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import org.apache.spark.ml.Pipeline

val data = Seq("I'd like to say we didn't expect that. Jane's boyfriend.").toDF("text")
val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")
val tokenizer = new Tokenizer().setInputCols("document").setOutputCol("token").fit(data)

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer)).fit(data)
val result = pipeline.transform(data)

result.selectExpr("token.result").show(false)
+-----------------------------------------------------------------------+
|output                                                                 |
+-----------------------------------------------------------------------+
|[I'd, like, to, say, we, didn't, expect, that, ., Jane's, boyfriend, .]|
+-----------------------------------------------------------------------+

{%- endcapture -%}

{%- capture approach_api_link -%}
[Tokenizer](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/Tokenizer)
{%- endcapture -%}

{%- capture approach_python_api_link -%}
[Tokenizer](/api/python/reference/autosummary/python/sparknlp/annotator/token/tokenizer/index.html#sparknlp.annotator.token.tokenizer.Tokenizer)
{%- endcapture -%}

{%- capture approach_source_link -%}
[Tokenizer](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/Tokenizer.scala)
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
approach_note="All these APIs receive regular expressions so please make sure that you escape special characters according to Java conventions."
%}
