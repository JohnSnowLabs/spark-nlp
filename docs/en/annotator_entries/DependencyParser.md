{%- capture title -%}
DependencyParser
{%- endcapture -%}

{%- capture model_description -%}
Unlabeled parser that finds a grammatical relation between two words in a sentence.

Dependency parser provides information about word relationship. For example, dependency parsing can tell you what
the subjects and objects of a verb are, as well as which words are modifying (describing) the subject. This can help
you find precise answers to specific questions.

This is the instantiated model of the DependencyParserApproach.
For training your own model, please see the documentation of that class.

Pretrained models can be loaded with `pretrained` of the companion object:
```
val dependencyParserApproach = DependencyParserModel.pretrained()
  .setInputCols("sentence", "pos", "token")
  .setOutputCol("dependency")
```
The default model is `"dependency_conllu"`, if no name is provided.
For available pretrained models please see the [Models Hub](https://nlp.johnsnowlabs.com/models).

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/3.SparkNLP_Pretrained_Models.ipynb)
and the [DependencyParserApproachTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/parser/dep/DependencyParserApproachTestSpec.scala).
{%- endcapture -%}

{%- capture model_input_anno -%}
[String]DOCUMENT, POS, TOKEN
{%- endcapture -%}

{%- capture model_output_anno -%}
DEPENDENCY
{%- endcapture -%}

{%- capture model_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentence = SentenceDetector() \
    .setInputCols("document") \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols("sentence") \
    .setOutputCol("token")

posTagger = PerceptronModel.pretrained() \
    .setInputCols("sentence", "token") \
    .setOutputCol("pos")

dependencyParser = DependencyParserModel.pretrained() \
    .setInputCols("sentence", "pos", "token") \
    .setOutputCol("dependency")

pipeline = Pipeline().setStages([
    documentAssembler,
    sentence,
    tokenizer,
    posTagger,
    dependencyParser
])

data = spark.createDataFrame([[
    "Unions representing workers at Turner Newall say they are 'disappointed' after talks with stricken parent " +
      "firm Federal Mogul."
]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.selectExpr("explode(arrays_zip(token.result, dependency.result)) as cols") \
    .selectExpr("cols['0'] as token", "cols['1'] as dependency").show(8, truncate = False)
+------------+------------+
|token       |dependency  |
+------------+------------+
|Unions      |ROOT        |
|representing|workers     |
|workers     |Unions      |
|at          |Turner      |
|Turner      |workers     |
|Newall      |say         |
|say         |Unions      |
|they        |disappointed|
+------------+------------+

{%- endcapture -%}

{%- capture model_scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentence = new SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")

val tokenizer = new Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

val posTagger = PerceptronModel.pretrained()
  .setInputCols("sentence", "token")
  .setOutputCol("pos")

val dependencyParser = DependencyParserModel.pretrained()
  .setInputCols("sentence", "pos", "token")
  .setOutputCol("dependency")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentence,
  tokenizer,
  posTagger,
  dependencyParser
))

val data = Seq(
  "Unions representing workers at Turner Newall say they are 'disappointed' after talks with stricken parent " +
    "firm Federal Mogul."
).toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("explode(arrays_zip(token.result, dependency.result)) as cols")
  .selectExpr("cols['0'] as token", "cols['1'] as dependency").show(8, truncate = false)
+------------+------------+
|token       |dependency  |
+------------+------------+
|Unions      |ROOT        |
|representing|workers     |
|workers     |Unions      |
|at          |Turner      |
|Turner      |workers     |
|Newall      |say         |
|say         |Unions      |
|they        |disappointed|
+------------+------------+

{%- endcapture -%}

{%- capture model_api_link -%}
[DependencyParserModel](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/parser/dep/DependencyParserModel)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[DependencyParserModel](/api/python/reference/autosummary/python/sparknlp/annotator/dependency/dependency_parser/index.html#sparknlp.annotator.dependency.dependency_parser.DependencyParserModel)
{%- endcapture -%}

{%- capture model_source_link -%}
[DependencyParserModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/parser/dep/DependencyParserModel.scala)
{%- endcapture -%}

{%- capture approach_description -%}
Trains an unlabeled parser that finds a grammatical relations between two words in a sentence.

For instantiated/pretrained models, see DependencyParserModel.

Dependency parser provides information about word relationship. For example, dependency parsing can tell you what
the subjects and objects of a verb are, as well as which words are modifying (describing) the subject. This can help
you find precise answers to specific questions.

The required training data can be set in two different ways (only one can be chosen for a particular model):
  - Dependency treebank in the [Penn Treebank format](http://www.nltk.org/nltk_data/) set with `setDependencyTreeBank`
  - Dataset in the [CoNLL-U format](https://universaldependencies.org/format.html) set with `setConllU`

Apart from that, no additional training data is needed.

See [DependencyParserApproachTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/parser/dep/DependencyParserApproachTestSpec.scala) for further reference on how to use this API.
{%- endcapture -%}

{%- capture approach_input_anno -%}
DOCUMENT, POS, TOKEN
{%- endcapture -%}

{%- capture approach_output_anno -%}
DEPENDENCY
{%- endcapture -%}

{%- capture approach_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentence = SentenceDetector() \
    .setInputCols("document") \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols("sentence") \
    .setOutputCol("token")

posTagger = PerceptronModel.pretrained() \
    .setInputCols("sentence", "token") \
    .setOutputCol("pos")

dependencyParserApproach = DependencyParserApproach() \
    .setInputCols("sentence", "pos", "token") \
    .setOutputCol("dependency") \
    .setDependencyTreeBank("src/test/resources/parser/unlabeled/dependency_treebank")

pipeline = Pipeline().setStages([
    documentAssembler,
    sentence,
    tokenizer,
    posTagger,
    dependencyParserApproach
])

# Additional training data is not needed, the dependency parser relies on the dependency tree bank / CoNLL-U only.
emptyDataSet = spark.createDataFrame([[""]]).toDF("text")
pipelineModel = pipeline.fit(emptyDataSet)

{%- endcapture -%}

{%- capture approach_scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserApproach
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentence = new SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")

val tokenizer = new Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

val posTagger = PerceptronModel.pretrained()
  .setInputCols("sentence", "token")
  .setOutputCol("pos")

val dependencyParserApproach = new DependencyParserApproach()
  .setInputCols("sentence", "pos", "token")
  .setOutputCol("dependency")
  .setDependencyTreeBank("src/test/resources/parser/unlabeled/dependency_treebank")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentence,
  tokenizer,
  posTagger,
  dependencyParserApproach
))

// Additional training data is not needed, the dependency parser relies on the dependency tree bank / CoNLL-U only.
val emptyDataSet = Seq.empty[String].toDF("text")
val pipelineModel = pipeline.fit(emptyDataSet)

{%- endcapture -%}

{%- capture approach_api_link -%}
[DependencyParserApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/parser/dep/DependencyParserApproach)
{%- endcapture -%}

{%- capture approach_python_api_link -%}
[DependencyParserApproach](/api/python/reference/autosummary/python/sparknlp/annotator/dependency/dependency_parser/index.html#sparknlp.annotator.dependency.dependency_parser.DependencyParserApproach)
{%- endcapture -%}

{%- capture approach_source_link -%}
[DependencyParserApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/parser/dep/DependencyParserApproach.scala)
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
