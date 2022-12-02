{%- capture title -%}
TypedDependencyParser
{%- endcapture -%}

{%- capture model_description -%}
Labeled parser that finds a grammatical relation between two words in a sentence.
Its input is either a CoNLL2009 or ConllU dataset.

Dependency parsers provide information about word relationship. For example, dependency parsing can tell you what
the subjects and objects of a verb are, as well as which words are modifying (describing) the subject. This can help
you find precise answers to specific questions.

The parser requires the dependant tokens beforehand with e.g. [DependencyParser](/docs/en/annotators#dependencyparser).

Pretrained models can be loaded with `pretrained` of the companion object:
```
val typedDependencyParser = TypedDependencyParserModel.pretrained()
  .setInputCols("dependency", "pos", "token")
  .setOutputCol("dependency_type")
```
The default model is `"dependency_typed_conllu"`, if no name is provided.
For available pretrained models please see the [Models Hub](https://nlp.johnsnowlabs.com/models).

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/3.SparkNLP_Pretrained_Models.ipynb)
and the [TypedDependencyModelTestSpec](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/annotators/parser/typdep/TypedDependencyModelTestSpec.scala).
{%- endcapture -%}

{%- capture model_input_anno -%}
TOKEN, POS, DEPENDENCY
{%- endcapture -%}

{%- capture model_output_anno -%}
LABELED_DEPENDENCY
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
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

posTagger = PerceptronModel.pretrained() \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("pos")

dependencyParser = DependencyParserModel.pretrained() \
    .setInputCols(["sentence", "pos", "token"]) \
    .setOutputCol("dependency")

typedDependencyParser = TypedDependencyParserModel.pretrained() \
    .setInputCols(["dependency", "pos", "token"]) \
    .setOutputCol("dependency_type")

pipeline = Pipeline().setStages([
    documentAssembler,
    sentence,
    tokenizer,
    posTagger,
    dependencyParser,
    typedDependencyParser
])

data = spark.createDataFrame([[
    "Unions representing workers at Turner Newall say they are 'disappointed' after talks with stricken parent " +
      "firm Federal Mogul."
]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.selectExpr("explode(arrays_zip(token.result, dependency.result, dependency_type.result)) as cols") \
    .selectExpr("cols['0'] as token", "cols['1'] as dependency", "cols['2'] as dependency_type") \
    .show(8, truncate = False)
+------------+------------+---------------+
|token       |dependency  |dependency_type|
+------------+------------+---------------+
|Unions      |ROOT        |root           |
|representing|workers     |amod           |
|workers     |Unions      |flat           |
|at          |Turner      |case           |
|Turner      |workers     |flat           |
|Newall      |say         |nsubj          |
|say         |Unions      |parataxis      |
|they        |disappointed|nsubj          |
+------------+------------+---------------+

{%- endcapture -%}

{%- capture model_scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel
import com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserModel
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

val typedDependencyParser = TypedDependencyParserModel.pretrained()
  .setInputCols("dependency", "pos", "token")
  .setOutputCol("dependency_type")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentence,
  tokenizer,
  posTagger,
  dependencyParser,
  typedDependencyParser
))

val data = Seq(
  "Unions representing workers at Turner Newall say they are 'disappointed' after talks with stricken parent " +
    "firm Federal Mogul."
).toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("explode(arrays_zip(token.result, dependency.result, dependency_type.result)) as cols")
  .selectExpr("cols['0'] as token", "cols['1'] as dependency", "cols['2'] as dependency_type")
  .show(8, truncate = false)
+------------+------------+---------------+
|token       |dependency  |dependency_type|
+------------+------------+---------------+
|Unions      |ROOT        |root           |
|representing|workers     |amod           |
|workers     |Unions      |flat           |
|at          |Turner      |case           |
|Turner      |workers     |flat           |
|Newall      |say         |nsubj          |
|say         |Unions      |parataxis      |
|they        |disappointed|nsubj          |
+------------+------------+---------------+

{%- endcapture -%}

{%- capture model_api_link -%}
[TypedDependencyParserModel](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/parser/typdep/TypedDependencyParserModel)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[TypedDependencyParserModel](/api/python/reference/autosummary/python/sparknlp/annotator/dependency/typed_dependency_parser/index.html#sparknlp.annotator.dependency.typed_dependency_parser.TypedDependencyParserModel)
{%- endcapture -%}

{%- capture model_source_link -%}
[TypedDependencyParserModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/parser/typdep/TypedDependencyParserModel.scala)
{%- endcapture -%}

{%- capture approach_description -%}
Labeled parser that finds a grammatical relation between two words in a sentence.
Its input is either a CoNLL2009 or ConllU dataset.

For instantiated/pretrained models, see TypedDependencyParserModel.

Dependency parsers provide information about word relationship. For example, dependency parsing can tell you what
the subjects and objects of a verb are, as well as which words are modifying (describing) the subject. This can help
you find precise answers to specific questions.

The parser requires the dependant tokens beforehand with e.g. [DependencyParser](/docs/en/annotators#dependencyparser).
The required training data can be set in two different ways (only one can be chosen for a particular model):
  - Dataset in the [CoNLL 2009 format](https://ufal.mff.cuni.cz/conll2009-st/trial-data.html) set with `setConll2009`
  - Dataset in the [CoNLL-U format](https://universaldependencies.org/format.html) set with `setConllU`

Apart from that, no additional training data is needed.

See [TypedDependencyParserApproachTestSpec](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/annotators/parser/typdep/TypedDependencyParserApproachTestSpec.scala) for further reference on this API.
{%- endcapture -%}

{%- capture approach_input_anno -%}
TOKEN, POS, DEPENDENCY
{%- endcapture -%}

{%- capture approach_output_anno -%}
LABELED_DEPENDENCY
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
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

posTagger = PerceptronModel.pretrained() \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("pos")

dependencyParser = DependencyParserModel.pretrained() \
    .setInputCols(["sentence", "pos", "token"]) \
    .setOutputCol("dependency")

typedDependencyParser = TypedDependencyParserApproach() \
    .setInputCols(["dependency", "pos", "token"]) \
    .setOutputCol("dependency_type") \
    .setConllU("src/test/resources/parser/labeled/train_small.conllu.txt") \
    .setNumberOfIterations(1)

pipeline = Pipeline().setStages([
    documentAssembler,
    sentence,
    tokenizer,
    posTagger,
    dependencyParser,
    typedDependencyParser
])

# Additional training data is not needed, the dependency parser relies on CoNLL-U only.
emptyDataSet = spark.createDataFrame([[""]]).toDF("text")
pipelineModel = pipeline.fit(emptyDataSet)

{%- endcapture -%}

{%- capture approach_scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel
import com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserApproach
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

val typedDependencyParser = new TypedDependencyParserApproach()
  .setInputCols("dependency", "pos", "token")
  .setOutputCol("dependency_type")
  .setConllU("src/test/resources/parser/labeled/train_small.conllu.txt")
  .setNumberOfIterations(1)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentence,
  tokenizer,
  posTagger,
  dependencyParser,
  typedDependencyParser
))

// Additional training data is not needed, the dependency parser relies on CoNLL-U only.
val emptyDataSet = Seq.empty[String].toDF("text")
val pipelineModel = pipeline.fit(emptyDataSet)

{%- endcapture -%}

{%- capture approach_api_link -%}
[TypedDependencyParserApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/parser/typdep/TypedDependencyParserApproach)
{%- endcapture -%}

{%- capture approach_python_api_link -%}
[TypedDependencyParserApproach](/api/python/reference/autosummary/python/sparknlp/annotator/dependency/typed_dependency_parser/index.html#sparknlp.annotator.dependency.typed_dependency_parser.TypedDependencyParserApproach)
{%- endcapture -%}

{%- capture approach_source_link -%}
[TypedDependencyParserApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/parser/typdep/TypedDependencyParserApproach.scala)
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
