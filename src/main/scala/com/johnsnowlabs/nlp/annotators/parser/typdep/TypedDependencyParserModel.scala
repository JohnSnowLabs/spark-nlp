package com.johnsnowlabs.nlp.annotators.parser.typdep

import com.johnsnowlabs.nlp.AnnotatorType.{DEPENDENCY, LABELED_DEPENDENCY, POS, TOKEN}
import com.johnsnowlabs.nlp.annotators.common.{ConllSentence, LabeledDependency}
import com.johnsnowlabs.nlp.annotators.parser.typdep.util.{DependencyLabel, Dictionary, DictionarySet}
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, HasPretrained, ParamsAndFeaturesReadable, HasSimpleAnnotate}
import gnu.trove.map.hash.TObjectIntHashMap
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable

/** Labeled parser that finds a grammatical relation between two words in a sentence.
 * Its input is either a CoNLL2009 or ConllU dataset.
 *
 * Dependency parsers provide information about word relationship. For example, dependency parsing can tell you what
 * the subjects and objects of a verb are, as well as which words are modifying (describing) the subject. This can help
 * you find precise answers to specific questions.
 *
 * The parser requires the dependant tokens beforehand with e.g. [[com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel DependencyParser]].
 *
 * Pretrained models can be loaded with `pretrained` of the companion object:
 * {{{
 * val typedDependencyParser = TypedDependencyParserModel.pretrained()
 *   .setInputCols("dependency", "pos", "token")
 *   .setOutputCol("dependency_type")
 * }}}
 * The default model is `"dependency_typed_conllu"`, if no name is provided.
 * For available pretrained models please see the [[https://nlp.johnsnowlabs.com/models Models Hub]].
 *
 * For extended examples of usage, see the [[https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/databricks_notebooks/3.SparkNLP_Pretrained_Models_v3.0.ipynb Spark NLP Workshop]]
 * and the [[https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/annotators/parser/typdep/TypedDependencyModelTestSpec.scala TypedDependencyModelTestSpec]].
 *
 * ==Example==
 * {{{
 * import spark.implicits._
 * import com.johnsnowlabs.nlp.base.DocumentAssembler
 * import com.johnsnowlabs.nlp.annotators.Tokenizer
 * import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
 * import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
 * import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel
 * import com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserModel
 * import org.apache.spark.ml.Pipeline
 *
 * val documentAssembler = new DocumentAssembler()
 *   .setInputCol("text")
 *   .setOutputCol("document")
 *
 * val sentence = new SentenceDetector()
 *   .setInputCols("document")
 *   .setOutputCol("sentence")
 *
 * val tokenizer = new Tokenizer()
 *   .setInputCols("sentence")
 *   .setOutputCol("token")
 *
 * val posTagger = PerceptronModel.pretrained()
 *   .setInputCols("sentence", "token")
 *   .setOutputCol("pos")
 *
 * val dependencyParser = DependencyParserModel.pretrained()
 *   .setInputCols("sentence", "pos", "token")
 *   .setOutputCol("dependency")
 *
 * val typedDependencyParser = TypedDependencyParserModel.pretrained()
 *   .setInputCols("dependency", "pos", "token")
 *   .setOutputCol("dependency_type")
 *
 * val pipeline = new Pipeline().setStages(Array(
 *   documentAssembler,
 *   sentence,
 *   tokenizer,
 *   posTagger,
 *   dependencyParser,
 *   typedDependencyParser
 * ))
 *
 * val data = Seq(
 *   "Unions representing workers at Turner Newall say they are 'disappointed' after talks with stricken parent " +
 *     "firm Federal Mogul."
 * ).toDF("text")
 * val result = pipeline.fit(data).transform(data)
 *
 * result.selectExpr("explode(arrays_zip(token.result, dependency.result, dependency_type.result)) as cols")
 *   .selectExpr("cols['0'] as token", "cols['1'] as dependency", "cols['2'] as dependency_type")
 *   .show(8, truncate = false)
 * +------------+------------+---------------+
 * |token       |dependency  |dependency_type|
 * +------------+------------+---------------+
 * |Unions      |ROOT        |root           |
 * |representing|workers     |amod           |
 * |workers     |Unions      |flat           |
 * |at          |Turner      |case           |
 * |Turner      |workers     |flat           |
 * |Newall      |say         |nsubj          |
 * |say         |Unions      |parataxis      |
 * |they        |disappointed|nsubj          |
 * +------------+------------+---------------+
 * }}}
 *
 * @groupname anno Annotator types
 * @groupdesc anno Required input and expected output annotator types
 * @groupname Ungrouped Members
 * @groupname param Parameters
 * @groupname setParam Parameter setters
 * @groupname getParam Parameter getters
 * @groupname Ungrouped Members
 * @groupprio param  1
 * @groupprio anno  2
 * @groupprio Ungrouped 3
 * @groupprio setParam  4
 * @groupprio getParam  5
 * @groupdesc param A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
 * */
class TypedDependencyParserModel(override val uid: String) extends AnnotatorModel[TypedDependencyParserModel] with HasSimpleAnnotate[TypedDependencyParserModel] {

  def this() = this(Identifiable.randomUID("TYPED_DEPENDENCY"))

  /** Outputs column type LABELED_DEPENDENCY
   *
   * @group anno
   * */
  override val outputAnnotatorType: String = LABELED_DEPENDENCY
  /** Input requires column types TOKEN, POS, DEPENDENCY
   *
   * @group anno
   * */
  override val inputAnnotatorTypes = Array(TOKEN, POS, DEPENDENCY)

  /** Training options */
  /** Options during training
   *
   * @group param
   * */
  val trainOptions: StructFeature[Options] = new StructFeature[Options](this, "trainOptions")
  /** Parameters during training
   *
   * @group param
   * */
  val trainParameters: StructFeature[Parameters] = new StructFeature[Parameters](this, "trainParameters")
  /** Dependency pipeline during training
   *
   * @group param
   * */
  val trainDependencyPipe: StructFeature[DependencyPipe] = new StructFeature[DependencyPipe](this, "trainDependencyPipe")
  /** CoNLL training format of this model
   *
   * @group param
   * */
  val conllFormat: Param[String] = new Param[String](this, "conllFormat", "CoNLL Format")

  /** @group setParam */
  def setOptions(targetOptions: Options): this.type = set(trainOptions, targetOptions)

  /** @group setParam */
  def setDependencyPipe(targetDependencyPipe: DependencyPipe): this.type = set(trainDependencyPipe, targetDependencyPipe)

  /** @group setParam */
  def setConllFormat(value: String): this.type = set(conllFormat, value)

  /** @group param */
  private lazy val options = $$(trainOptions)
  /** @group param */
  private lazy val dependencyPipe = $$(trainDependencyPipe)
  /** @group param */
  private lazy val parameters = new Parameters(dependencyPipe, options)

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    var sentenceId = 0
    val dictionariesValues = dependencyPipe.getDictionariesSet.getDictionaries.map { dictionary =>
      val predictionParameters = getPredictionParametersInstance
      val troveMap = getTroveMap(predictionParameters, dictionary)
      val numEntries = dictionary.getNumEntries
      val growthStopped = dictionary.isGrowthStopped
      (troveMap, numEntries, growthStopped)
    }

    val dictionarySet = deserializeDictionaries(dictionariesValues.toList)

    dependencyPipe.setDictionariesSet(dictionarySet)

    val typedDependencyParser = getTypedDependencyParserInstance
    typedDependencyParser.setOptions(options)
    typedDependencyParser.setParameters(parameters)
    typedDependencyParser.setDependencyPipe(dependencyPipe)
    typedDependencyParser.getDependencyPipe.closeAlphabets()

    val conllDocument = LabeledDependency.unpack(annotations).toArray
    var conllSentence = conllDocument.filter(_.sentence == sentenceId)
    var labeledDependenciesDocument = Seq[Annotation]()

    while (conllSentence.length > 0) {

      val document = Array(conllSentence, Array(ConllSentence("end", "sentence", "ES", "ES", "ES", -2, 0, 0, 0)))
      val documentData = transformToConllData(document)
      val dependencyLabels = typedDependencyParser.predictDependency(documentData, $(conllFormat))

      val labeledSentences = dependencyLabels.map{dependencyLabel =>
        getDependencyLabelValues(dependencyLabel, sentenceId)
      }

      val labeledDependenciesSentence = LabeledDependency.pack(labeledSentences)
      labeledDependenciesDocument = labeledDependenciesDocument ++ labeledDependenciesSentence
      sentenceId += 1
      conllSentence = conllDocument.filter(_.sentence == sentenceId)
    }

    labeledDependenciesDocument
  }


  private def getPredictionParametersInstance: PredictionParameters = {
    new PredictionParameters()
  }

  private def getTroveMap(predictionParameters: PredictionParameters, dictionary: Dictionary): TObjectIntHashMap[_] = {
    if (dictionary.getMapAsString != null) {
      predictionParameters.transformToTroveMap(dictionary.getMapAsString)
    } else {
      dictionary.getMap
    }
  }

  private def deserializeDictionaries(dictionariesValues: List[(TObjectIntHashMap[_], Int, Boolean)]): DictionarySet = {

    val dictionarySet = getDictionarySetInstance

    dictionariesValues.zipWithIndex.foreach { case (dictionaryValue, index) =>
      val dictionaries = dictionarySet.getDictionaries
      dictionaries(index).setMap(dictionaryValue._1)
      dictionaries(index).setNumEntries(dictionaryValue._2)
      dictionaries(index).setGrowthStopped(dictionaryValue._3)
    }

    dictionarySet
  }

  private def getDictionarySetInstance: DictionarySet = {
    new DictionarySet()
  }

  private def getTypedDependencyParserInstance: TypedDependencyParser = {
    new TypedDependencyParser
  }

  private def transformToConllData(document: Array[Array[ConllSentence]]): Array[Array[ConllData]] = {
    document.map { sentence =>
      sentence.map { word =>
        new ConllData(word.dependency, word.lemma, word.uPos, word.xPos, word.deprel, word.head, word.begin, word.end)
      }
    }
  }

  private def getDependencyLabelValues(dependencyLabel: DependencyLabel, sentenceId: Int): ConllSentence = {
    if (dependencyLabel != null) {
      ConllSentence(dependencyLabel.getDependency, "", "", "", dependencyLabel.getLabel, dependencyLabel.getHead,
        sentenceId, dependencyLabel.getBegin, dependencyLabel.getEnd)
    } else {
      ConllSentence("ROOT", "root", "", "", "ROOT", -1, sentenceId, -1, 0)
    }
  }

}

trait ReadablePretrainedTypedDependency extends ParamsAndFeaturesReadable[TypedDependencyParserModel] with HasPretrained[TypedDependencyParserModel] {
  override val defaultModelName = Some("dependency_typed_conllu")

  /** Java compliant-overrides */
  override def pretrained(): TypedDependencyParserModel = super.pretrained()

  override def pretrained(name: String): TypedDependencyParserModel = super.pretrained(name)

  override def pretrained(name: String, lang: String): TypedDependencyParserModel = super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): TypedDependencyParserModel = super.pretrained(name, lang, remoteLoc)
}

/**
 * This is the companion object of [[TypedDependencyParserModel]]. Please refer to that class for the documentation.
 */
object TypedDependencyParserModel extends ReadablePretrainedTypedDependency
