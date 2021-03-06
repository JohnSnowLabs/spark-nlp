package com.johnsnowlabs.nlp.annotators.parser.typdep

import com.johnsnowlabs.nlp.AnnotatorType.{DEPENDENCY, LABELED_DEPENDENCY, POS, TOKEN}
import com.johnsnowlabs.nlp.annotators.common.{ConllSentence, LabeledDependency}
import com.johnsnowlabs.nlp.annotators.parser.typdep.util.{DependencyLabel, Dictionary, DictionarySet}
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, HasPretrained, ParamsAndFeaturesReadable, HasSimpleAnnotate}
import gnu.trove.map.hash.TObjectIntHashMap
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable
/** Labeled parser that finds a grammatical relation between two words in a sentence. Its input is a CoNLL2009 or ConllU dataset.
  *
  * See [[https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/annotators/parser/typdep ]] for further reference on this API.
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
  * @groupdesc Parameters A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
  **/
class TypedDependencyParserModel(override val uid: String) extends AnnotatorModel[TypedDependencyParserModel] with HasSimpleAnnotate[TypedDependencyParserModel] {

  def this() = this(Identifiable.randomUID("TYPED_DEPENDENCY"))

  /** Outputs column type LABELED_DEPENDENCY
    *
    * @group anno
    **/
  override val outputAnnotatorType: String = LABELED_DEPENDENCY
  /** Input requires column types TOKEN, POS, DEPENDENCY
    *
    * @group anno
    **/
  override val inputAnnotatorTypes = Array(TOKEN, POS, DEPENDENCY)

  /** Training options */
  /** @group param */
  val trainOptions: StructFeature[Options] = new StructFeature[Options](this, "trainOptions")
  /** @group param */
  val trainParameters: StructFeature[Parameters] = new StructFeature[Parameters](this, "trainParameters")
  /** @group param */
  val trainDependencyPipe: StructFeature[DependencyPipe] = new StructFeature[DependencyPipe](this, "trainDependencyPipe")
  /** @group param */
  val conllFormat: Param[String] = new Param[String](this, "conllFormat", "CoNLL Format")

  /** Training options */
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

    while (conllSentence.length > 0){

      val document = Array(conllSentence, Array(ConllSentence("end","sentence","ES","ES","ES",-2, 0, 0, 0)))
      val documentData = transformToConllData(document)
      val dependencyLabels = typedDependencyParser.predictDependency(documentData, $(conllFormat))

      val labeledSentences = dependencyLabels.map{dependencyLabel =>
        getDependencyLabelValues(dependencyLabel)
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
    if (dictionary.getMapAsString != null){
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
    document.map{sentence =>
      sentence.map{word =>
        new ConllData(word.dependency, word.lemma, word.uPos, word.xPos, word.deprel, word.head, word.begin, word.end)
      }
    }
  }

  private def getDependencyLabelValues(dependencyLabel: DependencyLabel): ConllSentence = {
    if (dependencyLabel != null) {
      ConllSentence(dependencyLabel.getDependency, "", "", "", dependencyLabel.getLabel, dependencyLabel.getHead,
        0, dependencyLabel.getBegin, dependencyLabel.getEnd)
    } else {
      ConllSentence("ROOT", "root", "", "", "ROOT", -1, 0, -1, 0)
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

object TypedDependencyParserModel extends ReadablePretrainedTypedDependency
