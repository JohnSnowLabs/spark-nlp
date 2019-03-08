package com.johnsnowlabs.nlp.annotators.parser.typdep

import com.johnsnowlabs.nlp.AnnotatorType.{DEPENDENCY, LABELED_DEPENDENCY, POS, TOKEN}
import com.johnsnowlabs.nlp.annotators.common.{ConllSentence, LabeledDependency}
import com.johnsnowlabs.nlp.annotators.parser.typdep.util.{DependencyLabel, Dictionary, DictionarySet}
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, ParamsAndFeaturesReadable}
import gnu.trove.map.hash.TObjectIntHashMap
import org.apache.spark.ml.util.Identifiable

class
TypedDependencyParserModel(override val uid: String) extends AnnotatorModel[TypedDependencyParserModel] {

  def this() = this(Identifiable.randomUID("TYPED DEPENDENCY"))

  override val annotatorType: String = LABELED_DEPENDENCY
  override val requiredAnnotatorTypes = Array(TOKEN, POS, DEPENDENCY)

  val trainOptions: StructFeature[Options] = new StructFeature[Options](this, "TDP options")
  val trainParameters: StructFeature[Parameters] = new StructFeature[Parameters](this, "TDP parameters")
  val trainDependencyPipe: StructFeature[DependencyPipe] = new StructFeature[DependencyPipe](this, "TDP dependency pipe")

  def setOptions(targetOptions: Options): this.type = set(trainOptions, targetOptions)
  def setDependencyPipe(targetDependencyPipe: DependencyPipe): this.type = set(trainDependencyPipe, targetDependencyPipe)

  private lazy val options = $$(trainOptions)
  private lazy val dependencyPipe = $$(trainDependencyPipe)
  private lazy val parameters = new Parameters(dependencyPipe, options)

  var sentenceId = 1

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

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

      val document = Array(conllSentence, Array(ConllSentence("end","sentence","ES","ES",-2, 0, 0, 0)))
      val documentData = transformToConllData(document)
      val dependencyLabels = typedDependencyParser.predictDependency(documentData)

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
        new ConllData(word.dependency, word.lemma, word.pos, word.deprel, word.head, word.begin, word.end)
      }
    }
  }

  private def getDependencyLabelValues(dependencyLabel: DependencyLabel): ConllSentence = {
    if (dependencyLabel != null){
      val label = getLabel(dependencyLabel.getLabel, dependencyLabel.getDependency)
      ConllSentence(dependencyLabel.getDependency, "", "", label, dependencyLabel.getHead,
        0, dependencyLabel.getBegin, dependencyLabel.getEnd)
    } else {
      ConllSentence("ROOT", "root", "", "ROOT", -1, 0, -1, 0)
    }
  }

  def getLabel(label: String, dependency: String): String = {
    val head = getHead(dependency)
    if (label == "<no-type>" && head == "ROOT"){
      "ROOT"
    } else {
      label
    }
  }

  def getHead(dependency: String): String = {
    val beginIndex = dependency.indexOf("(") + 1
    val endIndex = dependency.indexOf(",")
    val head = dependency.substring(beginIndex, endIndex)
    head
  }
}

trait PretrainedTypedDependencyParserModel {
  def pretrained(name: String = "tdp_fast", language: Option[String] = Some("en"),
                 remoteLoc: String = ResourceDownloader.publicLoc): TypedDependencyParserModel =
    ResourceDownloader.downloadModel(TypedDependencyParserModel, name, language, remoteLoc)
}

object TypedDependencyParserModel extends ParamsAndFeaturesReadable[TypedDependencyParserModel] with PretrainedTypedDependencyParserModel
