package com.johnsnowlabs.nlp.annotators.parser.typdep

import com.johnsnowlabs.nlp.AnnotatorType.{DEPENDENCY, LABELED_DEPENDENCY, POS, TOKEN}
import com.johnsnowlabs.nlp.annotators.common.{Conll2009Sentence, LabeledDependency}
import com.johnsnowlabs.nlp.annotators.parser.typdep.util.{DependencyLabel, Dictionary, DictionarySet}
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, AnnotatorType}
import gnu.trove.map.hash.TObjectIntHashMap
import org.apache.spark.ml.util.Identifiable

class
TypedDependencyParserModel(override val uid: String) extends AnnotatorModel[TypedDependencyParserModel] {

  def this() = this(Identifiable.randomUID("TYPED DEPENDENCY"))

  override val annotatorType: String = LABELED_DEPENDENCY
  override val requiredAnnotatorTypes = Array(TOKEN, POS, DEPENDENCY)

  val model: StructFeature[TrainParameters] = new StructFeature[TrainParameters](this, "TDP model")

  def setModel(targetModel: TrainParameters): this.type = set(model, targetModel)

  private lazy val options = $$(model).options
  private lazy val parameters = $$(model).parameters
  private lazy val dependencyPipe = $$(model).dependencyPipe

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

    val conll2009Document = LabeledDependency.unpack(annotations).toArray
    var conll2009Sentence = conll2009Document.filter(_.sentence == sentenceId)
    var labeledDependenciesDocument = Seq[Annotation]()

    while (conll2009Sentence.length > 0){

      val document = Array(conll2009Sentence, Array(Conll2009Sentence("end","sentence","ES","ES",-2, 0, 0, 0)))
      val documentData = transformToConll09Data(document)
      val dependencyLabels = typedDependencyParser.predictDependency(documentData)

      val labeledSentences = dependencyLabels.map{dependencyLabel =>
        getDependencyLabelValues(dependencyLabel)
      }

      val labeledDependenciesSentence = LabeledDependency.pack(labeledSentences)
      labeledDependenciesDocument = labeledDependenciesDocument ++ labeledDependenciesSentence
      sentenceId += 1
      conll2009Sentence = conll2009Document.filter(_.sentence == sentenceId)
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

  private def transformToConll09Data(document: Array[Array[Conll2009Sentence]]): Array[Array[Conll09Data]] = {
    document.map{sentence =>
      sentence.map{word =>
        new Conll09Data(word.dependency, word.lemma, word.pos, word.deprel, word.head, word.begin, word.end)
      }
    }
  }

  private def getDependencyLabelValues(dependencyLabel: DependencyLabel): Conll2009Sentence = {
    if (dependencyLabel != null){
      val label = getLabel(dependencyLabel.getLabel, dependencyLabel.getDependency)
      Conll2009Sentence(dependencyLabel.getDependency, "", "", label, dependencyLabel.getHead,
        0, dependencyLabel.getBegin, dependencyLabel.getEnd)
    } else {
      Conll2009Sentence("ROOT", "root", "", "ROOT", -1, 0, -1, 0)
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
