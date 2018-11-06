package com.johnsnowlabs.nlp.annotators.parser.typdep

import com.johnsnowlabs.nlp.AnnotatorType.{DEPENDENCY, LABELED_DEPENDENCY, POS, TOKEN}
import com.johnsnowlabs.nlp.annotators.common.{Conll2009Sentence, LabeledDependency}
import com.johnsnowlabs.nlp.annotators.parser.typdep.util.{DependencyLabel, DictionarySet}
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import gnu.trove.map.hash.TObjectIntHashMap
import org.apache.spark.ml.util.Identifiable

class TypedDependencyParserModel(override val uid: String) extends AnnotatorModel[TypedDependencyParserModel] {

  def this() = this(Identifiable.randomUID("TYPED DEPENDENCY"))

  override val annotatorType: String = LABELED_DEPENDENCY
  override val requiredAnnotatorTypes = Array(TOKEN, POS, DEPENDENCY)

  val model: StructFeature[TrainParameters] = new StructFeature[TrainParameters](this, "TDP model")

  def setModel(targetModel: TrainParameters): this.type = set(model, targetModel)

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val options = $$(model).options
    val parameters = $$(model).parameters
    val dependencyPipe = $$(model).dependencyPipe

    val dictionariesValues = dependencyPipe.getDictionariesSet.getDictionaries.map { dictionary =>
      val predictionParameters = getPredictionParametersInstance
      val troveMap: TObjectIntHashMap[_] = predictionParameters.transformToTroveMap(dictionary.getMapAsString)
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

    val document = Array(conll2009Document, Array(Conll2009Sentence("end","sentence","ES","ES",-2, 0, 0, 0)))
    val documentData = transformToConll09Data(document)

    val dependencyLabels = typedDependencyParser.predictDependency(documentData)

    val labeledSentences = dependencyLabels.map{dependencyLabel =>
        getDependencyLabelValues(dependencyLabel)
    }

    val labeledDependencies = LabeledDependency.pack(labeledSentences)
    println(labeledDependencies.size)
    labeledDependencies
  }

  private def getPredictionParametersInstance: PredictionParameters = {
    new PredictionParameters()
  }

  private def getDictionarySetInstance: DictionarySet = {
    new DictionarySet()
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

  private def getTypedDependencyParserInstance: TypedDependencyParser = {
    new TypedDependencyParser
  }

  private def transformToConll09Data(document: Array[Array[Conll2009Sentence]]): Array[Array[Conll09Data]] = {
    document.map{sentence =>
      sentence.map{word =>
        new Conll09Data(word.form, word.lemma, word.pos, word.deprel, word.head, word.begin, word.end)
      }
    }
  }

  private def getDependencyLabelValues(dependencyLabel: DependencyLabel): Conll2009Sentence = {
    //TODO: Verify if sentence id is required to send to dependency label as well
    if (dependencyLabel != null){
      Conll2009Sentence(dependencyLabel.getToken, "", "", dependencyLabel.getLabel, dependencyLabel.getHead,
        0, dependencyLabel.getBegin, dependencyLabel.getEnd)
    } else {
      Conll2009Sentence("ROOT", "root", "", "ROOT", -1, 0, -1, 0)
    }

  }

}
