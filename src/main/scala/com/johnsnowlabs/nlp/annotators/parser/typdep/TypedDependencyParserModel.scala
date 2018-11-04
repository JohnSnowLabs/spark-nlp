package com.johnsnowlabs.nlp.annotators.parser.typdep

import com.johnsnowlabs.nlp.AnnotatorType.{DEPENDENCY, LABELED_DEPENDENCY, POS}
import com.johnsnowlabs.nlp.annotators.common.LabeledDependency
import com.johnsnowlabs.nlp.annotators.parser.typdep.util.DictionarySet
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import gnu.trove.map.hash.TObjectIntHashMap
import org.apache.spark.ml.util.Identifiable

class TypedDependencyParserModel(override val uid: String) extends AnnotatorModel[TypedDependencyParserModel] {

  def this() = this(Identifiable.randomUID("TYPED DEPENDENCY"))

  override val annotatorType: String = LABELED_DEPENDENCY
  override val requiredAnnotatorTypes = Array(POS, DEPENDENCY)

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

    //TODO Make a class similar to Tagged to unpack values following a structure similar to example.pred
    val conll = LabeledDependency.unpack(annotations)

    Seq(Annotation(annotatorType, 0, 1, "annotate", Map("sentence" -> "protected")))
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

}
