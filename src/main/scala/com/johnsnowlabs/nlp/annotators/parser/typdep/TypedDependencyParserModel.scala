package com.johnsnowlabs.nlp.annotators.parser.typdep

import com.johnsnowlabs.nlp.AnnotatorType.{DEPENDENCY, LABELED_DEPENDENCY}
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import org.apache.spark.ml.util.Identifiable

class TypedDependencyParserModel(override val uid: String) extends AnnotatorModel[TypedDependencyParserModel]{

  def this() = this(Identifiable.randomUID("DE-IDENTIFICATION"))

  override val annotatorType:String = LABELED_DEPENDENCY
  override val requiredAnnotatorTypes = Array(DEPENDENCY)

  val model: StructFeature[TrainParameters] = new StructFeature[TrainParameters](this, "TDP model")

  def setModel(targetModel: TrainParameters): this.type = set(model, targetModel)

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val pruebas = $$(model).options
    Seq(Annotation(annotatorType, 0, 1, "annotate", Map("sentence" -> "protected")))
  }

}
