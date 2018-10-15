package com.johnsnowlabs.nlp.annotators.parser.typdep

import com.johnsnowlabs.nlp.AnnotatorType.{DEPENDENCY, DOCUMENT, POS, TOKEN}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import org.apache.spark.ml.util.Identifiable

class TypedDependencyParserModel(override val uid: String) extends AnnotatorModel[TypedDependencyParserModel]{

  def this() = this(Identifiable.randomUID("DE-IDENTIFICATION"))

  override val annotatorType:String = DEPENDENCY
  override val requiredAnnotatorTypes = Array(DOCUMENT, POS, TOKEN)

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    Seq(Annotation(annotatorType, 0, 1, "annotate", Map("sentence" -> "protected")))
  }

}
