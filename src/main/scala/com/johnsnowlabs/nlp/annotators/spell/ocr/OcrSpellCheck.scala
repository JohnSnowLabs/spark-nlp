package com.johnsnowlabs.nlp.annotators.spell.ocr

import com.johnsnowlabs.nlp.{AnnotatorApproach, AnnotatorType}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.Dataset

class OcrSpellCheckApproach(override val uid: String) extends AnnotatorApproach[OcrSpellCheckModel]{
  override val description: String = "Ocr specific Spell Checking"

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): OcrSpellCheckModel = ???

  override val annotatorType: AnnotatorType = AnnotatorType.TOKEN
  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val requiredAnnotatorTypes: Array[String] = Array(AnnotatorType.TOKEN)

  /* this is a list of functions that return the distance of a string to a particular regex */
  private var tokenClasses = List[(String) => Tuple2[String, Float]]()

}
