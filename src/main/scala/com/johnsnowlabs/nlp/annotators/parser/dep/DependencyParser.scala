package com.johnsnowlabs.nlp.annotators.parser.dep

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType._
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

class DependencyParser(override val uid: String) extends AnnotatorApproach[DependencyParserModel] {
  override val description: String = "Dependency Parser Estimator used to train"

  def this() = this(Identifiable.randomUID(DEPENDENCY))

  override val annotatorType = DEPENDENCY

  override val requiredAnnotatorTypes = Array(DOCUMENT, POS, TOKEN)

  override def train(dataset: Dataset[_]): DependencyParserModel = {
    new DependencyParserModel()
  }
}

object DependencyParser extends DefaultParamsReadable[DependencyParser]