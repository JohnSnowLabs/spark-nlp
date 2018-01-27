package com.johnsnowlabs.nlp.annotators.parser.dep

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType._
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

class DependencyParserApproach(override val uid: String) extends AnnotatorApproach[DependencyParserModel] {
  override val description: String = "Dependency Parser Estimator used to train"

  def this() = this(Identifiable.randomUID(DEPENDENCY))

  val sourcePath = new Param[String](this, "sourcePath", "source file for dependency model")

  def setSourcePath(value: String): this.type = set(sourcePath, value)

  override val annotatorType = DEPENDENCY

  override val requiredAnnotatorTypes = Array(DOCUMENT, POS, TOKEN)

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): DependencyParserModel = {
    new DependencyParserModel()
      .setSourcePath($(sourcePath))
  }
}

object DependencyParserApproach extends DefaultParamsReadable[DependencyParserApproach]