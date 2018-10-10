package com.johnsnowlabs.nlp.annotators.parser.dep

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.ExternalResource
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

class DependencyParserApproach(override val uid: String) extends AnnotatorApproach[DependencyParserModel] {
  override val description: String = "Dependency Parser Estimator used to train"

  def this() = this(Identifiable.randomUID(DEPENDENCY))

  val source = new ExternalResourceParam(this, "source", "source file for dependency model")

  def setSource(value: ExternalResource): this.type = set(source, value)

  override val annotatorType:String = DEPENDENCY

  override val requiredAnnotatorTypes = Array(DOCUMENT, POS, TOKEN)

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): DependencyParserModel = {
    dataset.show()
    new DependencyParserModel()
      .setSourcePath($(source))
  }
}

object DependencyParserApproach extends DefaultParamsReadable[DependencyParserApproach]