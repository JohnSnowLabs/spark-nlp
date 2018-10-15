package com.johnsnowlabs.nlp.annotators.parser.typdep

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{DEPENDENCY, DOCUMENT, POS, TOKEN}
import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset

class TypedDependencyParserApproach(override val uid: String) extends AnnotatorApproach[TypedDependencyParserModel]{

  override val description: String = "Typed Dependency Parser is a parser that labels the relationship between word in a document"
  override val annotatorType:String = DEPENDENCY
  override val requiredAnnotatorTypes = Array(DOCUMENT, POS, TOKEN)

  def this() = this(Identifiable.randomUID("TYPED DEPENDENCY"))

  def loadPretrainedModel(): DependencyParserModel = {
    DependencyParserModel.read.load("./tmp/dp_model")
  }

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): TypedDependencyParserModel = {

    require(!dataset.rdd.isEmpty(), "Training file with CoNLL 2009 format is required")

    val dependencyParserModel = this.loadPretrainedModel()

    new TypedDependencyParserModel()
  }

}
