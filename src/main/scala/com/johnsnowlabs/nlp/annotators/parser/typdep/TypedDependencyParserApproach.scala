package com.johnsnowlabs.nlp.annotators.parser.typdep

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{DEPENDENCY, DOCUMENT, POS, TOKEN}
import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{BooleanParam, FloatParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset

class TypedDependencyParserApproach(override val uid: String) extends AnnotatorApproach[TypedDependencyParserModel]{

  override val description: String =
    "Typed Dependency Parser is a labeled parser that shows the relationship between words in a document"
  override val annotatorType:String = DEPENDENCY
  override val requiredAnnotatorTypes = Array(DOCUMENT, POS, TOKEN)

  def this() = this(Identifiable.randomUID("TYPED DEPENDENCY"))

  val numberOfPreTrainingIterations = new IntParam(this, "numberOfPreTrainingIterations",
    "Number of iterations used in a pre-training phase, converges to better accuracy")
  val numberOfTrainingIterations = new IntParam(this, "numberOfTrainingIterations",
    "Number of iterations in training, converges to better accuracy")
  val initTensorWithPreTrain: BooleanParam = new BooleanParam(this, "initTensorWithPretrain",
    "whether to include tensor with pre-training. Defaults to true.")
  val regularization: FloatParam = new FloatParam(this, "regularization",
    "Run the model with regularization. In math formulas is known as variable C")
  val gammaLabel: FloatParam = new FloatParam(this, "gammaLabel",
    "wWight of the traditional features in the scoring function ")
  val rankFirstOrderTensor = new IntParam(this, "rankFirstOrderTensor",
    "Rank of the first-order tensor. In math formulas is know as R")
  val rankSecondOrderTensor = new IntParam(this, "rankSecondOrderTensor",
    "Rank of the second-order tensor. In math formulas is know as R2")


  case class TrainingOptions(numberOfPreTrainingIterations: Int, numberOfTrainingIterations: Int,
                             initTensorWithPreTrain: Boolean, regularization: Float, gammaLabel: Float,
                             rankFirstOrderTensor: Int, rankSecondOrderTensor: Int)

  def loadPretrainedModel(): DependencyParserModel = {
    DependencyParserModel.read.load("./tmp/dp_model")
  }

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): TypedDependencyParserModel = {

    require(!dataset.rdd.isEmpty(), "Training file with CoNLL 2009 format is required")

    new TypedDependencyParserModel()
  }

}
