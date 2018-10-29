package com.johnsnowlabs.nlp.annotators.parser.typdep

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{DEPENDENCY, LABELED_DEPENDENCY}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{BooleanParam, FloatParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset

class TypedDependencyParserApproach(override val uid: String) extends AnnotatorApproach[TypedDependencyParserModel]{

  override val description: String =
    "Typed Dependency Parser is a labeled parser that shows the relationship between words in a document"
  override val annotatorType:String = LABELED_DEPENDENCY
  override val requiredAnnotatorTypes = Array(DEPENDENCY)

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

  val conll2009FilePath = new ExternalResourceParam(this, "conll2009FilePath",
      "Path to file with CoNLL 2009 format")

  def setConll2009FilePath(path: String, readAs: ReadAs.Format = ReadAs.LINE_BY_LINE,
                           options: Map[String, String] = Map.empty[String, String]): this.type = {
    set(conll2009FilePath, ExternalResource(path, readAs, options))
  }

  setDefault(conll2009FilePath, ExternalResource("", ReadAs.LINE_BY_LINE,  Map.empty[String, String]))

  private lazy val trainFile = {
    ResourceHelper.validFile($(conll2009FilePath).path)
    $(conll2009FilePath).path
  }

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): TypedDependencyParserModel = {

    //require(!dataset.rdd.isEmpty(), "Training file with CoNLL 2009 format is required")

    require(!trainFile.equals(""), "Training file with CoNLL 2009 format is required")

    val options = getOptionsInstance
    val typedDependencyParser = getTypedDependencyParserInstance
    typedDependencyParser.setOptions(options)

    val dependencyPipe = getDependencyPipeInstance(options)
    typedDependencyParser.setPipe(dependencyPipe)
    dependencyPipe.createAlphabets(trainFile)

    val trainDependencies = getTrainDependenciesInstance(trainFile, dependencyPipe, typedDependencyParser, options)
    trainDependencies.startTraining()
    val trainParameters = TrainParameters(options)
    println("Before setting")
    new TypedDependencyParserModel()
      .setModel(trainParameters)
  }

  private def getOptionsInstance: Options = {
    new Options()
  }

  private def getTypedDependencyParserInstance: TypedDependencyParser = {
    new TypedDependencyParser
  }

  private def getDependencyPipeInstance(options: Options): DependencyPipe = {
    new DependencyPipe(options)
  }

  private def getTrainDependenciesInstance(trainFile: String, dependencyPipe: DependencyPipe,
                                           typedDependencyParser: TypedDependencyParser,
                                           options: Options): TrainDependencies = {
    new TrainDependencies(trainFile, dependencyPipe, typedDependencyParser, options)
  }

}
