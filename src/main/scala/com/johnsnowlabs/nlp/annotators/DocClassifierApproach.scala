package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{LABEL, DOCUMENT, SENTENCE_EMBEDDINGS}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{BooleanParam, Param}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{Dataset}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.param.shared.HasSeed
import org.slf4j.LoggerFactory

class DocClassifierApproach(override val uid: String)
    extends AnnotatorApproach[DocClassifierModel]
    with HasSeed
    with DataPreparation {

  def this() = this(Identifiable.randomUID("TRF"))

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT, SENTENCE_EMBEDDINGS)
  override val outputAnnotatorType: AnnotatorType = LABEL

  private val logger = LoggerFactory.getLogger("DocClassifier")

  override val description = "ML based Text Classifier Estimator"

  val labelCol = new Param[String](this, "labelCol", "column with the intent result of every row.")
  def setLabelCol(value: String): this.type = set(labelCol, value)
  def getLabelCol: String = $(labelCol)

  val featureCol = new Param[String](this, "featureCol", "column to output the sentence embeddings as SparkML vector.")
  def setFeatureCol(value: String): this.type = set(featureCol, value)
  def getFeatureCol: String = $(featureCol)

  val encodedLabelCol = new Param[String](this, "encodedLabelCol", "column to output the label ordinally encoded.")
  def setEncodedLabelCol(value: String): this.type = set(featureCol, value)
  def getEncodedLabelCol: String = $(featureCol)

  val multiLabel = new BooleanParam (this, "multiLabel", "is this a multilabel or single label classification problem")
  def setMultiLabel(value: Boolean): this.type = set(multiLabel, value)
  def getMultiLabel: Boolean = $(multiLabel)

  setDefault(
    inputCols -> Array(DOCUMENT, SENTENCE_EMBEDDINGS),
    labelCol -> LABEL,
    outputCol -> LABEL.concat("_output"),
    featureCol -> SENTENCE_EMBEDDINGS.concat("_vector"),
    encodedLabelCol -> LABEL.concat("_encoded"),
    multiLabel -> false
  )

  val sentenceCol = $(inputCols)(0)
  override val featuresAnnotationCol = $(inputCols)(1)
  override val featuresVectorCol: String = $(featureCol)
  override val labelRawCol: String = $(labelCol)
  override val labelEncodedCol: String = $(encodedLabelCol)


  lazy val sparkClassifier = new RandomForestClassifier()
    .setFeaturesCol($(featureCol))
    .setLabelCol($(encodedLabelCol))
    .setSeed($(seed))

  // TODO: accuracyMetrics, cv, etc.

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): DocClassifierModel = {

    val preparedDataset = prepareData(dataset)

    val Array(trainData, testData) = preparedDataset.randomSplit(Array(0.7, 0.3), $(seed))
    // TODO: use testData to return metrics

    val sparkClassificationModel = sparkClassifier.fit(trainData)

    new DocClassifierModel(sparkClassificationModel)
      .setInputCols($(inputCols))
      .setOutputCol($(outputCol))
      .setFeatureCol($(featureCol))
  }
}






