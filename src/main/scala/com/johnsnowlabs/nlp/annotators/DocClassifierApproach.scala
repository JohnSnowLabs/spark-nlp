package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, LABEL, SENTENCE_EMBEDDINGS}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.classification.SparkNLPRandomForestClassifier
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.shared.HasSeed
import org.apache.spark.sql.{Dataset, functions => F}
import org.slf4j.LoggerFactory

class DocClassifierApproach(override val uid: String)
    extends AnnotatorApproach[DocClassifierModel]
    with HasSeed {

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
  def setEncodedLabelCol(value: String): this.type = set(encodedLabelCol, value)
  def getEncodedLabelCol: String = $(encodedLabelCol)

  setDefault(
    inputCols -> Array(DOCUMENT, SENTENCE_EMBEDDINGS),
    labelCol -> LABEL,
    outputCol -> LABEL.concat("_output"),
    featureCol -> SENTENCE_EMBEDDINGS.concat("_vector"),
    encodedLabelCol -> LABEL.concat("_encoded")
  )

  val sentenceCol = $(inputCols)(0)
  val featuresAnnotationCol = $(inputCols)(1)
  val featuresVectorCol: String = $(featureCol)
  val labelRawCol: String = $(labelCol)
  val labelEncodedCol: String = $(encodedLabelCol)


  lazy val sparkClassifier = new SparkNLPRandomForestClassifier()
    .setFeaturesCol($(featureCol))
    .setLabelCol($(encodedLabelCol))
    .setSeed($(seed))

  // TODO: accuracyMetrics, cv, etc.

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): DocClassifierModel = {

    val (labels, preparedDataset) = prepareData(dataset)

    val Array(trainData: Dataset[_], testData: Dataset[_]) = preparedDataset.randomSplit(Array(0.7, 0.3), $(seed))
    // TODO: use testData to return metrics

    val sparkClassificationModel = sparkClassifier.fit(trainData)

    new DocClassifierModel(sparkClassificationModel)
      .setInputCols($(inputCols))
      .setOutputCol($(outputCol))
      .setFeatureCol($(featureCol))
      .setLabels(labels)
  }

  protected def prepareData(dataset: Dataset[_]): (Array[String], Dataset[_]) = {

    val (labels:Array[String], indexedDataset: Dataset[_]) = if(labelRawCol != "" & !dataset.columns.contains(labelEncodedCol))
    {
      val indexer = new StringIndexer()
        .setInputCol(labelRawCol)
        .setOutputCol(labelEncodedCol)
        .fit(dataset)

      (indexer.labels, indexer.transform(dataset))
    }
    else (Array(""), dataset)

    val convertToVectorUDF = F.udf((matrix : Seq[Float]) => { Vectors.dense(matrix.toArray.map(_.toDouble)) })

    (labels, indexedDataset.withColumn(featuresVectorCol, convertToVectorUDF(F.expr(s"$featuresAnnotationCol.embeddings[0]"))))
  }
}






