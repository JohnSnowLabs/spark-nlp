package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, LABEL, SENTENCE_EMBEDDINGS}
import org.apache.spark.ml.classification.SparkNLPRandomForestClassificationModel
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.shared.HasSeed
import org.apache.spark.ml.param.{Param, StringArrayParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, Dataset, functions => F}
import org.slf4j.LoggerFactory

import scala.collection.Map


class DocClassifierModel(override val uid: String, val sparkClassificationModel: SparkNLPRandomForestClassificationModel)
  extends AnnotatorModel[DocClassifierModel]
  with HasSeed {


  def this(sparkClassificationModel: SparkNLPRandomForestClassificationModel) = this(Identifiable.randomUID("TRF"), sparkClassificationModel)

  private val logger = LoggerFactory.getLogger("DocClassifier")
    
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT, SENTENCE_EMBEDDINGS)
  override val outputAnnotatorType: AnnotatorType = LABEL

  val featureCol = new Param[String](this, "featureCol", "column to output the sentence embeddings as SparkML vector.")
  def setFeatureCol(value: String): this.type = set(featureCol, value)
  def getFeatureCol: String = $(featureCol)

  val decodedOutputCol = new Param[String](this, "encodedOutputCol", "column to output the label in the original form.")
  def setDecodedOutputCol(value: String): this.type = set(decodedOutputCol, value)
  def getDecodedOutputCol: String = $(decodedOutputCol)

  val labels = new Param[Array[String]](this, "labels", "column to output the label in the original form.")
  def setLabels(value: Array[String]): this.type = set(labels, value)
  def getLabels: Array[String] = $(labels)

  setDefault(
    inputCols -> Array(DOCUMENT, SENTENCE_EMBEDDINGS),
    outputCol -> LABEL.concat("_output"),
    featureCol -> SENTENCE_EMBEDDINGS.concat("_vector")
  )

  val sentenceCol = $(inputCols)(0)
  val featuresAnnotationCol = $(inputCols)(1)
  val featuresVectorCol: String = $(featureCol)

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    require(labels.isValid(Array()), "the parameter labels should be set to be able to annotate")
    val labelsArray =  get(labels).getOrElse(Array("NoLabels"))
    annotations.filter(x => x.annotatorType == SENTENCE_EMBEDDINGS).map(x => {
      val prediction = sparkClassificationModel.predictRawPublic(Vectors.dense(x.embeddings.toArray.map(_.toDouble)))
      val idx = Math.min(labelsArray.length-1, prediction._1).toInt
      val currentLabel = labelsArray(idx)
      Annotation(outputAnnotatorType, x.begin, x.end, currentLabel, prediction._2)
    })
  }

  def prepareData(dataset: Dataset[_]): Dataset[_] = {

    val convertToVectorUDF = F.udf((matrix : Seq[Float]) => { Vectors.dense(matrix.toArray.map(_.toDouble)) })

    dataset.withColumn(featuresVectorCol, convertToVectorUDF(F.expr(s"$featuresAnnotationCol.embeddings[0]")))
  }
}
