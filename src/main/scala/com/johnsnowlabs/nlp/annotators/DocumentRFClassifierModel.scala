package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, LABEL, SENTENCE_EMBEDDINGS}
import com.johnsnowlabs.nlp.serialization.StructFeature
import org.apache.spark.ml.classification.SparkNLPRandomForestClassificationModel
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.shared.HasSeed
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{Dataset, functions => F}
import org.slf4j.LoggerFactory


class DocumentRFClassifierModel(override val uid: String)
  extends AnnotatorModel[DocumentRFClassifierModel]
  with HasSeed {


  def this() = this(Identifiable.randomUID("TRF"))

  private val logger = LoggerFactory.getLogger("DocClassifier")
    
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(SENTENCE_EMBEDDINGS)
  override val outputAnnotatorType: AnnotatorType = LABEL

  val classificationModel = new StructFeature[SparkNLPRandomForestClassificationModel](this, "wrappedModel")
  def setClassificationModel(value: SparkNLPRandomForestClassificationModel): this.type = set(classificationModel, value)
  def getClassificationModel: SparkNLPRandomForestClassificationModel = $$(classificationModel)

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
    inputCols -> Array(SENTENCE_EMBEDDINGS),
    outputCol -> LABEL.concat("_output"),
    featureCol -> SENTENCE_EMBEDDINGS.concat("_vector")
  )

  val featuresAnnotationCol = $(inputCols)(0)
  val featuresVectorCol: String = $(featureCol)

  override def beforeAnnotate(dataset: Dataset[_]): Dataset[_] = {
    prepareData(dataset)
  }

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    require(labels.isValid(Array()), "the parameter labels should be set to be able to annotate")
    val labelsArray =  get(labels).getOrElse(Array("NoLabels"))
    annotations.filter(x => x.annotatorType == SENTENCE_EMBEDDINGS).map(x => {
      val prediction = $$(classificationModel).predictRawPublic(Vectors.dense(x.embeddings.toArray.map(_.toDouble)))
      val idx = Math.min(labelsArray.length-1, prediction._1).toInt
      val currentLabel = labelsArray(idx)
      Annotation(outputAnnotatorType, x.begin, x.end, currentLabel, prediction._2)
    })
  }

  protected def prepareData(dataset: Dataset[_]): Dataset[_] = {
    vectorizeFeatures(dataset)
  }

  protected def vectorizeFeatures(dataset: Dataset[_]): Dataset[_] = {
    val convertToVectorUDF = F.udf((matrix : Seq[Float]) => { Vectors.dense(matrix.toArray.map(_.toDouble)) })
    dataset.withColumn(featuresVectorCol, convertToVectorUDF(F.expr(s"$featuresAnnotationCol.embeddings[0]")))
  }

}
