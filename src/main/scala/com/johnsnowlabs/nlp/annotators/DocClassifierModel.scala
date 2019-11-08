package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import com.johnsnowlabs.nlp.AnnotatorType.{LABEL, DOCUMENT, SENTENCE_EMBEDDINGS}
import org.apache.spark.ml.classification.{RandomForestClassificationModel}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.shared.HasSeed
import org.apache.spark.ml.param.{BooleanParam, Param}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{Dataset, functions => F}
import org.slf4j.LoggerFactory

import scala.collection.Map


class DocClassifierModel(override val uid: String, val sparkClassificationModel: RandomForestClassificationModel)
  extends AnnotatorModel[DocClassifierModel]
  with HasSeed
  with DataPreparation {


  def this(sparkClassificationModel: RandomForestClassificationModel) = this(Identifiable.randomUID("TRF"), sparkClassificationModel)

  private val logger = LoggerFactory.getLogger("DocClassifier")
    
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT, SENTENCE_EMBEDDINGS)
  override val outputAnnotatorType: AnnotatorType = LABEL

  val featureCol = new Param[String](this, "featureCol", "column to output the sentence embeddings as SparkML vector.")
  def setFeatureCol(value: String): this.type = set(featureCol, value)
  def getFeatureCol: String = $(featureCol)

  val multiLabel = new BooleanParam (this, "multiLabel", "is this a multilabel or single label classification problem")
  def setMultiLabel(value: Boolean): this.type = set(multiLabel, value)
  def getMultiLabel: Boolean = $(multiLabel)

  setDefault(
    inputCols -> Array(DOCUMENT, SENTENCE_EMBEDDINGS),
    outputCol -> LABEL.concat("_output"),
    featureCol -> SENTENCE_EMBEDDINGS.concat("_vector"),
    multiLabel -> false
  )

  val sentenceCol = $(inputCols)(0)
  override val featuresAnnotationCol = $(inputCols)(1)
  override val featuresVectorCol: String = $(featureCol)

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    annotations.filter(x => x.annotatorType == SENTENCE_EMBEDDINGS).map(x => {
      val result = sparkClassificationModel.predictRawPublic(Vectors.dense(x.embeddings.toArray.map(_.toDouble)))
      Annotation(outputAnnotatorType, x.begin, x.end, result.toString, Map())
    })

  }
}

trait DataPreparation {
  val featuresAnnotationCol: String
  val featuresVectorCol: String
  val labelRawCol: String = ""
  val labelEncodedCol: String = ""
  def prepareData(dataset: Dataset[_]): Dataset[_] = {

    val indexedDataset: Dataset[_] = if(labelRawCol != "" & !dataset.columns.contains(labelEncodedCol))
        new StringIndexer()
        .setInputCol(labelRawCol)
        .setOutputCol(labelEncodedCol)
        .fit(dataset).transform(dataset)
      else dataset

    val convertToVectorUDF = F.udf((matrix : Seq[Float]) => { Vectors.dense(matrix.toArray.map(_.toDouble)) })
    indexedDataset.withColumn(featuresVectorCol, convertToVectorUDF(F.expr(s"$featuresAnnotationCol.embeddings[0]")))

  }
}
