package com.johnsnowlabs.nlp.annotators.assertion.dl

import com.johnsnowlabs.ml.tensorflow.{AssertionDatasetEncoder, DatasetEncoderParams, TensorflowAssertion, TensorflowWrapper}
import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.annotators.datasets.AssertionAnnotationWithLabel
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.embeddings.ApproachWithWordEmbeddings
import org.apache.commons.io.IOUtils
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{FloatParam, IntParam, Param}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.tensorflow.{Graph, Session}

import scala.collection.mutable

/**
  * Created by jose on 14/03/18.
  */
class AssertionDLApproach(override val uid: String)
  extends ApproachWithWordEmbeddings[AssertionDLApproach, AssertionDLModel]{

  override val requiredAnnotatorTypes = Array(DOCUMENT)
  override val description: String = "Deep Learning based Assertion Status"

  // example of possible values, 'Negated', 'Affirmed', 'Historical'
  val labelCol = new Param[String](this, "label", "Column with one label per document")

  val startCol = new Param[String](this, "startCol", "Column with token number for first target token")
  val endCol = new Param[String](this, "endCol", "Column with token number for last target token")
  val nerCol = new Param[String](this, "nerCol", "Column of NER Annotations to use instead of start and end columns")


  val batchSize = new IntParam(this, "batchSize", "Size for each batch in the optimization process")
  val epochs = new IntParam(this, "epochs", "Number of epochs for the optimization process")
  val learningRate = new FloatParam(this, "learningRate", "Learning rate for the optimization process")
  val dropout = new FloatParam(this, "dropout", "Dropout at the output of each layer")
  val classes = new IntParam(this, "classes", "The number of classes")


  def setLabelCol(label: String): this.type = set(label, label)
  def setTargetCol(target: String): this.type = set(target, target)

  def setStartCol(s: String): this.type = set(startCol, s)
  def setEndCol(e: String): this.type = set(endCol, e)
  def setNerCol(col: String): this.type = set(nerCol, col)

  def setBatchSize(size: Int): this.type = set(batchSize, size)
  def setEpochs(number: Int): this.type = set(epochs, number)
  def setLearningRate(rate: Float): this.type = set(learningRate, rate)
  def setDropout(factor: Float): this.type = set(dropout, factor)
  def setClasses(k: Int): this.type = set(classes, k)


  // defaults
  setDefault(labelCol -> "label",
    batchSize -> 64,
    epochs -> 5,
    learningRate -> 0.0012f,
    dropout -> 0.05f,
    classes -> 2)

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): AssertionDLModel = {

    val sentences = dataset.
      withColumn("_text", extractTextUdf(col(getInputCols.head))).
      select("_text").
      collect().
      map(row => row.getAs[String]("_text").split(" "))

    val annotations: Array[AssertionAnnotationWithLabel] = {
      if (get(nerCol).isDefined) {
        dataset.
          select(col(getOrDefault(labelCol)), col(getOrDefault(nerCol))).
          collect.
          flatMap(row => AssertionAnnotationWithLabel.fromNer(
            row.getAs[String](getOrDefault(labelCol)),
            row.getAs(getOrDefault(nerCol))
          ))
      } else if (get(startCol).isDefined && get(endCol).isDefined) {
        dataset.
          select(col(getOrDefault(labelCol)), col($(startCol)), col($(endCol))).
          collect.
          map(row => AssertionAnnotationWithLabel(
            row.getAs[String](getOrDefault(labelCol)),
            row.getAs[Int]($(startCol)),
            row.getAs[Int]($(endCol))
          ))
      } else {
        throw new IllegalArgumentException("Either nerCol or startCol and endCol must be defined in order to train AssertionDL")
      }
    }

    /* infer labels and assign a number to each */
    val labelMappings = annotations.map(_.label).distinct

    val graph = new Graph()
    val session = new Session(graph)

    val graphStream = getClass.
      getResourceAsStream(s"/assertion_dl/blstm_34_32_30_${getOrDefault(embeddingsNDims)}_${getOrDefault(classes)}.pb")
    require(graphStream != null, "Graph not found for input parameters")

    val graphBytesDef = IOUtils.toByteArray(graphStream)
    graph.importGraphDef(graphBytesDef)

    val tf = new TensorflowWrapper(session, graph)
    val params = DatasetEncoderParams(labelMappings.toList, List.empty)
    val encoder = new AssertionDatasetEncoder(embeddings.get.getEmbeddings, params)

    val model = new TensorflowAssertion(tf, encoder, getOrDefault(batchSize), Verbose.All)

    model.train(sentences.zip(annotations),
      getOrDefault(learningRate),
      getOrDefault(batchSize),
      getOrDefault(dropout),
      0,
      getOrDefault(epochs)
    )

    val dlModel = new AssertionDLModel().
      setTensorflow(tf).
      setDatasetParams(model.encoder.params).
      setBatchSize($(batchSize)).
      setInputCols(getOrDefault(inputCols))

    if (get(nerCol).isDefined)
      dlModel
        .setNerCol($(nerCol))
    else
      dlModel
        .setStartCol($(startCol))
        .setEndCol($(endCol))
  }

  override val annotatorType: AnnotatorType = ASSERTION
  def this() = this(Identifiable.randomUID("ASSERTION"))

  def extractTextUdf: UserDefinedFunction = udf { document:mutable.WrappedArray[GenericRowWithSchema] =>
    document.head.getAs[String]("result")
  }
}

object AssertionDLApproach extends DefaultParamsReadable[AssertionDLApproach]