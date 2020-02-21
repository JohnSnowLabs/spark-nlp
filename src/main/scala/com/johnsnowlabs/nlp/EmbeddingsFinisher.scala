package com.johnsnowlabs.nlp

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.{BooleanParam, ParamMap, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Dataset, Row}

class EmbeddingsFinisher(override val uid: String)
  extends Transformer
    with DefaultParamsWritable {

  protected val inputCols: StringArrayParam =
    new StringArrayParam(this, "inputCols", "name of input annotation cols containing embeddings")
  protected val outputCols: StringArrayParam =
    new StringArrayParam(this, "outputCols", "name of EmbeddingsFinisher output cols")
  protected val cleanAnnotations: BooleanParam =
    new BooleanParam(this, "cleanAnnotations", "whether to remove all the existing annotation columns")
  protected val outputAsVector: BooleanParam =
    new BooleanParam(this, "outputAsVector", "if enabled it will output the embeddings as Vectors instead of arrays")

  def setInputCols(value: Array[String]): this.type = set(inputCols, value)
  def setInputCols(value: String*): this.type = setInputCols(value.toArray)
  def setOutputCols(value: Array[String]): this.type = set(outputCols, value)
  def setOutputCols(value: String*): this.type = setOutputCols(value.toArray)
  def setCleanAnnotations(value: Boolean): this.type = set(cleanAnnotations, value)
  def setOutputAsVector(value: Boolean): this.type = set(outputAsVector, value)

  def getOutputCols: Array[String] = get(outputCols).getOrElse(getInputCols.map("finished_" + _))
  def getInputCols: Array[String] = $(inputCols)
  def getCleanAnnotations: Boolean = $(cleanAnnotations)
  def getOutputAsVector: Boolean = $(outputAsVector)

  setDefault(
    cleanAnnotations -> true,
    outputAsVector -> false
  )

  def this() = this(Identifiable.randomUID("embeddings_finisher"))

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {

    require(getInputCols.length == getOutputCols.length, "inputCols and outputCols length must match")

    val embeddingsAnnotators = Seq(
      AnnotatorType.WORD_EMBEDDINGS,
      AnnotatorType.SENTENCE_EMBEDDINGS
    )

    getInputCols.foreach {
      annotationColumn =>

        /**
          * Check if the inpuptCols exist
          */
        require(getInputCols.forall(schema.fieldNames.contains),
          s"pipeline annotator stages incomplete. " +
            s"expected: ${getInputCols.mkString(", ")}, " +
            s"found: ${schema.fields.filter(_.dataType == ArrayType(Annotation.dataType)).map(_.name).mkString(", ")}, " +
            s"among available: ${schema.fieldNames.mkString(", ")}")

        /**
          * Check if the annotationColumn is(are) Spark NLP annotations
          */
        require(schema(annotationColumn).dataType == ArrayType(Annotation.dataType),
          s"column [$annotationColumn] must be an NLP Annotation column")

        /**
          * Check if the annotationColumn has embeddings
          * It must be at least of one these annotators: WordEmbeddings, BertEmbeddings, ChunkEmbeddings, or SentenceEmbeddings
          */
        require(embeddingsAnnotators.contains(schema(annotationColumn).metadata.getString("annotatorType")),
          s"column [$annotationColumn] must be a type of either WordEmbeddings, BertEmbeddings, ChunkEmbeddings, or SentenceEmbeddings")

    }
    val metadataFields =  getOutputCols.flatMap(outputCol => {
      if ($(outputAsVector))
        Some(StructField(outputCol + "_metadata", MapType(StringType, StringType), nullable = false))
      else
        None
    })

    val outputFields = schema.fields ++
      getOutputCols.map(outputCol => {
        if ($(outputAsVector))
          StructField(outputCol, ArrayType(VectorType), nullable = false)
        else
          StructField(outputCol, ArrayType(FloatType), nullable = false)
      }) ++ metadataFields

    val cleanFields = if ($(cleanAnnotations)) outputFields.filterNot(f =>
      f.dataType == ArrayType(Annotation.dataType)
    ) else outputFields
    StructType(cleanFields)
  }

  private def vectorsAsArray: UserDefinedFunction = udf { embeddings: Seq[Seq[Float]] =>
    embeddings
  }

  private def vectorsAsVectorType: UserDefinedFunction = udf { embeddings: Seq[Seq[Float]] =>
    embeddings.map(embedding =>
      Vectors.dense(embedding.toArray.map(_.toDouble))
    )
  }

  override def transform(dataset: Dataset[_]): Dataset[Row] = {
    require(getInputCols.length == getOutputCols.length, "inputCols and outputCols length must match")

    val cols = getInputCols.zip(getOutputCols)
    var flattened = dataset
    cols.foreach { case (inputCol, outputCol) =>
      flattened = {
        flattened.withColumn(
          outputCol, {
            if ($(outputAsVector))
              vectorsAsVectorType(flattened.col(inputCol + ".embeddings"))
            else
              vectorsAsArray(flattened.col(inputCol + ".embeddings"))
          }
        )
      }
    }

    if ($(cleanAnnotations)) flattened.drop(
      flattened.schema.fields
        .filter(_.dataType == ArrayType(Annotation.dataType))
        .map(_.name):_*)
    else flattened.toDF()
  }

}
object EmbeddingsFinisher extends DefaultParamsReadable[EmbeddingsFinisher]