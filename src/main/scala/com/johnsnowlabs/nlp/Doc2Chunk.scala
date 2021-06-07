package com.johnsnowlabs.nlp

import org.apache.spark.ml.param.{BooleanParam, Param}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{ArrayType, MetadataBuilder, StringType, StructType}
import org.slf4j.LoggerFactory

/**
  * Created by saif on 06/07/17.
  */
/**
  * Converts `DOCUMENT` type annotations into `CHUNK` type with the contents of a `chunkCol`.
  * Chunk text must be contained within input `DOCUMENT`. May be either `StringType` or `ArrayType[StringType]`
  * (using [[setIsArray]]). Useful for annotators that require a CHUNK type input.
  *
  * For more extended examples on document pre-processing see the
  * [[https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/databricks_notebooks/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers_v3.0.ipynb Spark NLP Workshop]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.{Doc2Chunk, DocumentAssembler}
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")
  * val chunkAssembler = new Doc2Chunk()
  *   .setInputCols("document")
  *   .setChunkCol("target")
  *   .setOutputCol("chunk")
  *   .setIsArray(true)
  *
  * val data = Seq(
  *   ("Spark NLP is an open-source text processing library for advanced natural language processing.",
  *     Seq("Spark NLP", "text processing library", "natural language processing"))
  * ).toDF("text", "target")
  *
  * val pipeline = new Pipeline().setStages(Array(documentAssembler, chunkAssembler)).fit(data)
  * val result = pipeline.transform(data)
  *
  * result.selectExpr("chunk.result", "chunk.annotatorType").show(false)
  * +-----------------------------------------------------------------+---------------------+
  * |result                                                           |annotatorType        |
  * +-----------------------------------------------------------------+---------------------+
  * |[Spark NLP, text processing library, natural language processing]|[chunk, chunk, chunk]|
  * +-----------------------------------------------------------------+---------------------+
  * }}}
  *
  * @see [[Chunk2Doc]] for converting `CHUNK` annotations to `DOCUMENT`
  * @param uid required uid for storing annotator to disk
  * @groupname anno Annotator types
  * @groupdesc anno Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupname Ungrouped Members
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio Ungrouped 3
  * @groupprio setParam  4
  * @groupprio getParam  5
  * @groupdesc param A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
  */
class Doc2Chunk(override val uid: String) extends RawAnnotator[Doc2Chunk]{

  import com.johnsnowlabs.nlp.AnnotatorType._

  /**
    * Output annotator types: CHUNK
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = CHUNK

  /**
    * Input annotator types: DOCUMENT
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT)

  private val logger = LoggerFactory.getLogger("ChunkAssembler")

  /**
    * Column that contains string. Must be part of DOCUMENT
    * @group param
    */
  val chunkCol = new Param[String](this, "chunkCol", "Column that contains string. Must be part of DOCUMENT")

  /**
    * Column that has a reference of where the chunk begins
    * @group param
    */
  val startCol = new Param[String](this, "startCol", "Column that has a reference of where the chunk begins")

  /**
    * Whether start col is by whitespace tokens (Default: `false`)
    * @group param
    */
  val startColByTokenIndex = new BooleanParam(this, "startColByTokenIndex", "Whether start col is by whitespace tokens (Default: `false`)")

  /**
    * Whether the chunkCol is an array of strings (Default: `false`)
    * @group param
    */
  val isArray = new BooleanParam(this, "isArray", "Whether the chunkCol is an array of strings (Default: `false")

  /**
    * Whether to fail the job if a chunk is not found within document, return empty otherwise (Default: `false`)
    * @group param
    */
  val failOnMissing = new BooleanParam(this, "failOnMissing", "Whether to fail the job if a chunk is not found within document, return empty otherwise (Default: `false`)")

  /**
    * Whether to lower case for matching case (Default: `true`)
    * @group param
    */
  val lowerCase = new BooleanParam(this, "lowerCase", "Whether to lower case for matching case (Default: `true")

  setDefault(
    startColByTokenIndex -> false,
    isArray -> false,
    failOnMissing -> false,
    lowerCase -> true
  )

  /**
    * Column that contains string. Must be part of DOCUMENT
    * @group setParam
    */
  def setChunkCol(value: String): this.type = set(chunkCol, value)

  /**
    * Column that contains string. Must be part of DOCUMENT
    * @group getParam
    */
  def getChunkCol: String = $(chunkCol)

  /**
    * Column that has a reference of where the chunk begins
    * @group setParam
    */
  def setStartCol(value: String): this.type = set(startCol, value)

  /**
    * Column that has a reference of where the chunk begins
    * @group getParam
    */
  def getStartCol: String = $(startCol)

  /**
    * Whether start col is by whitespace tokens (Default: `false`)
    * @group setParam
    */
  def setStartColByTokenIndex(value: Boolean): this.type = set(startColByTokenIndex, value)

  /**
    * Whether start col is by whitespace tokens (Default: `false`)
    * @group getParam
    */
  def getStartColByTokenIndex: Boolean = $(startColByTokenIndex)

  /**
    * Whether the chunkCol is an array of strings (Default: `false`)
    * @group setParam
    */
  def setIsArray(value: Boolean): this.type = set(isArray, value)

  /**
    * Whether the chunkCol is an array of strings (Default: `false`)
    * @group getParam
    */
  def getIsArray: Boolean = $(isArray)

  /**
    * Whether to fail the job if a chunk is not found within document, return empty otherwise (Default: `false`)
    * @group setParam
    */
  def setFailOnMissing(value: Boolean): this.type = set(failOnMissing, value)

  /**
    * Whether to fail the job if a chunk is not found within document, return empty otherwise (Default: `false`)
    * @group getParam
    */
  def getFailOnMissing: Boolean = $(failOnMissing)

  /**
    * Whether to lower case for matching case (Default: `true`)
    * @group setParam
    */
  def setLowerCase(value: Boolean): this.type = set(lowerCase, value)

  /**
    * Whether to lower case for matching case (Default: `true`)
    * @group getParam
    */
  def getLowerCase: Boolean = $(lowerCase)

  def this() = this(Identifiable.randomUID("DOC2CHUNK"))

  override protected def extraValidate(structType: StructType): Boolean = {
    if (get(chunkCol).isEmpty)
      true
    else if ($(isArray))
      structType.fields.find(_.name == $(chunkCol)).exists(_.dataType == ArrayType(StringType, containsNull=true))
    else
      structType.fields.find(_.name == $(chunkCol)).exists(_.dataType == StringType)
  }

  override protected def extraValidateMsg: AnnotatorType =
    if ($(isArray)) s"${$(chunkCol)} must be ArrayType(StringType)"
    else s"${$(chunkCol)} must be StringType"

  private def buildFromChunk(annotation: Annotation, chunk: String, startIndex: Int, chunkIdx: Int) = {
    /** This will break if there are two identical chunks */
    val beginning = get(lowerCase) match {
      case Some(true) => annotation.result.toLowerCase.indexOf(chunk, startIndex)
      case _ => annotation.result.indexOf(chunk, startIndex)
    }
    val ending = beginning + chunk.length - 1
    if (chunk.trim.isEmpty || beginning == -1) {
      val message = s"Cannot proceed to assemble CHUNK, because could not find: `$chunk` within: `${annotation.result}`"
      if ($(failOnMissing))
        throw new Exception(message)
      else
        logger.warn(message)
      None
    } else {
      Some(Annotation(
        outputAnnotatorType,
        beginning,
        ending,
        chunk,
        annotation.metadata ++ Map("chunk" -> chunkIdx.toString)
      ))
    }
  }

  def tokenIndexToCharIndex(text: String, tokenIndex: Int): Int = {
    var i = 0
    text.split(" ").map(token => {
      val o = (token, i)
      i += token.length + 1
      o
    }).apply(tokenIndex)._2
  }

  private def convertDocumentToChunk = udf {
    document: Seq[Row] =>
      val annotations = document.map(Annotation(_))
      annotations.map{annotation =>
        Annotation(
          AnnotatorType.CHUNK,
          annotation.begin,
          annotation.end,
          annotation.result,
          annotation.metadata ++ Map("chunk" -> "0")
        )
      }
  }

  private def assembleChunks = udf {
    (annotationProperties: Seq[Row], chunks: Seq[String]) =>
      val annotations = annotationProperties.map(Annotation(_))
      annotations.flatMap(annotation => {
        chunks.zipWithIndex.flatMap{case (chunk, idx) => buildFromChunk(annotation, chunk, 0, idx)}
      })
  }

  private def assembleChunk = udf {
    (annotationProperties: Seq[Row], chunk: String) =>
      val annotations = annotationProperties.map(Annotation(_))
      annotations.flatMap(annotation => {
        buildFromChunk(annotation, chunk, 0, 0)
      })
  }

  private def assembleChunkWithStart = udf {
    (annotationProperties: Seq[Row], chunk: String, start: Int) =>
      val annotations = annotationProperties.map(Annotation(_))
      annotations.flatMap(annotation => {
        if ($(startColByTokenIndex))
          buildFromChunk(annotation, chunk, tokenIndexToCharIndex(annotation.result, start), 0)
        else
          buildFromChunk(annotation, chunk, start, 0)
      })
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    if (get(chunkCol).isEmpty)
      dataset.withColumn($(outputCol), wrapColumnMetadata(convertDocumentToChunk(col(getInputCols.head))))
    else if ($(isArray))
      dataset.withColumn($(outputCol), wrapColumnMetadata(assembleChunks(col(getInputCols.head), col($(chunkCol)))))
    else if (get(startCol).isDefined)
      dataset.withColumn($(outputCol), wrapColumnMetadata(assembleChunkWithStart(
        col($(inputCols).head),
        col($(chunkCol)),
        col($(startCol))
      )))
    else
      dataset.withColumn($(outputCol), wrapColumnMetadata(assembleChunk(col(getInputCols.head), col($(chunkCol)))))
  }

}
object Doc2Chunk extends DefaultParamsReadable[Doc2Chunk]
