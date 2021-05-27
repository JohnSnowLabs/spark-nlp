package com.johnsnowlabs.nlp

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{BooleanParam, Param, ParamMap, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.types._

/**
  * Converts annotation results into a format that easier to use. It is useful to extract the results from Spark NLP
  * Pipelines. The Finisher outputs annotation(s) values into `String`.
  *
  * For more extended examples on document pre-processing see the
  * [[https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/databricks_notebooks/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers_v3.0.ipynb Spark NLP Workshop]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
  * import com.johnsnowlabs.nlp.Finisher
  *
  * val data = Seq((1, "New York and New Jersey aren't that far apart actually.")).toDF("id", "text")
  *
  * // Extracts Named Entities amongst other things
  * val pipeline = PretrainedPipeline("explain_document_dl")
  *
  * val finisher = new Finisher().setInputCols("entities").setOutputCols("output")
  * val explainResult = pipeline.transform(data)
  *
  * explainResult.selectExpr("explode(entities)").show(false)
  * +------------------------------------------------------------------------------------------------------------------------------------------------------+
  * |entities                                                                                                                                              |
  * +------------------------------------------------------------------------------------------------------------------------------------------------------+
  * |[[chunk, 0, 7, New York, [entity -> LOC, sentence -> 0, chunk -> 0], []], [chunk, 13, 22, New Jersey, [entity -> LOC, sentence -> 0, chunk -> 1], []]]|
  * +------------------------------------------------------------------------------------------------------------------------------------------------------+
  *
  * val result = finisher.transform(explainResult)
  * result.select("output").show(false)
  * +----------------------+
  * |output                |
  * +----------------------+
  * |[New York, New Jersey]|
  * +----------------------+
  * }}}
  *
  * @see [[com.johnsnowlabs.nlp.EmbeddingsFinisher EmbeddingsFinisher]] for finishing embeddings
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
class Finisher(override val uid: String)
  extends Transformer
    with DefaultParamsWritable {

  /**
    * Name of input annotation cols
    * @group param
    */
  val inputCols: StringArrayParam =
    new StringArrayParam(this, "inputCols", "Name of input annotation cols")

  /**
    * Name of finisher output cols
    * @group param
    */
  val outputCols: StringArrayParam =
    new StringArrayParam(this, "outputCols", "Name of finisher output cols")

  /**
    * Character separating annotations (Default: `#`)
    * @group param
    */
  val valueSplitSymbol: Param[String] =
    new Param(this, "valueSplitSymbol", "Character separating annotations (Default: `#`)")

  /**
    * Character separating annotations (Default: `@`)
    * @group param
    */
  val annotationSplitSymbol: Param[String] =
    new Param(this, "annotationSplitSymbol", "Character separating annotations (Default: `#`)")

  /**
    * Whether to remove annotation columns (Default: `true`)
    * @group param
    */
  val cleanAnnotations: BooleanParam =
    new BooleanParam(this, "cleanAnnotations", "Whether to remove annotation columns (Default: `true`)")

  /**
    * Annotation metadata format (Default: `false`)
    * @group param
    */
  val includeMetadata: BooleanParam =
    new BooleanParam(this, "includeMetadata", "Annotation metadata format (Default: `false`)")

  /**
    * Finisher generates an Array with the results instead of string (Default: `true`)
    * @group param
    */
  val outputAsArray: BooleanParam =
    new BooleanParam(this, "outputAsArray", "Finisher generates an Array with the results instead of string (Default: `true`)")

  /**
    * Whether to include embeddings vectors in the process (Default: `false`)
    * @group param
    */
  val parseEmbeddingsVectors: BooleanParam =
    new BooleanParam(this, "parseEmbeddingsVectors", "Whether to include embeddings vectors in the process (Default: `false`)")


  /**
    * Name of input annotation cols
    * @group setParam
    */
  def setInputCols(value: Array[String]): this.type = set(inputCols, value)

  /**
    * Name of input annotation cols
    * @group setParam
    */
  def setInputCols(value: String*): this.type = setInputCols(value.toArray)

  /**
    * Name of finisher output cols
    * @group setParam
    */
  def setOutputCols(value: Array[String]): this.type = set(outputCols, value)

  /**
    * Name of finisher output cols
    * @group setParam
    */
  def setOutputCols(value: String*): this.type = setOutputCols(value.toArray)

  /**
    * Character separating annotations (Default: `#`)
    * @group setParam
    */
  def setValueSplitSymbol(value: String): this.type = set(valueSplitSymbol, value)

  /**
    * Character separating annotations (Default: `#`)
    * @group setParam
    */
  def setAnnotationSplitSymbol(value: String): this.type = set(annotationSplitSymbol, value)

  /**
    * Whether to remove annotation columns (Default: `true`)
    * @group setParam
    */
  def setCleanAnnotations(value: Boolean): this.type = set(cleanAnnotations, value)

  /**
    * Annotation metadata format (Default: `false`)
    * @group setParam
    */
  def setIncludeMetadata(value: Boolean): this.type = set(includeMetadata, value)

  /**
    * Finisher generates an Array with the results instead of string (Default: `true`)
    * @group setParam
    */
  def setOutputAsArray(value: Boolean): this.type = set(outputAsArray, value)

  /**
    * Name of input annotation cols
    * @group getParam
    */
  def getOutputCols: Array[String] = get(outputCols).getOrElse(getInputCols.map("finished_" + _))

  /**
    * Name of finisher output cols
    * @group getParam
    */
  def getInputCols: Array[String] = $(inputCols)

  /**
    * Character separating annotations (Default: `#`)
    * @group getParam
    */
  def getValueSplitSymbol: String = $(valueSplitSymbol)

  /**
    * Character separating annotations (Default: `#`)
    * @group getParam
    */
  def getAnnotationSplitSymbol: String = $(annotationSplitSymbol)

  /**
    * Whether to remove annotation columns (Default: `true`)
    * @group getParam
    */
  def getCleanAnnotations: Boolean = $(cleanAnnotations)

  /**
    * Annotation metadata format (Default: `false`)
    * @group getParam
    */
  def getIncludeMetadata: Boolean = $(includeMetadata)

  /**
    * Finisher generates an Array with the results instead of string (Default: `true`)
    * @group getParam
    */
  def getOutputAsArray: Boolean = $(outputAsArray)

  setDefault(
    cleanAnnotations -> true,
    includeMetadata -> false,
    outputAsArray -> true,
    parseEmbeddingsVectors -> false,
    valueSplitSymbol -> "#",
    annotationSplitSymbol -> "@"
  )

  def this() = this(Identifiable.randomUID("finisher"))

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    require(getInputCols.length == getOutputCols.length, "inputCols and outputCols length must match")
    getInputCols.foreach {
      annotationColumn =>
        require(getInputCols.forall(schema.fieldNames.contains),
          s"pipeline annotator stages incomplete. " +
            s"expected: ${getInputCols.mkString(", ")}, " +
            s"found: ${schema.fields.filter(_.dataType == ArrayType(Annotation.dataType)).map(_.name).mkString(", ")}, " +
            s"among available: ${schema.fieldNames.mkString(", ")}")
        require(schema(annotationColumn).dataType == ArrayType(Annotation.dataType),
          s"column [$annotationColumn] must be an NLP Annotation column")
    }
    val metadataFields =  getOutputCols.flatMap(outputCol => {
      if ($(outputAsArray))
        Some(StructField(outputCol + "_metadata", MapType(StringType, StringType), nullable = false))
      else
        None
    })
    val outputFields = schema.fields ++
      getOutputCols.map(outputCol => {
        if ($(outputAsArray))
          StructField(outputCol, ArrayType(StringType), nullable = false)
        else
          StructField(outputCol, StringType, nullable = true)
      }) ++ metadataFields
    val cleanFields = if ($(cleanAnnotations)) outputFields.filterNot(f =>
      f.dataType == ArrayType(Annotation.dataType)
    ) else outputFields
    StructType(cleanFields)
  }

  override def transform(dataset: Dataset[_]): Dataset[Row] = {
    /*For some reason, Dataset[_] -> Dataset[Row] is not accepted through foldRight
    val flattened = getInputCols.foldRight(dataset)((inputCol, data) =>
      data.withColumn(inputCol, Annotation.flatten(data.col(inputCol))).toDF()
    )
    */
    require(getInputCols.length == getOutputCols.length, "inputCols and outputCols length must match")
    val cols = getInputCols.zip(getOutputCols)
    var flattened = dataset
    cols.foreach { case (inputCol, outputCol) =>
      flattened = {
        flattened.withColumn(
          outputCol, {
            if ($(outputAsArray))
              Annotation.flattenArray($(parseEmbeddingsVectors))(flattened.col(inputCol))
            else if (!$(includeMetadata))
              Annotation.flatten($(valueSplitSymbol), $(annotationSplitSymbol), $(parseEmbeddingsVectors))(flattened.col(inputCol))
            else
              Annotation.flattenDetail($(valueSplitSymbol), $(annotationSplitSymbol), $(parseEmbeddingsVectors))(flattened.col(inputCol))
          }
        )
      }
    }
    if ($(outputAsArray) && $(includeMetadata))
      cols.foreach { case (inputCol, outputCol) =>
        flattened = flattened.withColumn(
          outputCol + "_metadata",
          Annotation.flattenArrayMetadata(flattened.col(inputCol))
        )
      }
    if ($(cleanAnnotations)) flattened.drop(
      flattened.schema.fields
        .filter(_.dataType == ArrayType(Annotation.dataType))
        .map(_.name):_*)
      
    else flattened.toDF()
  }

}
object Finisher extends DefaultParamsReadable[Finisher]