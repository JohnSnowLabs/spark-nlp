package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.util.FinisherUtil
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{BooleanParam, Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Row}

class GraphFinisher(override val uid: String) extends Transformer {

  def this() = this(Identifiable.randomUID("graph_finisher"))

  /**
    * Name of input annotation cols
    * @group param
    */
  val inputCol = new Param[String](this, "inputCol", "Name of input annotation col")

  /**
    * Name of finisher output cols
    * @group param
    */
  val outputCol =
    new Param[String](this, "outputCol", "Name of finisher output col")

  /**
    * Finisher generates an Array with the results instead of string (Default: `true`)
    * @group param
    */
  val outputAsArray: BooleanParam =
    new BooleanParam(this, "outputAsArray", "Finisher generates an Array with the results")

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
    * Name of input annotation col
    * @group setParam
    */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /**
    * Name of finisher output col
    * @group setParam
    */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /**
    * Finisher generates an Array with the results instead of string (Default: `true`)
    * @group setParam
    */
  def setOutputAsArray(value: Boolean): this.type = set(outputAsArray, value)

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
    * Name of input annotation col
    * @group getParam
    */
  def getOutputCol: String = get(outputCol).getOrElse("finished_" + getInputCol)

  /**
    * Name of EmbeddingsFinisher output cols
    * @group getParam
    */
  def getInputCol: String = $(inputCol)

  setDefault(cleanAnnotations -> true, outputAsArray -> true, includeMetadata -> false)

  override def transform(dataset: Dataset[_]): DataFrame = {
   var flattenedDataSet = dataset.withColumn($(outputCol), {
     if ($(outputAsArray)) flattenPathsAsArray(dataset.col($(inputCol))) else flattenPaths(dataset.col($(inputCol)))
   })

   if ($(includeMetadata)) {
     flattenedDataSet = flattenedDataSet.withColumn($(outputCol) + "_metadata", flattenMetadata(dataset.col($(inputCol))))
   }

   FinisherUtil.cleaningAnnotations($(cleanAnnotations), flattenedDataSet)
  }

  def flattenPathsAsArray: UserDefinedFunction = udf { annotations: Seq[Row] =>
    annotations.flatMap{ row =>
       val metadata = row.getMap[String, String](4)
       val paths = metadata.flatMap{case (key, value) =>
        if (key.contains("path")) Some (value.split(",")) else None
       }.toList
       val pathsInRDFFormat = paths.map{ path =>
         val evenPathIndices = path.indices.toList.filter(index => index % 2 == 0)
         val sliceIndices =  evenPathIndices zip evenPathIndices.tail
         sliceIndices.map(sliceIndex => path.slice(sliceIndex._1, sliceIndex._2 + 1).toList)
       }
       pathsInRDFFormat
    }
  }

  def flattenPaths: UserDefinedFunction = udf { annotations: Seq[Row] =>
    annotations.flatMap{ row =>
      val metadata = row.getMap[String, String](4)
      val paths = metadata.flatMap{case (key, value) =>
        if (key.contains("path")) Some (value.split(",")) else None
      }.toList
      val pathsInRDFFormat = paths.map{ path =>
        val evenPathIndices = path.indices.toList.filter(index => index % 2 == 0)
        val sliceIndices =  evenPathIndices zip evenPathIndices.tail
        sliceIndices.map{ sliceIndex =>
          val node = path.slice(sliceIndex._1, sliceIndex._2 + 1)
          "(" + node.mkString(",") + ")"
        }
      }
      pathsInRDFFormat
    }
  }

  def flattenMetadata: UserDefinedFunction = udf { annotations: Seq[Row] =>
    annotations.flatMap { row =>
      val metadata = row.getMap[String, String](4)
      val relationships = metadata.flatMap{case (key, value) =>
        if (key.contains("relationship") || key.contains("entities")) Some("(" + value + ")") else None
      }.toList
      relationships
    }
  }

  override def copy(extra: ParamMap): Transformer = super.defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {

    FinisherUtil.checkIfInputColsExist(Array(getInputCol), schema)
    FinisherUtil.checkIfAnnotationColumnIsSparkNLPAnnotation(schema, getInputCol)

    require(Seq(AnnotatorType.NODE).contains(schema(getInputCol).metadata.getString("annotatorType")),
      s"column [$getInputCol] must be a ${AnnotatorType.NODE} type")

    val metadataFields =  FinisherUtil.getMetadataFields(Array(getOutputCol), $(outputAsArray))
    val outputFields = schema.fields ++
      FinisherUtil.getOutputFields(Array(getOutputCol), $(outputAsArray)) ++ metadataFields
    val cleanFields = FinisherUtil.getCleanFields($(cleanAnnotations), outputFields)

    StructType(cleanFields)
  }

}
