/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.AnnotatorType.NODE
import com.johnsnowlabs.nlp.util.{AnnotationRowUtils, FinisherUtil, SparkNlpConfig}
import com.johnsnowlabs.util.Version
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{BooleanParam, Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

/** Helper class to convert the knowledge graph from GraphExtraction into a generic format, such
  * as RDF.
  *
  * ==Example==
  * This is a continuation of the example of
  * [[com.johnsnowlabs.nlp.annotators.GraphExtraction GraphExtraction]]. To see how the graph is
  * extracted, see the documentation of that class.
  * {{{
  * import com.johnsnowlabs.nlp.GraphFinisher
  *
  * val graphFinisher = new GraphFinisher()
  *   .setInputCol("graph")
  *   .setOutputCol("graph_finished")
  *   .setOutputAsArray(false)
  *
  * val finishedResult = graphFinisher.transform(result)
  * finishedResult.select("text", "graph_finished").show(false)
  * +-----------------------------------------------------+-----------------------------------------------------------------------+
  * |text                                                 |graph_finished                                                         |
  * +-----------------------------------------------------+-----------------------------------------------------------------------+
  * |You and John prefer the morning flight through Denver|[[(prefer,nsubj,morning), (morning,flat,flight), (flight,flat,Denver)]]|
  * +-----------------------------------------------------+-----------------------------------------------------------------------+
  * }}}
  * @see
  *   [[com.johnsnowlabs.nlp.annotators.GraphExtraction GraphExtraction]] to extract the graph.
  * @param uid
  *   required uid for storing annotator to disk
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
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
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */
class GraphFinisher(override val uid: String) extends Transformer {

  def this() = this(Identifiable.randomUID("graph_finisher"))

  /** Name of input annotation cols
    *
    * @group param
    */
  val inputCol = new Param[String](this, "inputCol", "Name of input annotation col")

  /** Name of finisher output cols
    *
    * @group param
    */
  val outputCol =
    new Param[String](this, "outputCol", "Name of finisher output col")

  /** Finisher generates an Array with the results instead of string (Default: `true`)
    *
    * @group param
    */
  val outputAsArray: BooleanParam =
    new BooleanParam(this, "outputAsArray", "Finisher generates an Array with the results")

  /** Whether to remove annotation columns (Default: `true`)
    *
    * @group param
    */
  val cleanAnnotations: BooleanParam =
    new BooleanParam(
      this,
      "cleanAnnotations",
      "Whether to remove annotation columns (Default: `true`)")

  /** Annotation metadata format (Default: `false`)
    *
    * @group param
    */
  val includeMetadata: BooleanParam =
    new BooleanParam(this, "includeMetadata", "Annotation metadata format (Default: `false`)")

  /** Name of input annotation col
    *
    * @group setParam
    */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** Name of finisher output col
    *
    * @group setParam
    */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** Finisher generates an Array with the results instead of string (Default: `true`)
    *
    * @group setParam
    */
  def setOutputAsArray(value: Boolean): this.type = set(outputAsArray, value)

  /** Whether to remove annotation columns (Default: `true`)
    *
    * @group setParam
    */
  def setCleanAnnotations(value: Boolean): this.type = set(cleanAnnotations, value)

  /** Annotation metadata format (Default: `false`)
    *
    * @group setParam
    */
  def setIncludeMetadata(value: Boolean): this.type = set(includeMetadata, value)

  /** Name of input annotation col
    *
    * @group getParam
    */
  def getOutputCol: String = get(outputCol).getOrElse("finished_" + getInputCol)

  /** Name of EmbeddingsFinisher output cols
    *
    * @group getParam
    */
  def getInputCol: String = $(inputCol)

  setDefault(cleanAnnotations -> true, outputAsArray -> true, includeMetadata -> false)

  private val METADATA_STRUCT_INDEX = 4

  override def transform(dataset: Dataset[_]): DataFrame = {
    if (isSpark4OrNewer(dataset)) {
      transformWithRows(dataset)
    } else {
      var flattenedDataSet = dataset.withColumn(
        $(outputCol), {
          if ($(outputAsArray)) flattenPathsAsArray(dataset.col($(inputCol)))
          else flattenPaths(dataset.col($(inputCol)))
        })

      if ($(includeMetadata)) {
        flattenedDataSet = flattenedDataSet.withColumn(
          $(outputCol) + "_metadata",
          flattenMetadata(dataset.col($(inputCol))))
      }

      FinisherUtil.cleaningAnnotations($(cleanAnnotations), flattenedDataSet)
    }
  }

  private def isSpark4OrNewer(dataset: Dataset[_]): Boolean =
    Version.parse(dataset.sparkSession.version).toFloat >= 4.0f

  private def outputSchemaWithReplacement(inputSchema: StructType): StructType = {
    val outputField =
      if ($(outputAsArray))
        StructField(
          getOutputCol,
          ArrayType(ArrayType(ArrayType(StringType, containsNull = true), containsNull = true)),
          nullable = false)
      else
        StructField(
          getOutputCol,
          ArrayType(ArrayType(StringType, containsNull = true), containsNull = true),
          nullable = true)

    var outputFields = inputSchema.fields
    val outputIndex = outputFields.indexWhere(_.name == outputField.name)
    outputFields =
      if (outputIndex >= 0) outputFields.updated(outputIndex, outputField)
      else outputFields :+ outputField

    if ($(includeMetadata)) {
      val metadataField =
        StructField(getOutputCol + "_metadata", ArrayType(StringType), nullable = false)
      val metadataIndex = outputFields.indexWhere(_.name == metadataField.name)
      outputFields =
        if (metadataIndex >= 0) outputFields.updated(metadataIndex, metadataField)
        else outputFields :+ metadataField
    }

    StructType(FinisherUtil.getCleanFields($(cleanAnnotations), outputFields))
  }

  private def transformWithRows(dataset: Dataset[_]): DataFrame = {
    val inputDataFrame = dataset.toDF()
    val outputSchema = outputSchemaWithReplacement(inputDataFrame.schema)
    val inputIndex = inputDataFrame.schema.fieldIndex(getInputCol)
    val inputFieldNames = inputDataFrame.schema.fieldNames

    implicit val encoder: ExpressionEncoder[Row] =
      SparkNlpConfig.getEncoder(inputDataFrame, outputSchema)

    inputDataFrame
      .mapPartitions { rows =>
        rows.map { row =>
          val annotationRows = AnnotationRowUtils.extractAnnotationRows(row, inputIndex)
          val inputValuesByName = inputFieldNames.zip(row.toSeq).toMap
          val outputValue =
            if ($(outputAsArray)) flattenPathRowsAsArray(annotationRows)
            else flattenPathRows(annotationRows)
          val baseValues = Map(getOutputCol -> outputValue)
          val outputValuesByName =
            if ($(includeMetadata)) {
              baseValues + (getOutputCol + "_metadata" -> flattenMetadataRows(annotationRows))
            } else {
              baseValues
            }

          Row.fromSeq(outputSchema.fields.toIndexedSeq.map { field =>
            outputValuesByName.getOrElse(field.name, inputValuesByName(field.name))
          })
        }
      }
      .toDF()
  }

  private def buildPathsAsArray(metadata: Map[String, String]): List[List[List[String]]] = {
    val paths = extractPaths(metadata)

    val pathsInRDFFormat = paths.map { path =>
      val evenPathIndices = path.indices.toList.filter(index => index % 2 == 0)
      val sliceIndices = evenPathIndices zip evenPathIndices.tail
      sliceIndices.map(sliceIndex => path.slice(sliceIndex._1, sliceIndex._2 + 1).toList)
    }
    pathsInRDFFormat
  }

  private def buildPaths(metadata: Map[String, String]): List[List[String]] = {
    val paths = extractPaths(metadata)

    val pathsInRDFFormat = paths.map { path =>
      val evenPathIndices = path.indices.toList.filter(index => index % 2 == 0)
      val sliceIndices = evenPathIndices zip evenPathIndices.tail
      sliceIndices.map { sliceIndex =>
        val node = path.slice(sliceIndex._1, sliceIndex._2 + 1)
        "(" + node.mkString(",") + ")"
      }
    }
    pathsInRDFFormat
  }

  private def extractPaths(metadata: Map[String, String]): List[Array[String]] = {
    val paths = metadata.flatMap { case (key, value) =>
      if (key.contains("path")) Some(value.split(",")) else None
    }.toList

    paths
  }

  private def flattenPathsAsArray: UserDefinedFunction = udf { annotations: Seq[Row] =>
    flattenPathRowsAsArray(annotations)
  }

  private def flattenPaths: UserDefinedFunction = udf { annotations: Seq[Row] =>
    flattenPathRows(annotations)
  }

  private def flattenMetadata: UserDefinedFunction = udf { annotations: Seq[Row] =>
    flattenMetadataRows(annotations)
  }

  private def flattenPathRowsAsArray(
      annotations: scala.collection.Seq[Row]): scala.collection.Seq[List[List[String]]] = {
    annotations.flatMap { row =>
      val metadata = row.getMap[String, String](METADATA_STRUCT_INDEX)
      buildPathsAsArray(metadata.toMap)
    }
  }

  private def flattenPathRows(
      annotations: scala.collection.Seq[Row]): scala.collection.Seq[List[String]] = {
    annotations.flatMap { row =>
      val metadata = row.getMap[String, String](METADATA_STRUCT_INDEX)
      buildPaths(metadata.toMap)
    }
  }

  private def flattenMetadataRows(
      annotations: scala.collection.Seq[Row]): scala.collection.Seq[String] = {
    annotations.flatMap { row =>
      val metadata = row.getMap[String, String](4)
      val relationships = metadata.flatMap { case (key, value) =>
        if (key.contains("relationship") || key.contains("entities")) Some("(" + value + ")")
        else None
      }.toList
      relationships
    }
  }

  def annotate(metadata: Map[String, String]): Seq[Annotation] = {
    val pathsInRDFFormat = buildPaths(metadata)
    val result = pathsInRDFFormat.map(pathRDF => "[" + pathRDF.mkString(",") + "]").mkString(",")
    Seq(Annotation(NODE, 0, 0, result, Map()))
  }

  override def copy(extra: ParamMap): Transformer = super.defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {

    FinisherUtil.checkIfInputColsExist(Array(getInputCol), schema)
    FinisherUtil.checkIfAnnotationColumnIsSparkNLPAnnotation(schema, getInputCol)

    require(
      Seq(AnnotatorType.NODE).contains(schema(getInputCol).metadata.getString("annotatorType")),
      s"column [$getInputCol] must be a ${AnnotatorType.NODE} type")

    val metadataFields = FinisherUtil.getMetadataFields(Array(getOutputCol), $(outputAsArray))
    val outputFields = schema.fields ++
      FinisherUtil.getOutputFields(Array(getOutputCol), $(outputAsArray)) ++ metadataFields
    val cleanFields = FinisherUtil.getCleanFields($(cleanAnnotations), outputFields)

    StructType(cleanFields)
  }

}
