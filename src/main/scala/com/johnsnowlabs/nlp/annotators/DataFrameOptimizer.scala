/*
 * Copyright 2017-2025 John Snow Labs
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
package com.johnsnowlabs.nlp.annotators

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{BooleanParam, IntParam, Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}


/**
 * Optimizes a [[org.apache.spark.sql.DataFrame]] by adjusting the number of partitions, optionally caching it,
 * and optionally persisting it to disk.
 *
 * This transformer is useful when fine-tuning performance in a Spark NLP pipeline or preparing data
 * for export or downstream tasks. It provides options to control the number of partitions directly,
 * or to calculate them using the number of executor cores and workers. Additionally, it can persist
 * the DataFrame in CSV, JSON, or Parquet format with configurable writer options.
 *
 * == Parameters ==
 * - `executorCores` (Int): Number of cores per executor, used to calculate partitions.
 * - `numWorkers` (Int): Total number of executor nodes.
 * - `numPartitions` (Int): Target number of partitions. Overrides the computed value from cores × workers.
 * - `doCache` (Boolean): Whether to cache the transformed DataFrame.
 * - `persistPath` (String): Optional path to write the output DataFrame.
 * - `persistFormat` (String): File format for persistence. Supported: `csv`, `json`, `parquet`.
 * - `outputOptions` (Map[String, String]): Extra options passed to the DataFrameWriter.
 *
 * == Example ==
 * {{{
 * val optimizer = new DataFrameOptimizer()
 *   .setExecutorCores(4)
 *   .setNumWorkers(5)
 *   .setDoCache(true)
 *   .setPersistPath("/tmp/output")
 *   .setPersistFormat("parquet")
 *   .setOutputOptions(Map("compression" -> "snappy"))
 *
 * val optimizedDF = optimizer.transform(inputDF)
 * }}}
 *
 * This transformer does not modify the schema of the DataFrame.
 *
 * @groupname param Param Definitions
 * @groupname getParam Get Param values
 * @groupname setParam Set Param values
 * @groupname transform Transformation functions
 * @groupname Ungrouped Members
 */

class DataFrameOptimizer(override val uid: String)
    extends Transformer
    with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("DATAFRAME_OPTIMIZER"))

  val executorCores = new IntParam(this, "executorCores", "Executor cores per worker")
  val numWorkers = new IntParam(this, "numWorkers", "Total number of Spark workers")
  val numPartitions = new IntParam(
    this,
    "numPartitions",
    "Target number of partitions (overrides executorCores * numWorkers)")
  val doCache = new BooleanParam(this, "doCache", "Whether to cache the DataFrame")
  val persistPath = new Param[String](this, "persistPath", "Path to persist DataFrame (optional)")
  val persistFormat =
    new Param[String](this, "persistFormat", "File format for persistence (csv/json/parquet)")
  val outputOptions = new Param[Map[String, String]](
    this,
    "outputOptions",
    "Writer options for output format (csv/json/parquet)")

  def setExecutorCores(value: Int): this.type = set(executorCores, value)
  def setNumWorkers(value: Int): this.type = set(numWorkers, value)
  def setNumPartitions(value: Int): this.type = set(numPartitions, value)
  def setDoCache(value: Boolean): this.type = set(doCache, value)
  def setPersistPath(value: String): this.type = set(persistPath, value)
  def setPersistFormat(value: String): this.type = set(persistFormat, value)
  def setOutputOptions(options: Map[String, String]): this.type = set(outputOptions, options)

  override def transformSchema(schema: StructType): StructType = schema

  override def transform(dataset: Dataset[_]): DataFrame = {
    val partCount = if (isDefined(numPartitions)) {
      $(numPartitions)
    } else if (isDefined(executorCores) && isDefined(numWorkers)) {
      $(executorCores) * $(numWorkers)
    } else {
      throw new IllegalArgumentException(
        "Must specify either numPartitions or both executorCores and numWorkers.")
    }

    var optimizedDf = dataset.toDF().repartition(partCount)

    if ($(doCache)) optimizedDf = optimizedDf.cache()

    val writer = if (isDefined(outputOptions)) {
      optimizedDf.write.mode("overwrite").options($(outputOptions))
    } else {
      optimizedDf.write.mode("overwrite")
    }

    if (isDefined(persistPath)) {
      val format = $(persistFormat).toLowerCase
      format match {
        case "csv" => writer.csv($(persistPath))
        case "json" => writer.json($(persistPath))
        case "parquet" => writer.parquet($(persistPath))
        case _ => throw new IllegalArgumentException(s"Unsupported format: $format")
      }
    }

    optimizedDf
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

}
