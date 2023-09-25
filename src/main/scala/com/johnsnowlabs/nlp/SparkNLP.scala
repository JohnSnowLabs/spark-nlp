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

import org.apache.spark.sql.SparkSession

object SparkNLP {

  val currentVersion = "5.1.2"
  val MavenSpark3 = s"com.johnsnowlabs.nlp:spark-nlp_2.12:$currentVersion"
  val MavenGpuSpark3 = s"com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:$currentVersion"
  val MavenSparkSilicon = s"com.johnsnowlabs.nlp:spark-nlp-silicon_2.12:$currentVersion"
  val MavenSparkAarch64 = s"com.johnsnowlabs.nlp:spark-nlp-aarch64_2.12:$currentVersion"

  /** Start SparkSession with Spark NLP
    *
    * @param gpu
    *   start Spark NLP with GPU
    * @param apple_silicon
    *   start Spark NLP for Apple M1 & M2 systems
    * @param aarch64
    *   start Spark NLP for Linux Aarch64 systems
    * @param memory
    *   set driver memory for SparkSession
    * @param cache_folder
    *   The location to download and extract pretrained Models and Pipelines (by default, it will
    *   be in the users home directory under `cache_pretrained`.)
    * @param log_folder
    *   The location to use on a cluster for temporarily files such as unpacking indexes for
    *   WordEmbeddings. By default, this locations is the location of `hadoop.tmp.dir` set via
    *   Hadoop configuration for Apache Spark. NOTE: `S3` is not supported and it must be local,
    *   HDFS, or DBFS.
    * @param cluster_tmp_dir
    *   The location to save logs from annotators during training (By default, it will be in the
    *   users home directory under `annotator_logs`.)
    * @param params
    *   Custom parameters to set for the Spark configuration (Default: `Map.empty`)
    * @return
    *   SparkSession
    */
  def start(
      gpu: Boolean = false,
      apple_silicon: Boolean = false,
      aarch64: Boolean = false,
      memory: String = "16G",
      cache_folder: String = "",
      log_folder: String = "",
      cluster_tmp_dir: String = "",
      params: Map[String, String] = Map.empty): SparkSession = {

    if (SparkSession.getActiveSession.isDefined)
      println("Warning: Spark Session already created, some configs may not be applied.")

    val builder = SparkSession
      .builder()
      .appName("Spark NLP")
      .config("spark.driver.memory", memory)
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.kryoserializer.buffer.max", "2000M")
      .config("spark.driver.maxResultSize", "0")

    // get the set cores by users since local[*] will override spark.driver.cores if set
    if (params.contains("spark.driver.cores")) {
      builder.master("local[" + params("spark.driver.cores") + "]")
    } else {
      builder.master("local[*]")
    }

    val sparkNlpJar =
      if (apple_silicon) MavenSparkSilicon
      else if (aarch64) MavenSparkAarch64
      else if (gpu) MavenGpuSpark3
      else MavenSpark3

    if (!params.contains("spark.jars.packages")) {
      builder.config("spark.jars.packages", sparkNlpJar)
    }

    params.foreach {
      case (key, value) if key == "spark.jars.packages" =>
        builder.config(key, sparkNlpJar + "," + value)
      case (key, value) =>
        builder.config(key, value)
    }

    if (cache_folder.nonEmpty)
      builder.config("spark.jsl.settings.pretrained.cache_folder", cache_folder)

    if (log_folder.nonEmpty)
      builder.config("spark.jsl.settings.annotator.log_folder", log_folder)

    if (cluster_tmp_dir.nonEmpty)
      builder.config("spark.jsl.settings.storage.cluster_tmp_dir", cluster_tmp_dir)

    builder.getOrCreate()
  }

  def version(): String = {
    currentVersion
  }

}
