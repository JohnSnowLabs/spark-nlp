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

  val currentVersion = "4.2.0"
  val MavenSpark3 = s"com.johnsnowlabs.nlp:spark-nlp_2.12:$currentVersion"
  val MavenGpuSpark3 = s"com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:$currentVersion"
  val MavenSparkM1 = s"com.johnsnowlabs.nlp:spark-nlp-m1_2.12:$currentVersion"
  val MavenSparkAarch64 = s"com.johnsnowlabs.nlp:spark-nlp-aarch64_2.12:$currentVersion"

  /** Start SparkSession with Spark NLP
    *
    * @param gpu
    *   start Spark NLP with GPU
    * @param m1
    *   start Spark NLP for Apple M1 systems
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
    * @return
    *   SparkSession
    */
  def start(
      gpu: Boolean = false,
      m1: Boolean = false,
      aarch64: Boolean = false,
      memory: String = "16G",
      cache_folder: String = "",
      log_folder: String = "",
      cluster_tmp_dir: String = ""): SparkSession = {

    val build = SparkSession
      .builder()
      .appName("Spark NLP")
      .master("local[*]")
      .config("spark.driver.memory", memory)
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.kryoserializer.buffer.max", "2000M")
      .config("spark.driver.maxResultSize", "0")

    if (m1) {
      build.config("spark.jars.packages", MavenSparkM1)
    } else if (aarch64) {
      build.config("spark.jars.packages", MavenSparkAarch64)
    } else if (gpu) {
      build.config("spark.jars.packages", MavenGpuSpark3)
    } else {
      build.config("spark.jars.packages", MavenSpark3)
    }

    if (cache_folder.nonEmpty)
      build.config("spark.jsl.settings.pretrained.cache_folder", cache_folder)

    if (log_folder.nonEmpty)
      build.config("spark.jsl.settings.annotator.log_folder", log_folder)

    if (cluster_tmp_dir.nonEmpty)
      build.config("spark.jsl.settings.storage.cluster_tmp_dir", cluster_tmp_dir)

    build.getOrCreate()
  }

  def version(): String = {
    currentVersion
  }

}
