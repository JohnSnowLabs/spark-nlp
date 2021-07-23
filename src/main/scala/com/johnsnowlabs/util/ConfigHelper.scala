/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.util

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.hadoop.fs.FileSystem


object ConfigHelper {

  private lazy val sparkSession = ResourceHelper.spark

  // Configures s3 bucket where pretrained models are stored
  val pretrainedS3BucketKey = "spark.jsl.settings.pretrained.s3_bucket"

  // Configures s3 bucket where community pretrained models are stored
  val pretrainedCommunityS3BucketKey = "spark.jsl.settings.pretrained.s3_community_bucket"

  // Configures s3 path where pretrained models are stored
  val pretrainedS3PathKey = "spark.jsl.settings.pretrained.s3_path"

  // Configures cache folder where to cache pretrained models
  val pretrainedCacheFolder = "spark.jsl.settings.pretrained.cache_folder"

  // Configures log folder where to store annotator logs using OutputHelper
  val annotatorLogFolder = "spark.jsl.settings.annotator.log_folder"

  // Stores credentials for AWS S3 private models
  val awsCredentials = "spark.jsl.settings.pretrained.credentials"
  val accessKeyId: String = awsCredentials + ".access_key_id"
  val secretAccessKey: String = awsCredentials + ".secret_access_key"
  val awsProfileName: String = awsCredentials + ".aws_profile_name"

  val s3SocketTimeout = "spark.jsl.settings.pretrained.s3_socket_timeout"

  val storageTmpDir = "spark.jsl.settings.storage.cluster_tmp_dir"

  val serializationMode = "spark.jsl.settings.annotatorSerializationFormat"
  val useBroadcast = "spark.jsl.settings.useBroadcastForFeatures"

  def getConfigValueOrElse(property: String, defaultValue: String): String = {
    sparkSession.conf.get(property, defaultValue)
  }

  def getFileSystem: FileSystem = {
    FileSystem.get(sparkSession.sparkContext.hadoopConfiguration)
  }

  def getHadoopTmpDir: String = {
    sparkSession.sparkContext.hadoopConfiguration.get("hadoop.tmp.dir")
  }

}
