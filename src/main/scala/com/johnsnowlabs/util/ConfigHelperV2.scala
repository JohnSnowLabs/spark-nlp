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


object ConfigHelperV2 {

  private lazy val sparkSession = ResourceHelper.spark

  // Configures s3 bucket where pretrained models are stored
  val pretrainedS3BucketKey = "jsl.settings.pretrained.s3_bucket"

  // Configures s3 bucket where community pretrained models are stored
  val pretrainedCommunityS3BucketKey = "jsl.settings.pretrained.s3_community_bucket"

  // Configures s3 path where pretrained models are stored
  val pretrainedS3PathKey = "jsl.settings.pretrained.s3_path"

  // Configures cache folder where to cache pretrained models
  val pretrainedCacheFolder = "jsl.settings.pretrained.cache_folder"

  // Configures log folder where to store annotator logs using OutputHelper
  val annotatorLogFolder = "jsl.settings.annotator.log_folder"

  // Stores credentials for AWS S3 private models
  val awsCredentials = "jsl.settings.pretrained.credentials"
  val accessKeyId: String = awsCredentials + ".access_key_id"
  val secretAccessKey: String = awsCredentials + ".secret_access_key"
  val awsProfileName: String = awsCredentials + ".aws_profile_name"

  val s3SocketTimeout = "jsl.settings.pretrained.s3_socket_timeout"

  val storageTmpDir = "jsl.settings.storage.cluster_tmp_dir"

  val serializationMode = "jsl.settings.annotatorSerializationFormat"
  val useBroadcast = "jsl.settings.useBroadcastForFeatures"

  def getConfigValueOrElse(property: String, defaultValue: String): String = {
//    getConfigValue(property) match {
//      case Success(value) => value
//      case Failure(_) => defaultValue
//    }
    sparkSession.conf.get(property, defaultValue)
  }

//  def getConfigValue(property: String): Try[String] = Try {
//    sparkSession.conf.get(property)
//  }

  def getFileSystem: FileSystem = {
    FileSystem.get(sparkSession.sparkContext.hadoopConfiguration)
  }

  def getHadoopTmpDir: String = {
    sparkSession.sparkContext.hadoopConfiguration.get("hadoop.tmp.dir")
  }

}
