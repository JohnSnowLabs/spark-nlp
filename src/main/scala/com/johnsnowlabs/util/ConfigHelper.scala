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

package com.johnsnowlabs.util

import com.johnsnowlabs.nlp.util.io.ResourceHelper

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
  val sessionToken: String = awsCredentials + ".session_token"
  val awsProfileName: String = awsCredentials + ".aws_profile_name"
  val awsRegion: String = awsCredentials + ".aws.region"
  val s3SocketTimeout = "spark.jsl.settings.pretrained.s3_socket_timeout"

  // Stores info for AWS S3 logging output when training models
  val awsExternalCredentials = "spark.jsl.settings.aws.credentials"
  val awsExternalAccessKeyId: String = awsExternalCredentials + ".access_key_id"
  val awsExternalSecretAccessKey: String = awsExternalCredentials + ".secret_access_key"
  val awsExternalSessionToken: String = awsExternalCredentials + ".session_token"
  val awsExternalProfileName: String = awsExternalCredentials + ".aws_profile_name"
  val awsExternalS3BucketKey = "spark.jsl.settings.aws.s3_bucket"
  val awsExternalRegion = "spark.jsl.settings.aws.region"

  val storageTmpDir = "spark.jsl.settings.storage.cluster_tmp_dir"

  val serializationMode = "spark.jsl.settings.annotatorSerializationFormat"
  val useBroadcast = "spark.jsl.settings.useBroadcastForFeatures"

  def getConfigValueOrElse(property: String, defaultValue: String): String = {
    sparkSession.conf.get(property, defaultValue)
  }

  def getHadoopTmpDir: String = {
    sparkSession.sparkContext.hadoopConfiguration.get("hadoop.tmp.dir")
  }

}
