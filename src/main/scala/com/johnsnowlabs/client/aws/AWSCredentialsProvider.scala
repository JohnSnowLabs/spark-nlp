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

package com.johnsnowlabs.client.aws

import com.amazonaws.AmazonClientException
import com.amazonaws.auth.profile.ProfileCredentialsProvider
import com.amazonaws.auth.{
  AWSCredentials,
  BasicAWSCredentials,
  DefaultAWSCredentialsProviderChain
}
import com.johnsnowlabs.nlp.util.io.ResourceHelper

class AWSCredentialsProvider extends Credentials {

  override val next: Option[Credentials] = Some(new AWSAnonymousCredentials)

  override def buildCredentials(credentialParams: CredentialParams): Option[AWSCredentials] = {
    if (credentialParams.accessKeyId != "anonymous" && credentialParams.region != "") {
      try {
        // check if default profile name works if not try
        logger.info("Connecting to AWS with AWS Credentials Provider...")
        return Some(new ProfileCredentialsProvider("spark_nlp").getCredentials)
      } catch {
        case _: Exception =>
          try {
            return Some(new DefaultAWSCredentialsProviderChain().getCredentials)
          } catch {
            case _: AmazonClientException =>
              if (ResourceHelper.spark.sparkContext.hadoopConfiguration.get(
                  "fs.s3a.access.key") != null) {
                val key =
                  ResourceHelper.spark.sparkContext.hadoopConfiguration.get("fs.s3a.access.key")
                val secret =
                  ResourceHelper.spark.sparkContext.hadoopConfiguration.get("fs.s3a.secret.key")
                return Some(new BasicAWSCredentials(key, secret))
              } else {
                next.get.buildCredentials(credentialParams)
              }
            case e: Exception => throw e
          }
      }
    }
    next.get.buildCredentials(credentialParams)
  }

}
