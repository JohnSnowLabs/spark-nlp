package com.johnsnowlabs.client.aws

import com.amazonaws.AmazonClientException
import com.amazonaws.auth.profile.ProfileCredentialsProvider
import com.amazonaws.auth.{AWSCredentials, BasicAWSCredentials, DefaultAWSCredentialsProviderChain}
import com.johnsnowlabs.client.CredentialParams
import com.johnsnowlabs.nlp.util.io.ResourceHelper

class AWSCredentialsProvider extends Credentials {

  override val next: Option[Credentials] = Some(new AWSAnonymousCredentials)

  override def buildCredentials(credentialParams: CredentialParams): Option[AWSCredentials] = {
    if (credentialParams.accessKeyId != "anonymous" && credentialParams.region != "") {
      try {
        //check if default profile name works if not try
        logger.info("Connecting to AWS with AWS Credentials Provider...")
        return Some(new ProfileCredentialsProvider("spark_nlp").getCredentials)
      } catch {
        case _: Exception =>
          try {
            return Some(new DefaultAWSCredentialsProviderChain().getCredentials)
          } catch {
            case _: AmazonClientException =>
              if (ResourceHelper.spark.sparkContext.hadoopConfiguration.get("fs.s3a.access.key") != null) {
                val key = ResourceHelper.spark.sparkContext.hadoopConfiguration.get("fs.s3a.access.key")
                val secret = ResourceHelper.spark.sparkContext.hadoopConfiguration.get("fs.s3a.secret.key")
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
