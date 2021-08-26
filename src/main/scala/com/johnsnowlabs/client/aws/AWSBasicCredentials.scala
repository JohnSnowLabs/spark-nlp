package com.johnsnowlabs.client.aws

import com.amazonaws.auth.{AWSCredentials, BasicAWSCredentials}
import com.johnsnowlabs.client.CredentialParams

class AWSBasicCredentials extends Credentials {

  override val next: Option[Credentials] = Some(new AWSProfileCredentials)

  override def buildCredentials(credentialParams: CredentialParams): Option[AWSCredentials] = {
    val credentialsValues = credentialParams.productIterator.toList.asInstanceOf[List[String]]
    val expectedNumberOfParams = credentialsValues.slice(0, 2).count(_.!=(""))
    if (expectedNumberOfParams == 2) {
      logger.info("Connecting to AWS with AWS Basic Credentials...")
      return Some(new BasicAWSCredentials(credentialParams.accessKeyId, credentialParams.secretAccessKey))
    }
    next.get.buildCredentials(credentialParams)
  }

}
