package com.johnsnowlabs.client.aws

import com.amazonaws.auth.{AWSCredentials, BasicSessionCredentials}
import com.johnsnowlabs.client.CredentialParams

class AWSTokenCredentials extends Credentials {

  override val next: Option[Credentials] = Some(new AWSBasicCredentials)

  override def buildCredentials(credentialParams: CredentialParams): Option[AWSCredentials] = {
    val credentialsValues = credentialParams.productIterator.toList.asInstanceOf[List[String]]
    val expectedNumberOfParams = credentialsValues.slice(0, 3).count(_.!=(""))
    if (expectedNumberOfParams == 3) {
      println("[INFO]: Connecting to AWS with AWS Token Credentials...")
      return Some(new BasicSessionCredentials(credentialParams.accessKeyId , credentialParams.secretAccessKey,
        credentialParams.sessionToken))
    }
    next.get.buildCredentials(credentialParams)
  }

}
