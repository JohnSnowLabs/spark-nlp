package com.johnsnowlabs.client.aws

import com.amazonaws.auth.AWSCredentials
import com.amazonaws.auth.profile.ProfileCredentialsProvider
import com.johnsnowlabs.client.CredentialParams

class AWSProfileCredentials extends Credentials {

  override val next: Option[Credentials] = Some(new AWSCredentialsProvider)

  override def buildCredentials(credentialParams: CredentialParams): Option[AWSCredentials] = {
    val credentialsValues = credentialParams.productIterator.toList.asInstanceOf[List[String]]
    val expectedNumberOfParams = credentialsValues.slice(3, 4).count(_.!=(""))
    if (expectedNumberOfParams == 1) {
      try {
        logger.info("Connecting to AWS with AWS Profile Credentials...")
        return Some(new ProfileCredentialsProvider(credentialParams.profile).getCredentials)
      } catch {
        case _: Exception =>
          logger.info(s"Profile ${credentialParams.profile} is not working. Attempting to use credentials provider")
          next.get.buildCredentials(credentialParams)
      }
    }
    next.get.buildCredentials(credentialParams)
  }

}
