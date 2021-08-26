package com.johnsnowlabs.client.aws

import com.amazonaws.auth.AWSCredentials
import com.johnsnowlabs.client.CredentialParams
import org.slf4j.{Logger, LoggerFactory}

trait Credentials {

  protected val logger: Logger = LoggerFactory.getLogger("Credentials")

  val next: Option[Credentials] = None

  def buildCredentials(credentialParams: CredentialParams): Option[AWSCredentials]

}
