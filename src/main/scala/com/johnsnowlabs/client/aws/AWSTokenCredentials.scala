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

import com.amazonaws.auth.{AWSCredentials, BasicSessionCredentials}

class AWSTokenCredentials extends Credentials {

  override val next: Option[Credentials] = Some(new AWSBasicCredentials)

  override def buildCredentials(credentialParams: CredentialParams): Option[AWSCredentials] = {
    val credentialsValues = credentialParams.productIterator.toList.asInstanceOf[List[String]]
    val expectedNumberOfParams = credentialsValues.slice(0, 3).count(_.!=(""))
    if (expectedNumberOfParams == 3) {
      logger.info("Connecting to AWS with AWS Token Credentials...")
      return Some(
        new BasicSessionCredentials(
          credentialParams.accessKeyId,
          credentialParams.secretAccessKey,
          credentialParams.sessionToken))
    }
    next.get.buildCredentials(credentialParams)
  }

}
