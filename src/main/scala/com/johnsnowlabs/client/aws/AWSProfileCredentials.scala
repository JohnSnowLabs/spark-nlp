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

import com.amazonaws.auth.AWSCredentials
import com.amazonaws.auth.profile.ProfileCredentialsProvider

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
          logger.info(
            s"Profile ${credentialParams.profile} is not working. Attempting to use credentials provider")
          next.get.buildCredentials(credentialParams)
      }
    }
    next.get.buildCredentials(credentialParams)
  }

}
