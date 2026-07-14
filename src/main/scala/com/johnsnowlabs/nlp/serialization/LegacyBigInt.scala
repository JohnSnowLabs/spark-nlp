/*
 * Copyright 2017-2026 John Snow Labs
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
package com.johnsnowlabs.nlp.serialization

import java.math.BigInteger

/** Compatibility reader for scala.math.BigInt instances serialized with Scala 2.12.
  *
  * Scala 2.12 BigInt serialized a single `bigInteger` field. Scala 2.13 keeps the same class name
  * but changed the internal representation to `_bigInteger` plus `_long`. Replacing the stream
  * descriptor directly with Scala 2.13 BigInt can make ObjectInputStream consume the old field
  * layout incorrectly and fail with StreamCorruptedException while loading legacy pretrained
  * features such as MapFeature[String, BigInt].
  */
private[serialization] class LegacyBigInt extends Serializable {

  private var bigInteger: BigInteger = _

  private def readResolve(): AnyRef = scala.math.BigInt(bigInteger)
}
