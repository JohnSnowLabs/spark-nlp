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

package com.johnsnowlabs.storage

trait Database extends Serializable {
  val name: String
  override def toString: String = {
    name
  }
}
object Database {
  type Name = Database
  val EMBEDDINGS: Name = new Name {
    override val name: String = "EMBEDDINGS"
  }
  val TMVOCAB: Name = new Name {
    override val name: String = "TMVOCAB"
  }
  val TMEDGES: Name = new Name {
    override val name: String = "TMEDGES"
  }
  val TMNODES: Name = new Name {
    override val name: String = "TMNODES"
  }
  @deprecated
  val ENTITY_PATTERNS: Name = new Name {
    override val name: String = "ENTITY_PATTERNS"
  }
  val ENTITY_REGEX_PATTERNS: Name = new Name {
    override val name: String = "ENTITY_REGEX_PATTERNS"
  }
}
