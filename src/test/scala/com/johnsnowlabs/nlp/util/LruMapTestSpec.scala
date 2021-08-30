/*
 * Copyright 2017-2021 John Snow Labs
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

package com.johnsnowlabs.nlp.util

import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

class LruMapTestSpec extends AnyFlatSpec {

  "A LruMap" should "Deque and enqueue correctly" taggedAs FastTest in {

    val lru = new LruMap[String, Double](5)

    val iv = Seq(
      ("a", 234.5),
      ("b", 345.6),
      ("c", 456.7),
      ("d", 567.8),
      ("e", 678.9)
    )

    iv.foreach{case (i, v) => lru.getOrElseUpdate(i, v)}

    assert(lru.getSize == 5, "Wrong initial size")

    lru.getOrElseUpdate("j", 25.1)
    lru.get("b").getOrElse(lru.getOrElseUpdate("b", 345.6))
    lru.get("a").getOrElse(lru.getOrElseUpdate("a", 234.5))
    lru.get("a").getOrElse(lru.getOrElseUpdate("a", 234.5))
    lru.get("d").getOrElse(lru.getOrElseUpdate("d", 567.8))
    lru.getOrElseUpdate("r", 22.7)
    lru.get("e").getOrElse(lru.getOrElseUpdate("e", 678.9))
    lru.get("e").getOrElse(lru.getOrElseUpdate("e", 678.9))
    lru.get("b").getOrElse(lru.getOrElseUpdate("b", 345.6))

    assert(lru.getSize == 5, "Size not as expected after getting and updated cache")

    lru.getOrElseUpdate("new", 1.11)
    lru.getOrElseUpdate("even newer", 4.13)

    assert(lru.getSize == 5, "Size not as expected after adding 2 new values")
    assert(lru.get("new").isDefined, "Recently added key is not in the LRU!")

    assert(lru.get("c").isEmpty, "value 'c' should not be in LRU since it was never used")
    assert(lru.get("d").isEmpty, "value 'd' should not be in LRU since it was rarely queried (once)")

  }

}
