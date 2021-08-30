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

package com.johnsnowlabs.util

import org.scalatest.flatspec.AnyFlatSpec
import com.johnsnowlabs.tags.{FastTest, SlowTest}

class VersionSpec extends AnyFlatSpec {
  "Version.isInteger" should "correctly determine whether string is integer" taggedAs FastTest in {
    assert(Version.isInteger("123"))
    assert(!Version.isInteger("23a"))
    assert(!Version.isInteger(""))
  }

  "Version.parse" should "correctly parse string" taggedAs FastTest in {
    assert(Version.parse("1.2.3") == Version(1, 2, 3))
    assert(Version.parse("1.2.3b") == Version(1, 2))
    assert(Version.parse("1.2_3b") == Version(1))
  }

  "Version.isCompatible" should "correctly checks version compatability" taggedAs FastTest in {
    assert(Version.isCompatible(Version(1, 2, 3), None))
    assert(!Version.isCompatible(Version(1, 2), Version(1, 2, 3)))
    assert(Version.isCompatible(Version(1, 2, 3), Version(1, 2, 3)))
    assert(Version.isCompatible(Version(1, 2, 3), Version(1, 2)))
    assert(Version.isCompatible(Version(1,2,3), Version(1,2,2)))
    assert(Version.isCompatible(Version(1,2,3,1), Version(1,2,2,8)))
    assert(!Version.isCompatible(Version(1,2,1,5), Version(1,2,2,2)))
    assert(Version.isCompatible(Version(2,3,4), Version(2,3)))
    assert(Version.isCompatible(Version(2,4,4), Version(2,3)))
    assert(Version.isCompatible(Version(2,3,4), Version(2)))
    assert(Version.isCompatible(Version(2,4,4), Version(2)))
    assert(Version.isCompatible(Version(2,4,4), Version(2,3,3)))
    assert(!Version.isCompatible(Version(3), Version(2,3)))
  }

}
