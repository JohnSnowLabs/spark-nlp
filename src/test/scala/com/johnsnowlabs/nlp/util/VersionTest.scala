package com.johnsnowlabs.nlp.util

import com.johnsnowlabs.util.Version
import org.scalatest.flatspec.AnyFlatSpec

class VersionTest extends AnyFlatSpec {

  "Version" should "cast to float version of 1 digit" in {

    val actualVersion1 = Version(1).toFloat
    val actualVersion15 = Version(15).toFloat

    assert(actualVersion1 == 1f)
    assert(actualVersion15 == 15f)

  }

  it should "cast to float version of 2 digits" in {
    val actualVersion1_2 = Version(List(1, 2)).toFloat
    val actualVersion2_7 = Version(List(2, 7)).toFloat

    assert(actualVersion1_2 == 1.2f)
    assert(actualVersion2_7 == 2.7f)
  }

  it should "cast to float version of 3 digits" in {
    val actualVersion1_2_5 = Version(List(1, 2, 5)).toFloat
    val actualVersion3_2_0 = Version(List(3, 2, 0)).toFloat
    val actualVersion2_0_6 = Version(List(2, 0, 6)).toFloat

    assert(actualVersion1_2_5 == 1.25f)
    assert(actualVersion3_2_0 == 3.2f)
    assert(actualVersion2_0_6 == 2.06f)
  }

  it should "raise error when casting to float version > 3 digits" in {
    assertThrows[UnsupportedOperationException] {
      Version(List(3, 0, 2, 5)).toFloat
    }
  }

}
