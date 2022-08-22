package com.johnsnowlabs.storage

import org.scalatest.flatspec.AnyFlatSpec

class BytesKeyTest extends AnyFlatSpec {

  "BytesKeyTest" should "find values for a byte array key" in {

    val key1 = new BytesKey(Array(1, 2, 3))
    val key2 = new BytesKey(Array(50, 48, 51))
    val map = Map(key1 -> "value1", key2 -> "value2")

    val retrievedValue1 = map(key1)
    val retrievedValue2 = map(key2)
    val retrievedValue3 = map(new BytesKey(Array(1, 2, 3)))

    assert("value1" == retrievedValue1)
    assert("value2" == retrievedValue2)
    assert("value1" == retrievedValue3)
  }

}
