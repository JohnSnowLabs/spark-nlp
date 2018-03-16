package com.johnsnowlabs.util

import org.scalatest.FlatSpec

class VersionSpec extends FlatSpec {
  "Version.isInteger" should "correctly determine whether string is integer" in {
    assert(Version.isInteger("123"))
    assert(!Version.isInteger("23a"))
    assert(!Version.isInteger(""))
  }

  "Version.parse" should "correctly parse string" in {
    assert(Version.parse("1.2.3") == Version(1, 2, 3))
    assert(Version.parse("1.2.3b") == Version(1, 2))
    assert(Version.parse("1.2_3b") == Version(1))
  }

  "Version.isCompatible" should "correctly checks version compatability" in {
    assert(Version.isCompatible(Version(1, 2, 3), None))
    assert(!Version.isCompatible(Version(1, 2), Version(1, 2, 3)))
    assert(Version.isCompatible(Version(1, 2, 3), Version(1, 2, 3)))
    assert(Version.isCompatible(Version(1, 2, 3), Version(1, 2)))
  }

}
