package com.johnsnowlabs.util

import org.scalatest.FlatSpec
import com.johnsnowlabs.tags.{FastTest, SlowTest}

class VersionSpec extends FlatSpec {
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
