val scoverageVersion = "1.6.1"

resolvers += "bintray-spark-packages" at "https://dl.bintray.com/spark-packages/maven/"

addSbtPlugin("org.foundweekends" % "sbt-bintray" % "0.6.1")

addSbtPlugin("org.xerial.sbt" % "sbt-sonatype" % "3.9.5")

addSbtPlugin("com.jsuereth" % "sbt-pgp" % "2.0.1")

/** scoverage */
addSbtPlugin("org.scoverage" % "sbt-scoverage" % scoverageVersion)

useCoursier := false