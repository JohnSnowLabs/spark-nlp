import sbt._

object Resolvers {
  val m2CentralRepo = "Maven Central"  at "https://repo1.maven.org/maven2/"

  val m2Resolvers = Seq(m2CentralRepo)
}