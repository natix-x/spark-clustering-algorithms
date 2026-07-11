name := "spark-clustering"
version := "0.1"

scalaVersion := "2.12.15"

val sparkVersion = "3.3.2"

resolvers += "SparkPackages" at "https://repos.spark-packages.org/"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core"   % sparkVersion % "provided",
  "org.apache.spark" %% "spark-sql"    % sparkVersion % "provided",
  "org.apache.spark" %% "spark-mllib"  % sparkVersion % "provided",
  "graphframes" % "graphframes" % "0.8.2-spark3.2-s_2.12",
  "org.apache.commons" % "commons-math3" % "3.6.1",
  "org.log4s" %% "log4s" % "1.10.0",
  "org.apache.logging.log4j" % "log4j-slf4j2-impl" % "2.20.0" % "provided",
  "com.networknt" % "json-schema-validator" % "1.0.76" % Test,
  "org.scalatest" %% "scalatest"            % "3.2.17" % Test
)

// Spark (and its json4s) are `provided`, so they are off the test classpath by
// default. The integration test needs the full Spark runtime — pull compile-scope
// (incl. provided) deps onto the test classpath. But that mixes two slf4j stacks:
// Spark's (slf4j-api 1.7 + log4j-slf4j-impl) and ours (slf4j-api 2.x +
// log4j-slf4j2-impl). Both ship org.apache.logging.slf4j.Log4jLoggerFactory, so
// they collide (NoSuchMethodError Log4jLoggerFactory.<init>). Keep only Spark's
// coherent 1.7 stack for tests; drop our slf4j-2 binding and its slf4j-api 2.x.
Test / dependencyClasspath := {
  val merged = ((Test / dependencyClasspath).value ++ (Compile / dependencyClasspath).value).distinct
  merged.filterNot { e =>
    val n = e.data.getName
    n.contains("log4j-slf4j2-impl") || n.matches("""slf4j-api-2\..*\.jar""")
  }
}
Test / fork := true

// Keep Jackson pinned to Spark 3.3.2's version everywhere; networknt (test-only)
// would otherwise drag a newer Jackson that breaks jackson-module-scala 2.13.4.
dependencyOverrides ++= Seq(
  "com.fasterxml.jackson.core" % "jackson-databind"    % "2.13.4",
  "com.fasterxml.jackson.core" % "jackson-core"        % "2.13.4",
  "com.fasterxml.jackson.core" % "jackson-annotations" % "2.13.4"
)

assembly / mainClass := Some("clustering.benchmark.BenchmarkRunner")
assembly / assemblyJarName := "spark-clustering-benchmark.jar"

assembly / assemblyMergeStrategy := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x if x.endsWith("module-info.class") => MergeStrategy.discard
  case x => MergeStrategy.first
}

exportJars := true