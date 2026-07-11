package clustering.benchmark.config

import org.json4s._

/** Thin typed accessor over a run's raw `JObject` params.
 *
 *  Centralises the "missing required" / "wrong type" handling so every factory
 *  reads its params the same way instead of re-deriving json4s plumbing. Keeps
 *  the json4s dependency out of the factories themselves. */
final class Params(json: JObject) {

  private implicit val formats: Formats = DefaultFormats

  def int(key: String): Int              = required(key).extract[Int]
  def intOpt(key: String, default: Int): Int = opt(key).map(_.extract[Int]).getOrElse(default)

  def double(key: String): Double        = required(key).extract[Double]
  def doubleOpt(key: String, default: Double): Double = opt(key).map(_.extract[Double]).getOrElse(default)

  def long(key: String): Long            = required(key).extract[Long]
  def longOpt(key: String, default: Long): Long = opt(key).map(_.extract[Long]).getOrElse(default)

  def string(key: String): String = required(key) match {
    case JString(s) => s
    case other      => throw new IllegalArgumentException(s"Param '$key' must be a string, got: $other")
  }

  /** Present, non-null value for `key`, else None. */
  def opt(key: String): Option[JValue] = (json \ key) match {
    case JNothing | JNull => None
    case v                => Some(v)
  }

  private def required(key: String): JValue =
    opt(key).getOrElse(throw new IllegalArgumentException(s"Missing required parameter: '$key'"))
}

object Params {
  def apply(json: JObject): Params = new Params(json)
}