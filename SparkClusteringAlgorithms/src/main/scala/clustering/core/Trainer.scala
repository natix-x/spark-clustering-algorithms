package clustering.core


trait Trainer[D, M <: Model] extends Serializable {
    def train(data: D): M
}
