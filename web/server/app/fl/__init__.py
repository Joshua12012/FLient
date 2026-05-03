"""
Federated Learning core package.

Modules:
- data: EMNIST byclass loader + non-IID Dirichlet partitioning + tf.data parallel pipeline
- model: Tiered Keras CNNs (lite / standard / full) with optional MirroredStrategy
- strategy: AdaptiveFedAvg Flower strategy that records communication-round metrics
- metrics: RoundMetric dataclass + JSON store + chart rendering
- inference: Save Keras .weights.h5 + float16 TFLite per tier each round
"""
