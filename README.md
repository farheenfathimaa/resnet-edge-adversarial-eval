# resnet-edge-adversarial-eval
Edge-optimized deployment and adversarial robustness evaluation of a fine-tuned ResNet-50 classifier. Quantizes the model for CPU-only inference (simulating Raspberry Pi/Jetson-class hardware), serves predictions via a lightweight local FastAPI endpoint, and benchmarks accuracy degradation under FGSM/PGD adversarial perturbation.
