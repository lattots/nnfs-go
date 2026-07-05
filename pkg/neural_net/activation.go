package neuralnet

import (
	"github.com/lattots/gonum/mat"

	"github.com/lattots/nnfs_go/pkg/mathematic"
)

// activationFunc type for activation functions
type activationFunc func(float32) float32

// activationDerivative type for activation function derivatives
type activationDerivative func(float32) float32

// activationType type tells the type of activation function to use
type activationFunctionType int

// Different types of activation function
const (
	Sigmoid activationFunctionType = iota
	ReLU
	Softmax // Add Softmax here
)

func getActivationFunc(actType activationFunctionType) activationFunc {
	switch actType {
	case Sigmoid:
		return mathematic.Sigmoid
	case ReLU:
		return mathematic.ReLU
	case Softmax: // Handle Softmax
		return func(x float32) float32 { // Adapt to the required type
			// Softmax operates on vector, not single float.
			// We need to handle it in Predict function.
			return x
		}
	default:
		return mathematic.ReLU
	}
}

func getActivationDerivative(actType activationFunctionType) activationDerivative {
	switch actType {
	case Sigmoid:
		return mathematic.SigmoidPrime
	case ReLU:
		return mathematic.ReLUPrime
	default: // Default to ReLU derivative
		return mathematic.ReLUPrime
	}
}

// lossFunc for loss functions
type lossFunc func(predicted, target *mat.Mat[float32]) (float32, error)

// lossDerivative type for loss function derivatives
type lossDerivative func(predicted, target *mat.Mat[float32]) (*mat.Mat[float32], error)

// LossFunctionType type tells the type of loss function to use
type LossFunctionType int

// Different types of loss functions
const (
	CrossEntropy LossFunctionType = iota
	BinaryCrossEntropy
)

// getLossFunction returns the loss function corresponding to LossFunctionType
func getLossFunction(lossType LossFunctionType) func(*mat.Mat[float32], *mat.Mat[float32]) (float32, error) {
	switch lossType {
	case CrossEntropy:
		return mathematic.CrossEntropy
	case BinaryCrossEntropy:
		return mathematic.BinaryCrossEntropy
	default:
		return mathematic.CrossEntropy // Default to Cross Entropy
	}
}

// getLossDerivative returns the loss function derivative corresponding to LossFunctionType
func getLossDerivative(lossType LossFunctionType) func(*mat.Mat[float32], *mat.Mat[float32]) (*mat.Mat[float32], error) {
	switch lossType {
	case CrossEntropy:
		return mathematic.CrossEntropyPrime
	case BinaryCrossEntropy:
		return mathematic.BinaryCrossEntropyPrime
	default:
		return mathematic.CrossEntropyPrime // Default to Cross Entropy Prime
	}
}
