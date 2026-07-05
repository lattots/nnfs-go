package neuralnet

import (
	"fmt"

	"github.com/lattots/gonum/mat"
)

type layer struct {
	weights        *mat.Mat[float32]
	biases         *mat.Mat[float32]
	activationType activationFunctionType
}

func (l *layer) forward(input *mat.Mat[float32]) (*mat.Mat[float32], error) {
	layerInput, err := mat.Dot(input, l.weights)
	if err != nil {
		return nil, fmt.Errorf("error calculating layer input: %w", err)
	}
	err = addBias(layerInput, l.biases)
	if err != nil {
		return nil, fmt.Errorf("error adding bias: %w", err)
	}

	layerActivations, err := mat.Zeros[float32](layerInput.M, layerInput.N)
	if err != nil {
		return nil, fmt.Errorf("error creating layer activations matrix: %w", err)
	}

	activationFunction := getActivationFunc(l.activationType)

	layerActivations = mat.Map(layerInput, activationFunction)

	return layerActivations, nil
}

func newLayer(weights, biases *mat.Mat[float32], activationType activationFunctionType) *layer {
	if weights == nil || biases == nil {
		fmt.Println("weights or biases are nil")
		return nil
	}
	if weights.M == 0 || weights.N == 0 || biases.M == 0 || biases.N == 0 {
		fmt.Println("weights or biases have zero dimensions")
		return nil
	}
	if weights.N != biases.N {
		fmt.Println("N of weights is not equal to N of biases")
		return nil
	}
	if biases.M != 1 {
		fmt.Println("there is more than one set of biases")
		return nil
	}

	return &layer{
		weights:        weights,
		biases:         biases,
		activationType: activationType,
	}
}

func newRandomLayer(config *LayerConfig, seed int) *layer {
	weights, err := randomMatrix(config.InputSize, config.NeuronCount, config.InitType, seed)
	if err != nil {
		return nil
	}
	biases, err := randomMatrix(1, config.NeuronCount, config.InitType, seed)
	if err != nil {
		return nil
	}
	return newLayer(weights, biases, config.ActivationType)
}

type LayerConfig struct {
	InputSize      int
	NeuronCount    int
	ActivationType activationFunctionType
	InitType       InitializationType
}

// forwardPreActivation calculates the weighted sum plus bias (z = input * weights + bias).
func (l *layer) forwardPreActivation(input *mat.Mat[float32]) (*mat.Mat[float32], error) {
	// z = input * weights
	z, err := mat.Dot(input, l.weights)
	if err != nil {
		return nil, fmt.Errorf("error multiplying input by weights: %w", err)
	}

	// z = z + bias
	// The addBias function broadcasts the bias vector to each row of the matrix z.
	err = addBias(z, l.biases)
	if err != nil {
		return nil, fmt.Errorf("error adding bias: %w", err)
	}

	return z, nil
}

func (l *layer) forwardActivation(z *mat.Mat[float32]) (*mat.Mat[float32], error) {
	activationFunction := getActivationFunc(l.activationType)

	a := mat.Map(z, activationFunction)

	return a, nil
}
