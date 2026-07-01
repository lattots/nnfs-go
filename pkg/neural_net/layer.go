package neuralnet

import (
	"fmt"

	"github.com/lattots/gonum/pkg/matrix"
)

type layer struct {
	weights        *matrix.Matrix
	biases         *matrix.Matrix
	activationType activationFunctionType
}

func (l *layer) forward(input *matrix.Matrix) (*matrix.Matrix, error) {
	layerInput, err := input.Multiply(l.weights)
	if err != nil {
		return nil, fmt.Errorf("error calculating layer input: %w", err)
	}
	err = addBias(layerInput, l.biases.Data[0])
	if err != nil {
		return nil, fmt.Errorf("error adding bias: %w", err)
	}

	layerActivations, err := matrix.NewZeroMatrix(layerInput.M, layerInput.N)
	if err != nil {
		return nil, fmt.Errorf("error creating layer activations matrix: %w", err)
	}

	activationFunction := getActivationFunc(l.activationType)

	for i := range layerInput.Data {
		for j := range layerInput.Data[i] {
			layerActivations.Data[i][j] = activationFunction(layerInput.Data[i][j])
		}
	}

	return layerActivations, nil
}

func newLayer(weights, biases *matrix.Matrix, activationType activationFunctionType) *layer {
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
func (l *layer) forwardPreActivation(input *matrix.Matrix) (*matrix.Matrix, error) {
	// z = input * weights
	z, err := input.Multiply(l.weights)
	if err != nil {
		return nil, fmt.Errorf("error multiplying input by weights: %w", err)
	}

	// z = z + bias
	// The addBias function broadcasts the bias vector to each row of the matrix z.
	err = addBias(z, l.biases.Data[0])
	if err != nil {
		return nil, fmt.Errorf("error adding bias: %w", err)
	}

	return z, nil
}

func (l *layer) forwardActivation(z *matrix.Matrix) (*matrix.Matrix, error) {
	activationFunction := getActivationFunc(l.activationType)

	// The Map function is a common utility in matrix libraries.
	// It applies a given function to every element of the matrix, returning a new matrix.
	// If your library doesn't have Map, you can use your original nested for-loop implementation.
	a, err := z.Map(activationFunction)
	if err != nil {
		return nil, fmt.Errorf("error applying activation function: %w", err)
	}

	return a, nil
}
