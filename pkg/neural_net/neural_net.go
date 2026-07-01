package neuralnet

import (
	"fmt"
	"math"

	"github.com/lattots/gonum/pkg/matrix"

	"github.com/lattots/nnfs_go/pkg/mathematic"
)

type Net struct {
	config Config
	layers []*layer
}

func NewNet(config Config) *Net {
	return &Net{config: config}
}

type Config struct {
	NumEpochs    int
	LearningRate float64
	LossFunction LossFunctionType

	LayerConfigs []*LayerConfig
}

func (n *Net) InitRandom(seed int) error {
	n.layers = make([]*layer, len(n.config.LayerConfigs))

	for i := 0; i < len(n.config.LayerConfigs); i++ {
		if n.config.LayerConfigs[i].InputSize == 0 {

			n.config.LayerConfigs[i].InputSize = n.config.LayerConfigs[i-1].NeuronCount
		}
		n.layers[i] = newRandomLayer(n.config.LayerConfigs[i], seed)
	}

	return nil
}

func (n *Net) Train(x, y *matrix.Matrix) error {
	for epoch := range n.config.NumEpochs {
		// Get both pre-activations (zs) and activations (activations)
		zs, activations, err := n.forwardPassAndGetOutputs(x)
		if err != nil {
			return fmt.Errorf("failed to forward pass: %w", err)
		}

		// The final output before softmax is the last item in activations
		finalOutput := activations[len(activations)-1]

		// Apply Softmax
		softmaxOutput, err := mathematic.SoftmaxMatrix(finalOutput)
		if err != nil {
			return fmt.Errorf("failed to softmax last layer activations: %w", err)
		}

		// For debugging, print softmax output
		for i := range softmaxOutput.Data[0] {
			fmt.Printf("%.3f ", softmaxOutput.Data[0][i])
		}
		fmt.Println()

		loss, err := getLossFunction(n.config.LossFunction)(softmaxOutput, y)
		if err != nil {
			return fmt.Errorf("error calculating loss: %w", err)
		}
		fmt.Printf("Epoch: %d, Loss: %.4f\n", epoch, loss)

		// Pass zs and the updated activations list to the backward pass
		if err = n.backwardPass(y, zs, activations, softmaxOutput); err != nil {
			return fmt.Errorf("backward pass failed: %w", err)
		}
	}
	return nil
}

func (n *Net) forwardPassAndGetOutputs(x *matrix.Matrix) (zs, activations []*matrix.Matrix, err error) {
	// Pre-pend the input matrix to a list of activations for easier backprop later
	activations = []*matrix.Matrix{x}
	currentInput := x

	for i, layer := range n.layers {
		// Calculate pre-activation z = (input * weights) + biases
		z, err := layer.forwardPreActivation(currentInput) // You'll need to create this helper method in your 'layer' struct
		if err != nil {
			return nil, nil, fmt.Errorf("error in pre-activation for layer %d: %w", i, err)
		}
		zs = append(zs, z)

		// Calculate activation a = g(z)
		a, err := layer.forwardActivation(z) // You'll need to create this helper method
		if err != nil {
			return nil, nil, fmt.Errorf("error in activation for layer %d: %w", i, err)
		}
		activations = append(activations, a)
		currentInput = a // The output of this layer is the input to the next
	}

	return zs, activations, nil
}

func (n *Net) backwardPass(y *matrix.Matrix, zs, activations []*matrix.Matrix, softmaxOutput *matrix.Matrix) error {
	// For Softmax + CCE, the initial delta is simply (prediction - actual)
	finalDelta := softmaxOutput.Subtract(y)

	deltas := make([]*matrix.Matrix, len(n.layers))
	deltas[len(deltas)-1] = finalDelta

	// Calculate deltas for hidden layers (from right to left)
	for i := len(n.layers) - 2; i >= 0; i-- {
		errorPropagated, err := deltas[i+1].Multiply(n.layers[i+1].weights.Transpose())
		if err != nil {
			return err
		}

		activationDerivative := getActivationDerivative(n.layers[i].activationType)
		// *** FIX: Use the pre-activation values (zs) to calculate the derivative ***
		slope, err := zs[i].Map(activationDerivative)
		if err != nil {
			return err
		}

		deltas[i], err = errorPropagated.MultiplyElements(slope)
		if err != nil {
			return err
		}
	}

	// Update weights and biases
	batchSize := float64(activations[0].M) // activations[0] is the input x
	for i := len(n.layers) - 1; i >= 0; i-- {
		// The input to layer 'i' is the activation from the previous layer, 'i'.
		// activations = [x, a_layer0, a_layer1, ...]
		prevLayerActivations := activations[i]

		weightGradient, err := prevLayerActivations.Transpose().Multiply(deltas[i])
		if err != nil {
			return err
		}
		weightGradient.Scale(n.config.LearningRate / batchSize)
		n.layers[i].weights = n.layers[i].weights.Subtract(weightGradient)
		if err != nil {
			return err
		}

		biasGradient := deltas[i].SumRows()
		biasGradient.Scale(n.config.LearningRate / batchSize)
		n.layers[i].biases = n.layers[i].biases.Subtract(biasGradient)
		if err != nil {
			return err
		}
	}

	return nil
}

// func (n *Net) Train(x, y *matrix.Matrix) error {
// 	for epoch := range n.config.NumEpochs {
// 		activations, err := n.forwardPassGetActivations(x)
// 		if err != nil {
// 			return fmt.Errorf("failed to forward pass: %w", err)
// 		}
//
// 		activations[len(activations)-1], err = mathematic.SoftmaxMatrix(activations[len(activations)-1])
// 		if err != nil {
// 			return fmt.Errorf("failed to softmax last layer activations: %w", err)
// 		}
//
// 		for i := range activations[len(activations)-1].Data[0] {
// 			fmt.Printf("%.3f ", activations[len(activations)-1].Data[0][i])
// 		}
// 		fmt.Println()
//
// 		loss, err := getLossFunction(n.config.LossFunction)(activations[len(activations)-1], y)
// 		if err != nil {
// 			return fmt.Errorf("error calculating loss: %w", err)
// 		}
// 		fmt.Printf("Epoch: %d, Loss: %.4f\n", epoch, loss)
//
// 		if err = n.backwardPass(x, y, activations); err != nil {
// 			return fmt.Errorf("backward pass failed: %w", err)
// 		}
// 	}
// 	return nil
// }
//
// func (n *Net) backwardPass(x, y *matrix.Matrix, activations []*matrix.Matrix) error {
// 	// For Softmax + CCE, the initial delta is simply (prediction - actual)
// 	// This combines the derivative of the loss and the derivative of softmax.
// 	finalDelta := activations[len(activations)-1].Subtract(y)
//
// 	deltas := make([]*matrix.Matrix, len(n.layers))
// 	deltas[len(deltas)-1] = finalDelta
//
// 	// Calculate deltas for hidden layers (from right to left)
// 	for i := len(n.layers) - 2; i >= 0; i-- {
// 		// Error propagated from the next layer
// 		errorPropagated, err := deltas[i+1].Multiply(n.layers[i+1].weights.Transpose())
// 		if err != nil {
// 			return err
// 		}
//
// 		// Get derivative of the current layer's activation function
// 		activationDerivative := getActivationDerivative(n.layers[i].activationType)
// 		slope, err := activations[i].Map(activationDerivative) // Assuming a Map function applies a func to each element
// 		if err != nil {
// 			return err
// 		}
//
// 		// Calculate the delta for the current layer
// 		deltas[i], err = errorPropagated.MultiplyElements(slope)
// 		if err != nil {
// 			return err
// 		}
// 	}
//
// 	// Update weights and biases (from right to left)
// 	batchSize := float64(x.M)
// 	for i := len(n.layers) - 1; i >= 0; i-- {
// 		var prevLayerActivations *matrix.Matrix
// 		if i == 0 {
// 			prevLayerActivations = x
// 		} else {
// 			prevLayerActivations = activations[i-1]
// 		}
//
// 		// Calculate weight gradient and apply update
// 		weightGradient, err := prevLayerActivations.Transpose().Multiply(deltas[i])
// 		if err != nil {
// 			return err
// 		}
// 		weightGradient.Scale(n.config.LearningRate / batchSize) // Average the gradient
// 		n.layers[i].weights = n.layers[i].weights.Subtract(weightGradient)
//
// 		// Calculate bias gradient and apply update
// 		biasGradient := deltas[i].SumRows()                   // Assumes SumRows sums columns down to a row vector
// 		biasGradient.Scale(n.config.LearningRate / batchSize) // Average the gradient
// 		n.layers[i].biases = n.layers[i].biases.Subtract(biasGradient)
// 	}
//
// 	return nil
// }

func (n *Net) Predict(x *matrix.Matrix) (*matrix.Matrix, error) {
	res, err := n.forwardPass(x)
	res, err = mathematic.SoftmaxMatrix(res)
	if err != nil {
		return nil, fmt.Errorf("error applying softmax: %w", err)
	}

	return res, nil
}

// Test returns an accuracy score of neural nets capability to predict outcomes in testingMatrix
func (n *Net) Test(testInput [][]float64, expectedOutput []float64) (float64, error) {
	var correct int
	for i := range testInput {
		inputMatrix, err := matrix.NewMatrix([][]float64{testInput[i]})
		if err != nil {
			return 0, fmt.Errorf("error creating input matrix: %w", err)
		}

		outputMatrix, err := n.Predict(inputMatrix)
		if err != nil {
			return 0, fmt.Errorf("error making prediction: %w", err)
		}

		expected := int(math.Round(expectedOutput[i]))

		largest := 0.0
		idx := -1
		for j := range outputMatrix.Data[0] {
			if outputMatrix.Data[0][j] > largest {
				idx = j
				largest = outputMatrix.Data[0][j]
			}
		}

		if idx == expected {
			// fmt.Printf("Correctly predicted %d as %d\n", expected, idx)
			correct++
		}
	}

	score := float64(correct) / float64(len(expectedOutput))
	return score, nil
}

func (n *Net) forwardPass(x *matrix.Matrix) (*matrix.Matrix, error) {
	res, err := n.layers[0].forward(x)
	if err != nil {
		return nil, err
	}
	for i := 1; i < len(n.layers); i++ { // loop through all
		res, err = n.layers[i].forward(res)
		if err != nil {
			return nil, fmt.Errorf("error forward passing through layer: %w", err)
		}
	}

	return res, err
}

func (n *Net) forwardPassGetActivations(x *matrix.Matrix) ([]*matrix.Matrix, error) {
	activations := make([]*matrix.Matrix, len(n.layers))
	var err error
	activations[0], err = n.layers[0].forward(x)
	if err != nil {
		return nil, err
	}

	for i := 1; i < len(n.layers); i++ {
		activations[i], err = n.layers[i].forward(activations[i-1])
		if err != nil {
			return nil, fmt.Errorf("error forward passing on layer %d: %w", i, err)
		}
	}

	return activations, nil
}
