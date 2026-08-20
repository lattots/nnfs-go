package util

import (
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"

	"github.com/lattots/gonum/mat"
)

// ReadCSVToFloat32 reads a CSV file and returns a [][]float32.
// It skips the header row and handles potential errors during parsing.
func ReadCSVToFloat32(filename string) ([][]float32, error) {
	// Open the CSV file.
	f, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer f.Close()

	// Create a new CSV reader.
	reader := csv.NewReader(f)

	// Skip the header row.
	_, err = reader.Read()
	if err != nil && err != io.EOF { // io.EOF is ok here, it just means empty file
		return nil, fmt.Errorf("failed to read header row: %w", err)
	}

	data := make([][]float32, 0)

	// Read the remaining rows.
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break // End of file
		}
		if err != nil {
			return nil, fmt.Errorf("failed to read CSV record: %w", err)
		}

		row := make([]float32, len(record))
		for i, val := range record {
			floatVal, err := strconv.ParseFloat(val, 32)
			if err != nil {
				return nil, fmt.Errorf("failed to parse float value '%s': %w", val, err)
			}
			row[i] = float32(floatVal)
		}
		data = append(data, row)
	}

	return data, nil
}

// ExtractOutput removes and returns the corresponding column index from the data matrix.
// It modifies the original matrix "data" in place.
func ExtractOutput(data *[][]float32, outputIndex int) ([]float32, error) {
	if data == nil || len(*data) == 0 {
		return nil, fmt.Errorf("data matrix is nil or empty")
	}

	columnCount := len((*data)[0])

	numRows := len(*data)
	if outputIndex < 0 || outputIndex >= len((*data)[0]) {
		return nil, fmt.Errorf("output index %d is out of range [0, %d]", outputIndex, len((*data)[0])-1)
	}

	output := make([]float32, numRows)
	for i := range *data {
		if len((*data)[i]) != columnCount {
			return nil, fmt.Errorf("rows have inconsistent lengths")
		}
		output[i] = (*data)[i][outputIndex]

		// Remove the element from the row by creating a new slice
		(*data)[i] = append((*data)[i][:outputIndex], (*data)[i][outputIndex+1:]...)
	}

	return output, nil
}

func ToOneHot(labels []float32, numClasses int) (*mat.Mat[float32], error) {
	if len(labels) == 0 {
		return nil, fmt.Errorf("labels slice cannot be empty")
	}
	if numClasses <= 0 {
		return nil, fmt.Errorf("numClasses must be positive, got %d", numClasses)
	}

	labelMatrix, err := mat.Zeros[float32](len(labels), numClasses)
	if err != nil {
		return nil, fmt.Errorf("failed to allocate label matrix: %w", err)
	}

	for i, v := range labels {
		classIdx := int(v)

		if float32(classIdx) != v {
			return nil, fmt.Errorf("label at index %d (%f) is not a whole integer", i, v)
		}
		if classIdx < 0 || classIdx >= numClasses {
			return nil, fmt.Errorf("label at index %d (%d) out of bounds [0, %d)", i, classIdx, numClasses)
		}

		labelMatrix.Data[i*numClasses+classIdx] = 1.0
	}

	return labelMatrix, nil
}

// NormalizeMatrix scales all values in the matrix to range from 0-1 while preserving their relative sizes
func NormalizeMatrix(m *mat.Mat[float32]) (*mat.Mat[float32], error) {
	largestAbs := max(float32(math.Abs(float64(mat.Min(m)))), mat.Max(m))

	res := mat.Scale(m, 1/largestAbs)

	return res, nil
}
