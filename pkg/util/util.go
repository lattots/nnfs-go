package util

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
)

// ReadCSVToFloat64 reads a CSV file and returns a [][]float64.
// It skips the header row and handles potential errors during parsing.
func ReadCSVToFloat64(filename string) ([][]float64, error) {
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

	data := make([][]float64, 0)

	// Read the remaining rows.
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break // End of file
		}
		if err != nil {
			return nil, fmt.Errorf("failed to read CSV record: %w", err)
		}

		row := make([]float64, len(record))
		for i, val := range record {
			floatVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				return nil, fmt.Errorf("failed to parse float value '%s': %w", val, err)
			}
			row[i] = floatVal
		}
		data = append(data, row)
	}

	return data, nil
}

// ExtractOutput removes and returns the corresponding column index from the data matrix.
// It modifies the original matrix "data" in place.
func ExtractOutput(data *[][]float64, outputIndex int) ([]float64, error) {
	if data == nil || len(*data) == 0 {
		return nil, fmt.Errorf("data matrix is nil or empty")
	}

	columnCount := len((*data)[0])

	numRows := len(*data)
	if outputIndex < 0 || outputIndex >= len((*data)[0]) {
		return nil, fmt.Errorf("output index %d is out of range [0, %d]", outputIndex, len((*data)[0])-1)
	}

	output := make([]float64, numRows)
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
