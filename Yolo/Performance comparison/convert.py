import pandas as pd

def convert_time_column(input_file, output_file):
    """
    Converts the 'Median Time per Frame (s)' column in a CSV file from seconds to milliseconds.
    
    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the updated CSV file.
    """
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(input_file)

        # Check if the column exists
        if 'Median Time per Frame (s)' not in df.columns:
            print("Column 'Median Time per Frame (s)' not found in the CSV file.")
            return

        # Convert the column to milliseconds
        df['Median Time per Frame (ms)'] = df['Median Time per Frame (s)'] * 1000

        # Save the updated DataFrame to a new CSV file
        df.to_csv(output_file, index=False)
        print(f"Updated file saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
input_csv = "model_comparison_results.csv"  # Replace with your input file path
output_csv = "model_comparison_results.csv"  # Desired output file path
convert_time_column(input_csv, output_csv)
