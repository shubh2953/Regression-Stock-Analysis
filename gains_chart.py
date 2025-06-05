# import sys
# import csv
# import matplotlib.pyplot as plt

# def create_gains_chart(input_file, output_file):
#     # Lists to store the specific columns we need
#     row_weights = []
#     model_values = []
#     random_line = []
    
#     # Read CSV file
#     with open(input_file, 'r') as file:
#         csv_reader = csv.DictReader(file)
#         for row in csv_reader:
#             # Skip the final AUC score row which contains summary statistics
#             if 'FINAL AUC SCORE' not in row['Predicted']:
#                 try:
#                     # Extract only the three columns we need
#                     row_weights.append(float(row['Row_Weight']))
#                     model_values.append(float(row['Model']))  
#                     random_line.append(float(row['Random_Line']))
#                 except (ValueError, KeyError) as e:
#                     print(f"Error reading data: {e}")
#                     sys.exit(1)

#     # Create the plot
#     plt.figure(figsize=(10, 6))
    
#     # Plot the actual lines using the three columns
#     plt.plot(row_weights, model_values, 'b-', label='Model', linewidth=2)
#     plt.plot(row_weights, random_line, 'r--', label='Random', linewidth=2)
    
#     # Customize the plot
#     plt.xlabel('Population Percentage (Row Weight)')
#     plt.ylabel('Cumulative Gain')
#     plt.title('Gains Chart')
#     plt.grid(True, linestyle='--', alpha=0.7)
    
#     # Format axes to show percentages
#     plt.gca().set_xticklabels([f'{x:.0%}' for x in plt.gca().get_xticks()])
#     plt.gca().set_yticklabels([f'{y:.0%}' for y in plt.gca().get_yticks()])
    
#     # Add legend
#     plt.legend(loc='lower right')
    
#     # Save the plot
#     try:
#         plt.savefig(output_file, dpi=300, bbox_inches='tight')
#         plt.close()
#     except Exception as e:
#         print(f"Error saving plot: {e}")
#         sys.exit(1)

# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         print("Usage: python gains_chart.py <input_csv> <output_png>")
#         sys.exit(1)
    
#     create_gains_chart(sys.argv[1], sys.argv[2])

# import sys
# import csv
# import matplotlib.pyplot as plt

# def read_auc_score(input_file):
#     """Read the AUC score from the CSV file."""
#     with open(input_file, 'r') as file:
#         csv_reader = csv.DictReader(file)
#         for row in csv_reader:
#             if 'FINAL AUC SCORE' in row['Predicted']:
#                 try:
#                     # Convert AUC to percentage
#                     return float(row['Model']) * 100
#                 except (ValueError, KeyError) as e:
#                     print(f"Error reading AUC score: {e}")
#                     return None
#     return None

# def create_gains_chart(input_file, output_file):
#     # Lists to store the specific columns we need
#     row_weights = []
#     model_values = []
#     random_line = []
    
#     # Get AUC score first
#     auc_score = read_auc_score(input_file)
    
#     # Read CSV file
#     with open(input_file, 'r') as file:
#         csv_reader = csv.DictReader(file)
#         for row in csv_reader:
#             # Skip the final AUC score row which contains summary statistics
#             if 'FINAL AUC SCORE' not in row['Predicted']:
#                 try:
#                     # Extract only the three columns we need
#                     row_weights.append(float(row['Row_Weight']))
#                     model_values.append(float(row['Model']))  
#                     random_line.append(float(row['Random_Line']))
#                 except (ValueError, KeyError) as e:
#                     print(f"Error reading data: {e}")
#                     sys.exit(1)

#     # Create the plot
#     plt.figure(figsize=(10, 6))
    
#     # Plot the actual lines using the three columns
#     plt.plot(row_weights, model_values, 'b-', label='Model', linewidth=2)
#     plt.plot(row_weights, random_line, 'r--', label='Random', linewidth=2)
    
#     # Customize the plot
#     plt.xlabel('Population Percentage (Row Weight)')
#     plt.ylabel('Cumulative Gain')
    
#     # Set title with AUC score if available
#     if auc_score is not None:
#         plt.title(f'Gains Chart (AUC: {auc_score:.1f}%)')
#     else:
#         plt.title('Gains Chart (AUC: N/A)')
    
#     plt.grid(True, linestyle='--', alpha=0.7)
    
#     # Format axes to show percentages
#     plt.gca().set_xticklabels([f'{x:.0%}' for x in plt.gca().get_xticks()])
#     plt.gca().set_yticklabels([f'{y:.0%}' for y in plt.gca().get_yticks()])
    
#     # Add legend
#     plt.legend(loc='lower right')
    
#     # Save the plot
#     try:
#         plt.savefig(output_file, dpi=300, bbox_inches='tight')
#         plt.close()
#     except Exception as e:
#         print(f"Error saving plot: {e}")
#         sys.exit(1)

# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         print("Usage: python gains_chart.py <input_csv> <output_png>")
#         sys.exit(1)
    
#     create_gains_chart(sys.argv[1], sys.argv[2])


import sys
import csv
import matplotlib.pyplot as plt

def read_auc_score(input_file):
    """Read the AUC score from the CSV file."""
    try:
        with open(input_file, 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                # Check if the Final_AUC column has a value
                if 'Final_AUC' in row and row['Final_AUC']:
                    try:
                        auc_value = float(row['Final_AUC'])
                        # Convert to percentage
                        return auc_value * 100
                    except (ValueError, KeyError) as e:
                        print(f"Error converting AUC value: {e}")
                        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    return None

def create_gains_chart(input_file, output_file):
    # Lists to store the specific columns we need
    row_weights = []
    model_values = []
    random_line = []
    
    # Get AUC score first
    auc_score = read_auc_score(input_file)
    
    # Read CSV file
    with open(input_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            # Only process rows that don't have a Final_AUC value
            if not row.get('Final_AUC'):
                try:
                    # Extract only the three columns we need
                    row_weights.append(float(row['Row_Weight']))
                    model_values.append(float(row['Model']))  
                    random_line.append(float(row['Random_Line']))
                except (ValueError, KeyError) as e:
                    print(f"Error reading data: {e}")
                    sys.exit(1)

    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot the actual lines using the three columns
    plt.plot(row_weights, model_values, 'b-', label='Model', linewidth=2)
    plt.plot(row_weights, random_line, 'r--', label='Random', linewidth=2)
    
    # Customize the plot
    plt.xlabel('Population Percentage (Row Weight)')
    plt.ylabel('Cumulative Gain')
    
    # Set title with AUC score if available
    if auc_score is not None:
        plt.title(f'Gains Chart (AUC: {auc_score:.1f}%)')
    else:
        plt.title('Gains Chart (AUC: N/A)')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Format axes to show percentages
    plt.gca().set_xticklabels([f'{x:.0%}' for x in plt.gca().get_xticks()])
    plt.gca().set_yticklabels([f'{y:.0%}' for y in plt.gca().get_yticks()])
    
    # Add legend
    plt.legend(loc='lower right')
    
    # Save the plot
    try:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error saving plot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python gains_chart.py <input_csv> <output_png>")
        sys.exit(1)
    
    create_gains_chart(sys.argv[1], sys.argv[2])