# # import matplotlib.pyplot as plt
# # import csv
# # from datetime import datetime
# # from collections import defaultdict

# # def plot_nav_chart(nav_csv, output_png):
# #     # Initialize dictionaries to store data (using last value for each date)
# #     date_data = defaultdict(dict)
    
# #     # Read the CSV file
# #     with open(nav_csv, 'r') as file:
# #         csv_reader = csv.DictReader(file)
# #         for row in csv_reader:
# #             # Skip rows with empty date
# #             if not row['Date']:
# #                 continue
                
# #             date = datetime.strptime(row['Date'], '%d-%m-%Y')
            
# #             # Store all relevant metrics for this date
# #             # Using try/except to handle potential empty or invalid values
# #             try:
# #                 date_data[date]['NAV'] = float(row['NAV'])
# #                 date_data[date]['Max_Drawdown'] = float(row['Max_Drawdown'])
# #                 date_data[date]['Annual_Return'] = float(row['Annual_Return'])
# #                 date_data[date]['AR_MD_Ratio'] = float(row['AR_MD_Ratio'])
# #             except (ValueError, KeyError) as e:
# #                 print(f"Warning: Skipping invalid data for date {date}: {str(e)}")
# #                 continue
    
# #     # Sort dates and prepare lists for plotting
# #     sorted_dates = sorted(date_data.keys())
# #     navs = [date_data[date]['NAV'] for date in sorted_dates]
    
# #     # Create figure and axis
# #     plt.style.use('seaborn')
# #     fig, ax = plt.subplots(figsize=(12, 6))
    
# #     # Plot NAV line
# #     ax.plot(sorted_dates, navs, linewidth=2, color='blue')
    
# #     # Customize plot
# #     ax.set_title('NAV Performance Over Time', fontsize=14, pad=20)
# #     ax.set_xlabel('Date', fontsize=12)
# #     ax.set_ylabel('NAV', fontsize=12)
    
# #     # Rotate x-axis labels
# #     plt.xticks(rotation=45)
    
# #     # Add grid
# #     ax.grid(True, linestyle='--', alpha=0.7)
    
# #     # Get last date's metrics for the text box
# #     last_date = sorted_dates[-1]
# #     last_data = date_data[last_date]
    
# #     # Add metrics as text
# #     metrics_text = f'Max Drawdown: {last_data["Max_Drawdown"]:.2%}\n'
# #     metrics_text += f'Annual Return: {last_data["Annual_Return"]:.2%}\n'
# #     metrics_text += f'AR/MD Ratio: {last_data["AR_MD_Ratio"]:.2f}'
    
# #     plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
# #                 bbox=dict(facecolor='white', alpha=0.8))
    
# #     # Adjust layout and save
# #     plt.tight_layout()
# #     plt.savefig(output_png, dpi=300, bbox_inches='tight')
# #     plt.close()

# # if __name__ == "__main__":
# #     import sys
# #     if len(sys.argv) != 3:
# #         print("Usage: python nav_plot.py <nav_csv_file> <output_png_file>")
# #         sys.exit(1)
    
# #     try:
# #         plot_nav_chart(sys.argv[1], sys.argv[2])
# #         print(f"Successfully created NAV chart: {sys.argv[2]}")
# #     except Exception as e:
# #         print(f"Error creating NAV chart: {str(e)}")

# # import matplotlib.pyplot as plt
# # import csv
# # from datetime import datetime
# # from collections import defaultdict

# # def plot_nav_chart(nav_csv, output_png):
# #     # Initialize dictionaries to store data (using last value for each date)
# #     date_data = defaultdict(dict)
    
# #     # Read the CSV file
# #     with open(nav_csv, 'r') as file:
# #         csv_reader = csv.DictReader(file)
# #         for row in csv_reader:
# #             # Skip rows with empty date
# #             if not row['Date']:
# #                 continue
                
# #             date = datetime.strptime(row['Date'], '%d-%m-%Y')
            
# #             # Store all relevant metrics for this date
# #             # Using try/except to handle potential empty or invalid values
# #             try:
# #                 date_data[date]['NAV'] = float(row['NAV'])
# #                 date_data[date]['Max_Drawdown'] = float(row['Max_Drawdown'])
# #                 date_data[date]['Annual_Return'] = float(row['Annual_Return'])
# #                 date_data[date]['AR_MD_Ratio'] = float(row['AR_MD_Ratio'])
# #             except (ValueError, KeyError) as e:
# #                 print(f"Warning: Skipping invalid data for date {date}: {str(e)}")
# #                 continue
    
# #     # Sort dates and prepare lists for plotting
# #     sorted_dates = sorted(date_data.keys())
# #     navs = [date_data[date]['NAV'] for date in sorted_dates]
    
# #     # Create figure and axis
# #     fig, ax = plt.subplots(figsize=(12, 6))
    
# #     # Plot NAV line
# #     ax.plot(sorted_dates, navs, linewidth=2, color='blue')
    
# #     # Customize plot
# #     ax.set_title('NAV Performance Over Time', fontsize=14, pad=20)
# #     ax.set_xlabel('Date', fontsize=12)
# #     ax.set_ylabel('NAV', fontsize=12)
    
# #     # Rotate x-axis labels
# #     plt.xticks(rotation=45)
    
# #     # Add grid
# #     ax.grid(True, linestyle='--', alpha=0.7)
    
# #     # Get last date's metrics for the text box
# #     last_date = sorted_dates[-1]
# #     last_data = date_data[last_date]
    
# #     # Add metrics as text
# #     metrics_text = f'Max Drawdown: {last_data["Max_Drawdown"]:.2%}\n'
# #     metrics_text += f'Annual Return: {last_data["Annual_Return"]:.2%}\n'
# #     metrics_text += f'AR/MD Ratio: {last_data["AR_MD_Ratio"]:.2f}'
    
# #     plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
# #                 bbox=dict(facecolor='white', alpha=0.8))
    
# #     # Adjust layout and save
# #     plt.tight_layout()
# #     plt.savefig(output_png, dpi=300, bbox_inches='tight')
# #     plt.close()

# # if __name__ == "__main__":
# #     import sys
# #     if len(sys.argv) != 3:
# #         print("Usage: python nav_plot.py <nav_csv_file> <output_png_file>")
# #         sys.exit(1)
    
# #     try:
# #         plot_nav_chart(sys.argv[1], sys.argv[2])
# #         print(f"Successfully created NAV chart: {sys.argv[2]}")
# #     except Exception as e:
# #         print(f"Error creating NAV chart: {str(e)}")

# # import matplotlib.pyplot as plt
# # import csv
# # from datetime import datetime
# # from collections import defaultdict

# # def plot_and_analyze_nav(nav_csv, output_png, output_metrics_csv):
# #     # Initialize dictionaries to store data
# #     date_data = defaultdict(dict)
    
# #     # Read the CSV file
# #     with open(nav_csv, 'r') as file:
# #         csv_reader = csv.DictReader(file)
# #         for row in csv_reader:
# #             # Skip rows with empty date
# #             if not row['Date']:
# #                 continue
                
# #             date = datetime.strptime(row['Date'], '%d-%m-%Y')
            
# #             # Store all relevant metrics for this date
# #             try:
# #                 date_data[date]['NAV'] = float(row['NAV'])
# #                 date_data[date]['Max_Drawdown'] = float(row['Max_Drawdown'])
# #                 date_data[date]['Annual_Return'] = float(row['Annual_Return'])
# #                 date_data[date]['AR_MD_Ratio'] = float(row['AR_MD_Ratio'])
# #             except (ValueError, KeyError) as e:
# #                 print(f"Warning: Skipping invalid data for date {date}: {str(e)}")
# #                 continue
    
# #     # Sort dates
# #     sorted_dates = sorted(date_data.keys())
    
# #     # Calculate normalized gains
# #     initial_nav = date_data[sorted_dates[0]]['NAV']
# #     for date in sorted_dates:
# #         current_nav = date_data[date]['NAV']
# #         date_data[date]['Normalized_Gain'] = (current_nav - initial_nav) / initial_nav
    
# #     # Create metrics CSV file
# #     with open(output_metrics_csv, 'w', newline='') as file:
# #         fieldnames = ['Date', 'NAV', 'Normalized_Gain', 'Max_Drawdown', 'Drawdown_Date']
# #         writer = csv.DictWriter(file, fieldnames=fieldnames)
# #         writer.writeheader()
        
# #         for date in sorted_dates:
# #             # Find drawdown date (date of previous peak)
# #             drawdown_date = None
# #             max_nav = 0
# #             for prev_date in sorted_dates:
# #                 if prev_date > date:
# #                     break
# #                 if date_data[prev_date]['NAV'] > max_nav:
# #                     max_nav = date_data[prev_date]['NAV']
# #                     drawdown_date = prev_date
            
# #             writer.writerow({
# #                 'Date': date.strftime('%d-%m-%Y'),
# #                 'NAV': f"{date_data[date]['NAV']:.4f}",
# #                 'Normalized_Gain': f"{date_data[date]['Normalized_Gain']:.4f}",
# #                 'Max_Drawdown': f"{date_data[date]['Max_Drawdown']:.4f}",
# #                 'Drawdown_Date': drawdown_date.strftime('%d-%m-%Y') if drawdown_date else ''
# #             })
    
# #     # Create plot
# #     navs = [date_data[date]['NAV'] for date in sorted_dates]
    
# #     # Create figure and axis
# #     fig, ax = plt.subplots(figsize=(12, 6))
    
# #     # Plot NAV line
# #     ax.plot(sorted_dates, navs, linewidth=2, color='blue')
    
# #     # Customize plot
# #     ax.set_title('NAV Performance Over Time', fontsize=14, pad=20)
# #     ax.set_xlabel('Date', fontsize=12)
# #     ax.set_ylabel('NAV', fontsize=12)
    
# #     # Rotate x-axis labels
# #     plt.xticks(rotation=45)
    
# #     # Add grid
# #     ax.grid(True, linestyle='--', alpha=0.7)
    
# #     # Get last date's metrics for the text box
# #     last_date = sorted_dates[-1]
# #     last_data = date_data[last_date]
    
# #     # Add metrics as text
# #     metrics_text = f'Max Drawdown: {last_data["Max_Drawdown"]:.2%}\n'
# #     metrics_text += f'Annual Return: {last_data["Annual_Return"]:.2%}\n'
# #     metrics_text += f'AR/MD Ratio: {last_data["AR_MD_Ratio"]:.2f}'
    
# #     plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
# #                 bbox=dict(facecolor='white', alpha=0.8))
    
# #     # Adjust layout and save
# #     plt.tight_layout()
# #     plt.savefig(output_png, dpi=300, bbox_inches='tight')
# #     plt.close()

# # if __name__ == "__main__":
# #     import sys
# #     if len(sys.argv) != 4:
# #         print("Usage: python nav_plot.py <nav_csv_file> <output_png_file> <output_metrics_csv>")
# #         sys.exit(1)
    
# #     try:
# #         plot_and_analyze_nav(sys.argv[1], sys.argv[2], sys.argv[3])
# #         print(f"Successfully created NAV chart: {sys.argv[2]}")
# #         print(f"Successfully created metrics CSV: {sys.argv[3]}")
# #     except Exception as e:
# #         print(f"Error in processing: {str(e)}")


# # import matplotlib.pyplot as plt
# # import csv
# # from datetime import datetime
# # from collections import defaultdict

# # def plot_and_analyze_nav(nav_csv, output_png, output_metrics_csv):
# #     # Initialize dictionaries to store data
# #     date_data = defaultdict(dict)
    
# #     # Read the CSV file
# #     with open(nav_csv, 'r') as file:
# #         csv_reader = csv.DictReader(file)
# #         for row in csv_reader:
# #             # Skip rows with empty date
# #             if not row['Date']:
# #                 continue
                
# #             date = datetime.strptime(row['Date'], '%d-%m-%Y')
            
# #             # Store NAV and drawdown for this date
# #             try:
# #                 date_data[date]['NAV'] = float(row['NAV'])
# #                 date_data[date]['Max_Drawdown'] = float(row['Max_Drawdown'])
# #             except (ValueError, KeyError) as e:
# #                 print(f"Warning: Skipping invalid data for date {date}: {str(e)}")
# #                 continue
    
# #     # Sort dates
# #     sorted_dates = sorted(date_data.keys())
    
# #     # Calculate summed gains (cumulative return)
# #     initial_nav = date_data[sorted_dates[0]]['NAV']
# #     running_sum = 0
    
# #     # Create metrics CSV file
# #     with open(output_metrics_csv, 'w', newline='') as file:
# #         fieldnames = ['Date', 'NAV', 'Summed_Gain', 'Drawdown']
# #         writer = csv.DictWriter(file, fieldnames=fieldnames)
# #         writer.writeheader()
        
# #         for date in sorted_dates:
# #             current_nav = date_data[date]['NAV']
# #             period_return = (current_nav - initial_nav) / initial_nav
# #             running_sum += period_return
            
# #             writer.writerow({
# #                 'Date': date.strftime('%d-%m-%Y'),
# #                 'NAV': f"{date_data[date]['NAV']:.4f}",
# #                 'Summed_Gain': f"{running_sum:.4f}",
# #                 'Drawdown': f"{date_data[date]['Max_Drawdown']:.4f}"
# #             })
    
# #     # Create plot
# #     navs = [date_data[date]['NAV'] for date in sorted_dates]
    
# #     # Create figure and axis
# #     fig, ax = plt.subplots(figsize=(12, 6))
    
# #     # Plot NAV line
# #     ax.plot(sorted_dates, navs, linewidth=2, color='blue')
    
# #     # Customize plot
# #     ax.set_title('NAV Performance Over Time', fontsize=14, pad=20)
# #     ax.set_xlabel('Date', fontsize=12)
# #     ax.set_ylabel('NAV', fontsize=12)
    
# #     # Rotate x-axis labels
# #     plt.xticks(rotation=45)
    
# #     # Add grid
# #     ax.grid(True, linestyle='--', alpha=0.7)
    
# #     # Get last date's metrics for the text box
# #     last_date = sorted_dates[-1]
# #     last_data = date_data[last_date]
    
# #     # Add metrics as text
# #     metrics_text = f'Current NAV: {last_data["NAV"]:.2f}\n'
# #     metrics_text += f'Total Drawdown: {last_data["Max_Drawdown"]:.2%}'
    
# #     plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
# #                 bbox=dict(facecolor='white', alpha=0.8))
    
# #     # Adjust layout and save
# #     plt.tight_layout()
# #     plt.savefig(output_png, dpi=300, bbox_inches='tight')
# #     plt.close()

# # if __name__ == "__main__":
# #     import sys
# #     if len(sys.argv) != 4:
# #         print("Usage: python nav_plot.py <nav_csv_file> <output_png_file> <output_metrics_csv>")
# #         sys.exit(1)
    
# #     try:
# #         plot_and_analyze_nav(sys.argv[1], sys.argv[2], sys.argv[3])
# #         print(f"Successfully created NAV chart: {sys.argv[2]}")
# #         print(f"Successfully created metrics CSV: {sys.argv[3]}")
# #     except Exception as e:
# #         print(f"Error in processing: {str(e)}")


# # import matplotlib.pyplot as plt
# # import csv
# # from datetime import datetime
# # from collections import defaultdict

# # def plot_and_analyze_nav(nav_csv, output_png, output_metrics_csv):
# #     # Initialize dictionaries to store data
# #     date_data = defaultdict(dict)
    
# #     # Read the CSV file
# #     with open(nav_csv, 'r') as file:
# #         csv_reader = csv.DictReader(file)
# #         for row in csv_reader:
# #             # Skip rows with empty date
# #             if not row['Date']:
# #                 continue
                
# #             date = datetime.strptime(row['Date'], '%d-%m-%Y')
            
# #             # Store NAV and Annual Return for this date
# #             try:
# #                 date_data[date]['NAV'] = float(row['NAV'])
# #                 date_data[date]['Annual_Return'] = float(row['Annual_Return'])
# #             except (ValueError, KeyError) as e:
# #                 print(f"Warning: Skipping invalid data for date {date}: {str(e)}")
# #                 continue
    
# #     # Sort dates
# #     sorted_dates = sorted(date_data.keys())
    
# #     # Calculate drawdown for each date
# #     peak_nav = float('-inf')
# #     for date in sorted_dates:
# #         current_nav = date_data[date]['NAV']
# #         # Update peak if current NAV is higher
# #         peak_nav = max(peak_nav, current_nav)
# #         # Calculate drawdown as percentage from peak
# #         if peak_nav > 0:
# #             drawdown = (peak_nav - current_nav) / peak_nav
# #         else:
# #             drawdown = 0
# #         date_data[date]['Drawdown'] = drawdown
    
# #     # Create metrics CSV file
# #     with open(output_metrics_csv, 'w', newline='') as file:
# #         fieldnames = ['Date', 'Summed_Gain', 'NAV', 'Drawdown']
# #         writer = csv.DictWriter(file, fieldnames=fieldnames)
# #         writer.writeheader()
        
# #         for date in sorted_dates:
# #             writer.writerow({
# #                 'Date': date.strftime('%d-%m-%Y'),
# #                 'Summed_Gain': f"{date_data[date]['Annual_Return']:.4f}",
# #                 'NAV': f"{date_data[date]['NAV']:.4f}",
# #                 'Drawdown': f"{date_data[date]['Drawdown']:.4f}"
# #             })
    
# #     # Create plot
# #     navs = [date_data[date]['NAV'] for date in sorted_dates]
    
# #     # Create figure and axis
# #     fig, ax = plt.subplots(figsize=(12, 6))
    
# #     # Plot NAV line
# #     ax.plot(sorted_dates, navs, linewidth=2, color='blue')
    
# #     # Customize plot
# #     ax.set_title('NAV Performance Over Time', fontsize=14, pad=20)
# #     ax.set_xlabel('Date', fontsize=12)
# #     ax.set_ylabel('NAV', fontsize=12)
    
# #     # Rotate x-axis labels
# #     plt.xticks(rotation=45)
    
# #     # Add grid
# #     ax.grid(True, linestyle='--', alpha=0.7)
    
# #     # Get last date's metrics for the text box
# #     last_date = sorted_dates[-1]
# #     last_data = date_data[last_date]
    
# #     # Add metrics as text
# #     metrics_text = f'Current NAV: {last_data["NAV"]:.2f}\n'
# #     metrics_text += f'Current Drawdown: {last_data["Drawdown"]:.2%}'
    
# #     plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
# #                 bbox=dict(facecolor='white', alpha=0.8))
    
# #     # Adjust layout and save
# #     plt.tight_layout()
# #     plt.savefig(output_png, dpi=300, bbox_inches='tight')
# #     plt.close()

# # if __name__ == "__main__":
# #     import sys
# #     if len(sys.argv) != 4:
# #         print("Usage: python nav_plot.py <nav_csv_file> <output_png_file> <output_metrics_csv>")
# #         sys.exit(1)
    
# #     try:
# #         plot_and_analyze_nav(sys.argv[1], sys.argv[2], sys.argv[3])
# #         print(f"Successfully created NAV chart: {sys.argv[2]}")
# #         print(f"Successfully created metrics CSV: {sys.argv[3]}")
# #     except Exception as e:
# #         print(f"Error in processing: {str(e)}")


# # import matplotlib.pyplot as plt
# # import csv
# # from datetime import datetime

# # def plot_and_analyze_nav(nav_csv, output_png, output_metrics_csv):
# #     # Lists to store data
# #     dates = []
# #     summed_gains = []
# #     navs = []
# #     drawdowns = []
    
# #     # Read the CSV file
# #     with open(nav_csv, 'r') as file:
# #         csv_reader = csv.DictReader(file)
# #         for row in csv_reader:
# #             try:
# #                 # Parse date
# #                 date = datetime.strptime(row['Date'].strip(), '%d-%m-%Y')
                
# #                 # Get values directly from columns
# #                 summed_gain = float(row['Summed_Gain'])
# #                 nav = float(row['NAV'])
# #                 drawdown = float(row['Drawdown'])
                
# #                 # Append to lists
# #                 dates.append(date)
# #                 summed_gains.append(summed_gain)
# #                 navs.append(nav)
# #                 drawdowns.append(drawdown)
                
# #             except (ValueError, KeyError) as e:
# #                 print(f"Warning: Skipping invalid row: {str(e)}")
# #                 continue
    
# #     # Create metrics CSV file
# #     with open(output_metrics_csv, 'w', newline='') as file:
# #         fieldnames = ['Date', 'Summed_Gain', 'NAV', 'Drawdown']
# #         writer = csv.DictWriter(file, fieldnames=fieldnames)
# #         writer.writeheader()
        
# #         for i in range(len(dates)):
# #             writer.writerow({
# #                 'Date': dates[i].strftime('%d-%m-%Y'),
# #                 'Summed_Gain': f"{summed_gains[i]:.6f}",
# #                 'NAV': f"{navs[i]:.6f}",
# #                 'Drawdown': f"{drawdowns[i]:.6f}"
# #             })
    
# #     # Create plot
# #     fig, ax = plt.subplots(figsize=(12, 6))
    
# #     # Plot NAV line
# #     ax.plot(dates, navs, linewidth=2, color='blue')
    
# #     # Customize plot
# #     ax.set_title('NAV Performance Over Time', fontsize=14, pad=20)
# #     ax.set_xlabel('Date', fontsize=12)
# #     ax.set_ylabel('NAV', fontsize=12)
    
# #     # Rotate x-axis labels
# #     plt.xticks(rotation=45)
    
# #     # Add grid
# #     ax.grid(True, linestyle='--', alpha=0.7)
    
# #     # Add metrics as text
# #     if dates:
# #         metrics_text = f'Current NAV: {navs[-1]:.6f}\n'
# #         metrics_text += f'Current Drawdown: {drawdowns[-1]:.6f}'
        
# #         plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
# #                     bbox=dict(facecolor='white', alpha=0.8))
    
# #     # Adjust layout and save
# #     plt.tight_layout()
# #     plt.savefig(output_png, dpi=300, bbox_inches='tight')
# #     plt.close()

# # if __name__ == "__main__":
# #     import sys
# #     if len(sys.argv) != 4:
# #         print("Usage: python nav_plot.py <nav_csv_file> <output_png_file> <output_metrics_csv>")
# #         sys.exit(1)
    
# #     try:
# #         plot_and_analyze_nav(sys.argv[1], sys.argv[2], sys.argv[3])
# #         print(f"Successfully created NAV chart: {sys.argv[2]}")
# #         print(f"Successfully created metrics CSV: {sys.argv[3]}")
# #     except Exception as e:
# #         print(f"Error in processing: {str(e)}")

# # import matplotlib.pyplot as plt
# # import csv
# # from datetime import datetime
# # from collections import defaultdict

# # def plot_and_analyze_nav(nav_csv, output_png, output_metrics_csv):
# #     # Dictionary to store data by date
# #     date_data = {}
    
# #     # Read the CSV file
# #     with open(nav_csv, 'r') as file:
# #         csv_reader = csv.DictReader(file)
# #         for row in csv_reader:
# #             try:
# #                 # Parse date
# #                 date = datetime.strptime(row['Date'].strip(), '%d-%m-%Y')
# #                 date_str = date.strftime('%d-%m-%Y')
                
# #                 # Get values directly from columns
# #                 summed_gain = float(row['Summed_Gain'])
# #                 nav = float(row['NAV'])
# #                 drawdown = float(row['Drawdown'])
                
# #                 # Store only the first occurrence of each date
# #                 if date_str not in date_data:
# #                     date_data[date_str] = {
# #                         'date': date,
# #                         'summed_gain': summed_gain,
# #                         'nav': nav,
# #                         'drawdown': drawdown
# #                     }
                
# #             except (ValueError, KeyError) as e:
# #                 print(f"Warning: Skipping invalid row: {str(e)}")
# #                 continue
    
# #     # Sort dates
# #     sorted_dates = sorted(date_data.keys())
    
# #     # Create metrics CSV file
# #     with open(output_metrics_csv, 'w', newline='') as file:
# #         fieldnames = ['Date', 'Summed_Gain', 'NAV', 'Drawdown']
# #         writer = csv.DictWriter(file, fieldnames=fieldnames)
# #         writer.writeheader()
        
# #         for date_str in sorted_dates:
# #             data = date_data[date_str]
# #             writer.writerow({
# #                 'Date': date_str,
# #                 'Summed_Gain': f"{data['summed_gain']:.6f}",
# #                 'NAV': f"{data['nav']:.6f}",
# #                 'Drawdown': f"{data['drawdown']:.6f}"
# #             })
    
# #     # Create plot
# #     dates = [date_data[d]['date'] for d in sorted_dates]
# #     navs = [date_data[d]['nav'] for d in sorted_dates]
    
# #     fig, ax = plt.subplots(figsize=(12, 6))
    
# #     # Plot NAV line
# #     ax.plot(dates, navs, linewidth=2, color='blue')
    
# #     # Customize plot
# #     ax.set_title('NAV Performance Over Time', fontsize=14, pad=20)
# #     ax.set_xlabel('Date', fontsize=12)
# #     ax.set_ylabel('NAV', fontsize=12)
    
# #     # Rotate x-axis labels
# #     plt.xticks(rotation=45)
    
# #     # Add grid
# #     ax.grid(True, linestyle='--', alpha=0.7)
    
# #     # Add metrics as text
# #     if sorted_dates:
# #         last_data = date_data[sorted_dates[-1]]
# #         metrics_text = f'Current NAV: {last_data["nav"]:.6f}\n'
# #         metrics_text += f'Current Drawdown: {last_data["drawdown"]:.6f}'
        
# #         plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
# #                     bbox=dict(facecolor='white', alpha=0.8))
    
# #     # Adjust layout and save
# #     plt.tight_layout()
# #     plt.savefig(output_png, dpi=300, bbox_inches='tight')
# #     plt.close()

# # if __name__ == "__main__":
# #     import sys
# #     if len(sys.argv) != 4:
# #         print("Usage: python nav_plot.py <nav_csv_file> <output_png_file> <output_metrics_csv>")
# #         sys.exit(1)
    
# #     try:
# #         plot_and_analyze_nav(sys.argv[1], sys.argv[2], sys.argv[3])
# #         print(f"Successfully created NAV chart: {sys.argv[2]}")
# #         print(f"Successfully created metrics CSV: {sys.argv[3]}")
# #     except Exception as e:
# #         print(f"Error in processing: {str(e)}")

# import matplotlib.pyplot as plt
# import csv
# from datetime import datetime

# def plot_and_analyze_nav(nav_csv, output_png, output_metrics_csv):
#     # Dictionary to store data by date
#     date_data = {}
    
#     # Read the CSV file
#     with open(nav_csv, 'r') as file:
#         csv_reader = csv.DictReader(file)
#         for row in csv_reader:
#             try:
#                 # Parse date
#                 date = datetime.strptime(row['Date'].strip(), '%d-%m-%Y')
#                 date_str = date.strftime('%d-%m-%Y')
                
#                 # Get values directly from columns
#                 summed_gain = float(row['Summed_Gain'])
#                 nav = float(row['NAV'])
#                 drawdown = float(row['Drawdown'])
                
#                 # For duplicate dates, we'll keep the first occurrence
#                 if date_str not in date_data:
#                     date_data[date_str] = {
#                         'date_obj': date,  # Store datetime object for sorting
#                         'summed_gain': summed_gain,
#                         'nav': nav,
#                         'drawdown': drawdown
#                     }
                
#             except (ValueError, KeyError) as e:
#                 print(f"Warning: Skipping invalid row: {str(e)}")
#                 continue
    
#     # Sort dates using datetime objects
#     sorted_dates = sorted(date_data.items(), key=lambda x: x[1]['date_obj'])
    
#     # Create metrics CSV file
#     with open(output_metrics_csv, 'w', newline='') as file:
#         fieldnames = ['Date', 'Summed_Gain', 'NAV', 'Drawdown']
#         writer = csv.DictWriter(file, fieldnames=fieldnames)
#         writer.writeheader()
        
#         for date_str, data in sorted_dates:
#             writer.writerow({
#                 'Date': date_str,
#                 'Summed_Gain': f"{data['summed_gain']:.6f}",
#                 'NAV': f"{data['nav']:.6f}",
#                 'Drawdown': f"{data['drawdown']:.6f}"
#             })
    
#     # Create plot
#     dates = [data['date_obj'] for _, data in sorted_dates]
#     navs = [data['nav'] for _, data in sorted_dates]
    
#     fig, ax = plt.subplots(figsize=(12, 6))
    
#     # Plot NAV line
#     ax.plot(dates, navs, linewidth=2, color='blue')
    
#     # Customize plot
#     ax.set_title('NAV Performance Over Time', fontsize=14, pad=20)
#     ax.set_xlabel('Date', fontsize=12)
#     ax.set_ylabel('NAV', fontsize=12)
    
#     # Rotate x-axis labels
#     plt.xticks(rotation=45)
    
#     # Add grid
#     ax.grid(True, linestyle='--', alpha=0.7)
    
#     # Add metrics as text
#     if sorted_dates:
#         last_data = sorted_dates[-1][1]
#         metrics_text = f'Current NAV: {last_data["nav"]:.6f}\n'
#         metrics_text += f'Current Drawdown: {last_data["drawdown"]:.6f}'
        
#         plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
#                     bbox=dict(facecolor='white', alpha=0.8))
    
#     # Adjust layout and save
#     plt.tight_layout()
#     plt.savefig(output_png, dpi=300, bbox_inches='tight')
#     plt.close()

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) != 4:
#         print("Usage: python nav_plot.py <nav_csv_file> <output_png_file> <output_metrics_csv>")
#         sys.exit(1)
    
#     try:
#         plot_and_analyze_nav(sys.argv[1], sys.argv[2], sys.argv[3])
#         print(f"Successfully created NAV chart: {sys.argv[2]}")
#         print(f"Successfully created metrics CSV: {sys.argv[3]}")
#     except Exception as e:
#         print(f"Error in processing: {str(e)}")

import matplotlib.pyplot as plt
import csv
from datetime import datetime

def parse_date(date_str):
    formats = ['%d-%m-%Y', '%Y-%m-%d']
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"Date {date_str} doesn't match any supported format")

def plot_and_analyze_nav(nav_csv, output_png, output_metrics_csv):
    date_data = {}
    
    with open(nav_csv, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            try:
                date = parse_date(row['Date'])
                date_str = date.strftime('%d-%m-%Y')
                
                summed_gain = float(row['Summed_Gain'])
                nav = float(row['NAV'])
                drawdown = float(row['Drawdown'])
                
                if date_str not in date_data:
                    date_data[date_str] = {
                        'date_obj': date,
                        'summed_gain': summed_gain,
                        'nav': nav,
                        'drawdown': drawdown
                    }
                
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping invalid row: {str(e)}")
                continue
    
    sorted_dates = sorted(date_data.items(), key=lambda x: x[1]['date_obj'])
    
    with open(output_metrics_csv, 'w', newline='') as file:
        fieldnames = ['Date', 'Summed_Gain', 'NAV', 'Drawdown']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        for date_str, data in sorted_dates:
            writer.writerow({
                'Date': date_str,
                'Summed_Gain': f"{data['summed_gain']:.6f}",
                'NAV': f"{data['nav']:.6f}",
                'Drawdown': f"{data['drawdown']:.6f}"
            })
    
    dates = [data['date_obj'] for _, data in sorted_dates]
    navs = [data['nav'] for _, data in sorted_dates]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, navs, linewidth=2, color='blue')
    
    ax.set_title('NAV Performance Over Time', fontsize=14, pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('NAV', fontsize=12)
    plt.xticks(rotation=45)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    if sorted_dates:
        last_data = sorted_dates[-1][1]
        metrics_text = f'Current NAV: {last_data["nav"]:.6f}\n'
        metrics_text += f'Current Drawdown: {last_data["drawdown"]:.6f}'
        plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python nav_plot.py <nav_csv_file> <output_png_file> <output_metrics_csv>")
        sys.exit(1)
    
    try:
        plot_and_analyze_nav(sys.argv[1], sys.argv[2], sys.argv[3])
        print(f"Successfully created NAV chart: {sys.argv[2]}")
        print(f"Successfully created metrics CSV: {sys.argv[3]}")
    except Exception as e:
        print(f"Error in processing: {str(e)}")