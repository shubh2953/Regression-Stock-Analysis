#include <iostream>
#include <sqlite3.h>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <chrono>
#include <ctime>
#include <numeric>
#include <iomanip>
#include <deque>
#include <tuple>
//#include <unordered_map>

std::deque<double> recent_ratio_0_percentiles;  // Global deque for ratio_0_percentile
std::deque<double> recent_ratio_100_percentiles;

// Function to load data from SQLite
std::vector<std::map<std::string, std::string>> load_data_from_sqlite() {
    sqlite3* db;
    sqlite3_stmt* stmt;
    std::vector<std::map<std::string, std::string>> data;

    if (sqlite3_open("stocks.db", &db) == SQLITE_OK) {
        std::string query = "SELECT stock_name, Date, Adj_Close FROM StockHistoricalPrices WHERE Date >= '2015-09-08'";
        if (sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, nullptr) == SQLITE_OK) {
            while (sqlite3_step(stmt) == SQLITE_ROW) {
                std::map<std::string, std::string> row;
                row["stock_name"] = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
                row["Date"] = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
                row["Adj_Close"] = std::to_string(sqlite3_column_double(stmt, 2));
                data.push_back(row);
            }
            sqlite3_finalize(stmt);
        }
        sqlite3_close(db);
    }
    return data;
}

//Function to prefilter data
std::vector<std::map<std::string, std::string>> prefilter_data(const std::vector<std::map<std::string, std::string>>& data) {
    std::map<std::string, int> ticker_counts;
    for (const auto& row : data) {
        ticker_counts[row.at("stock_name")]++;
    }

    std::vector<std::string> tickers_with_6_years_data;
    for (const auto& pair : ticker_counts) {
        if (pair.second >= 7 * 250) {
            tickers_with_6_years_data.push_back(pair.first);
        }
    }

    std::vector<std::map<std::string, std::string>> filtered_data;
    for (const auto& row : data) {
        if (std::find(tickers_with_6_years_data.begin(), tickers_with_6_years_data.end(), row.at("stock_name")) != tickers_with_6_years_data.end()) {
            filtered_data.push_back(row);
        }
    }
    return filtered_data;
}


// Function to merge ticker data
std::vector<std::pair<std::string, double>> merge_ticker_data(const std::vector<std::map<std::string, std::string>>& data, const std::string& ticker1, const std::string& ticker2, const std::string& start_date, const std::string& end_date) {
    std::vector<std::pair<std::string, double>> merged_data;
    for (const auto& row : data) {
        if ((row.at("stock_name") == ticker1 || row.at("stock_name") == ticker2) &&
            row.at("Date") >= start_date && row.at("Date") <= end_date) {
            merged_data.push_back({row.at("Date"), std::stod(row.at("Adj_Close"))});
        }
    }
    return merged_data;
}

// Function to calculate percentiles
std::tuple<double, double, double, double, double> get_percentiles(const std::vector<std::pair<std::string, double>>& merged_data) {
    std::vector<double> values;
    for (const auto& pair : merged_data) {
        values.push_back(pair.second);  // Use the double values directly
    }

    // Sort values to compute percentiles
    std::sort(values.begin(), values.end());

    // Calculate percentiles and average
    double avg = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double percentile_0 = values.front();  // 0th percentile (min)
    double percentile_1 = values[static_cast<size_t>(values.size() * 0.01)];  // 1st percentile
    double percentile_99 = values[static_cast<size_t>(values.size() * 0.99)];  // 99th percentile
    double percentile_100 = values.back();  // 100th percentile (max)

    return std::make_tuple(percentile_0, percentile_1, percentile_99, percentile_100, avg);
}


// Function to calculate ratio variance
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

double calculate_ratio_variance(const std::vector<std::pair<std::string, double>>& merged_data, 
                                const std::string& ticker1, const std::string& ticker_B, 
                                const std::string& ticker_C, const std::string& ticker_D) 
{
    std::vector<double> ratios;

    // Calculate the desired ratios for each ticker
    for (const auto& pair : merged_data) {
        // Assuming the data contains ratios directly for each ticker in merged_data
        // Calculate and add ratios
        double ratio = pair.second;  // directly using the second value as the ratio for simplicity
        ratios.push_back(ratio);
    }

    // Sort ratios to compute percentiles
    std::sort(ratios.begin(), ratios.end());

    // Calculate percentiles
    double percentile_1 = ratios[static_cast<size_t>(ratios.size() * 0.01)];
    double percentile_99 = ratios[static_cast<size_t>(ratios.size() * 0.99)];

    // Calculate X, Y, Z based on percentiles
    double X = std::sqrt(percentile_1 * percentile_99);
    double Y = (percentile_99 / X) - 1;
    double Z = std::sqrt(static_cast<double>(ratios.size()));

    return Z / Y;
}



#include <deque>
#include <tuple>

void update_recent_percentiles(double ratio_0_percentile, double ratio_100_percentile) {
    // Update the deque for ratio_0_percentile
    if (recent_ratio_0_percentiles.size() == 20) {
        recent_ratio_0_percentiles.pop_front(); // Remove the oldest value
    }
    recent_ratio_0_percentiles.push_back(ratio_0_percentile); // Add the new value

    // Update the deque for ratio_100_percentile
    if (recent_ratio_100_percentiles.size() == 20) {
        recent_ratio_100_percentiles.pop_front(); // Remove the oldest value
    }
    recent_ratio_100_percentiles.push_back(ratio_100_percentile); // Add the new value
}


std::vector<std::map<std::string, double>> create_ticker_price_df(
    const std::vector<std::map<std::string, std::string>>& data,
    const std::vector<std::string>& tickers,
    const std::string& start_date_h,
    const std::string& end_date_h) 
{
    sqlite3* db;
    sqlite3_stmt* stmt;
    std::vector<std::map<std::string, double>> ticker_price_df;
    std::map<std::string, std::map<std::string, double>> date_to_ticker_data;

    // Convert tickers list into a comma-separated string for SQL query
    std::string ticker_list;
    for (size_t i = 0; i < tickers.size(); ++i) {
        ticker_list += "'" + tickers[i] + "'";
        if (i < tickers.size() - 1) {
            ticker_list += ",";
        }
    }

    if (sqlite3_open("stocks.db", &db) == SQLITE_OK) {
        std::string query = "SELECT stock_name, Date, Adj_Close FROM StockHistoricalPrices "
                            "WHERE Date >= '" + start_date_h + "' AND Date <= '" + end_date_h +
                            "' AND stock_name IN (" + ticker_list + ")";
        
        if (sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, nullptr) == SQLITE_OK) {
            while (sqlite3_step(stmt) == SQLITE_ROW) {
                std::string ticker = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
                std::string row_date = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
                double adj_close = sqlite3_column_double(stmt, 2);

                // Initialize the date entry if it does not exist
                if (date_to_ticker_data.find(row_date) == date_to_ticker_data.end()) {
                    date_to_ticker_data[row_date] = {};
                    date_to_ticker_data[row_date]["Date"] = std::stod(row_date); // Assumes date can be converted to double
                }

                // Add the adjusted close price for the ticker
                date_to_ticker_data[row_date][ticker] = adj_close;
            }
            sqlite3_finalize(stmt);
        }
        sqlite3_close(db);
    }

    // Convert the date-to-ticker map into the final vector
    for (const auto& [date, ticker_data] : date_to_ticker_data) {
        ticker_price_df.push_back(ticker_data);
    }

    return ticker_price_df;
}



// Function to calculate volatility
double volatility_calculate(const std::vector<double>& ticker, int days) {
    std::vector<double> rate_of_change_wrt_previous_day;
    for (size_t i = 1; i < ticker.size(); ++i) {
        rate_of_change_wrt_previous_day.push_back(((ticker[i] / ticker[i - 1]) - 1) * 100);
    }
    double volatility = 0;
    for (const auto& change : rate_of_change_wrt_previous_day) {
        volatility += std::abs(change);
    }
    return volatility / days;
}

double calculate_5_day_avg(const std::vector<std::pair<std::string, double>>& data) {
    if (data.size() < 5) {
        throw std::invalid_argument("Insufficient data for 5-day average.");
    }

    // Sum the last 5 values
    double sum = 0.0;
    for (size_t i = data.size() - 5; i < data.size(); ++i) {
        sum += data[i].second; // `second` holds the ratio values
    }
    return sum / 5.0;
}

double calculate_20_day_avg(const std::vector<std::pair<std::string, double>>& data) {
    if (data.size() < 20) {
        throw std::invalid_argument("Insufficient data for 20-day average.");
    }

    // Sum the last 20 values
    double sum = 0.0;
    for (size_t i = data.size() - 20; i < data.size(); ++i) {
        sum += data[i].second; // `second` holds the ratio values
    }
    return sum / 20.0;
}


double calculate_pct_change(double current, double previous, const std::string& ticker, const std::string& date) {
    if (previous == 0) {
        // Print debug information to identify the problematic data
        std::cerr << "Error: Previous value is zero for ticker: " << ticker 
                  << " on date: " << date << ". Division by zero error." << std::endl;
        throw std::invalid_argument("Previous value is zero. Division by zero error.");
    }
    return (current - previous) /previous;
}

double calculate_pct_change_5_day(double current, double past) {
    if (past == 0) {
        throw std::invalid_argument("Past value is zero. Division by zero error.");
    }
    return (current - past) / past;
}

// Function to calculate 250-day drawdown
double calculate_drawdown(const std::vector<double>& ratio_values) {
    if (ratio_values.size() < 250) {
        throw std::invalid_argument("Insufficient data for 250-day drawdown.");
    }

    double max_ratio = *std::max_element(ratio_values.end() - 250, ratio_values.end());
    double current_ratio = ratio_values.back();

    return ((max_ratio - current_ratio) / max_ratio) * 100.0; // Expressed as a percentage
}

// Function to calculate the momentum indicator
int calculate_momentum_indicator(const std::vector<double>& ratio_values) {
    if (ratio_values.size() < 2) {
        throw std::invalid_argument("Insufficient data for momentum calculation.");
    }

    double today = ratio_values.back();
    double yesterday = *(ratio_values.end() - 2);

    return (today > yesterday) == (*(ratio_values.end() - 2) > *(ratio_values.end() - 3)) ? 1 : -1;
}

double calculate_average_momentum(const std::vector<int>& momentum_values) {
    if (momentum_values.size() < 10) {
        throw std::invalid_argument("Insufficient data for 10-day momentum average.");
    }

    return std::accumulate(momentum_values.end() - 10, momentum_values.end(), 0.0) / 10.0;
}

// Function to calculate percent change in 20-day volatility
double calculate_volatility_pct_change(const std::vector<double>& ratio_values) {
    if (ratio_values.size() < 21) {
        throw std::invalid_argument("Insufficient data for 20-day volatility calculation.");
    }

    double current_volatility = volatility_calculate(std::vector<double>(ratio_values.end() - 20, ratio_values.end()), 20);
    double prior_volatility = volatility_calculate(std::vector<double>(ratio_values.end() - 21, ratio_values.end() - 1), 20);

    if (prior_volatility == 0) {
        throw std::invalid_argument("Prior volatility is zero. Division by zero error.");
    }

    return ((current_volatility - prior_volatility) / prior_volatility) * 100.0;
}

// Function to calculate ratio and statistics for each combination
std::vector<std::pair<std::map<std::string, double>, std::map<std::string, std::string>>> calculate_for_combinations(const std::vector<std::map<std::string, std::string>>& data, const std::string& ticker1, const std::string& ticker_B, const std::string& ticker_C, const std::string& ticker_D, const std::string& start_date_h, const std::string& end_date_h, const std::string& start_date_s, const std::string& end_date_s, double correlation_diff) 
{

    auto merged_data_h = create_ticker_price_df(data, {ticker1, ticker_B, ticker_C, ticker_D}, start_date_h, end_date_h);
    auto merged_data_s = create_ticker_price_df(data, {ticker1, ticker_B, ticker_C, ticker_D}, start_date_s, end_date_s);
    //std::cout << "190" << std::endl;

    for (auto& row_h : merged_data_h) {
        row_h["Ratio"] = (row_h[ticker1] * row_h[ticker_B]) / (row_h[ticker_C] * row_h[ticker_D]);
    }
    for (auto& row_s : merged_data_s) {
        row_s["Ratio"] = (row_s[ticker1] * row_s[ticker_B]) / (row_s[ticker_C] * row_s[ticker_D]);
    }

    double end_date_ratio = merged_data_h.back()["Ratio"];
    double previous_day_ratio = merged_data_h[merged_data_h.size() - 2]["Ratio"];
    double predict_date_ratio = merged_data_s.back()["Ratio"];
    double end_date_ratio_pct_change_10 = ((merged_data_s.back()["Ratio"] - merged_data_s.front()["Ratio"]) / merged_data_s.front()["Ratio"])*100;
    double end_date_ratio_pct_change = ((merged_data_s[1]["Ratio"] - merged_data_s.front()["Ratio"]) / merged_data_s.front()["Ratio"])*100;

    double ten_days_before_ratio = merged_data_h[merged_data_h.size() - 11]["Ratio"];
    double percentage_change = ((end_date_ratio - ten_days_before_ratio) / ten_days_before_ratio)*100;


    std::vector<double> ratio_values;
    for (const auto& row : merged_data_h) {
    ratio_values.push_back(row.at("Ratio"));
    }



    double last_250_day_volatility = volatility_calculate(ratio_values, ratio_values.size());
    double last_day_volatility_250 = last_250_day_volatility;
    double last_20_day_volatility = volatility_calculate(std::vector<double>(ratio_values.end() - 20, ratio_values.end()), 20);
    double last_day_volatility_20 = last_20_day_volatility;

    std::vector<std::pair<std::string, double>> transformed_data;
    for (const auto& row : merged_data_h) {
    auto it = row.find("Ratio");
    if (it != row.end()) {
        transformed_data.emplace_back(it->first, it->second);
    }
    }


    // Pass the transformed data to get_percentiles
    auto [ratio_0_percentile, ratio_1_percentile, ratio_99_percentile, ratio_100_percentile, avg] = get_percentiles(transformed_data);

    double count_low = end_date_ratio==ratio_0_percentile?1:0;
    double count_high = end_date_ratio==ratio_100_percentile?1:0;

    double dist_from_low = (end_date_ratio - ratio_0_percentile) / (ratio_0_percentile)*100;
    double dist_from_high = (ratio_100_percentile - end_date_ratio) / (ratio_100_percentile)*100;

    double ratio_variance = calculate_ratio_variance(transformed_data, ticker1, ticker_B, ticker_C, ticker_D);

    double end_date_ratio_hist_avg = end_date_ratio / avg;

    std::string tag_rv;
    if (ratio_variance <= 15) {
        tag_rv = "Less Than 15";
    } else if (ratio_variance > 15 && ratio_variance <= 20) {
        tag_rv = "Between 15 and 20";
    } else if (ratio_variance > 20 && ratio_variance < 30) {
        tag_rv = "Between 20 and 30";
    } else {
        tag_rv = "More Than 30";
    }
    //std::cout << "248" << std::endl;
    std::vector<std::pair<std::string, double>> transformed_data_s;
    for (const auto& row : merged_data_s) {
    auto it = row.find("Ratio");
    if (it != row.end()) {
        transformed_data_s.emplace_back(it->first, it->second);
    }
    }
    
    //5d RHA
    double avg_5_day = calculate_5_day_avg(transformed_data);
    double end_date_ratio_5_day_avg = end_date_ratio / avg_5_day;
    //20 DRHA
    double avg_20_day = calculate_20_day_avg(transformed_data);
    double end_date_ratio_20_day_avg = end_date_ratio / avg_20_day;


    std::string date = std::to_string(merged_data_h.back().at("Date"));

double pct_change_ticker1 = calculate_pct_change(
    merged_data_h.back().at(ticker1),
    merged_data_h[merged_data_h.size() - 2].at(ticker1),
    ticker1,
    date
);

double pct_change_ticker_B = calculate_pct_change(
    merged_data_h.back().at(ticker_B),
    merged_data_h[merged_data_h.size() - 2].at(ticker_B),
    ticker_B,
    date
);

double pct_change_ticker_C = calculate_pct_change(
    merged_data_h.back().at(ticker_C),
    merged_data_h[merged_data_h.size() - 2].at(ticker_C),
    ticker_C,
    date
);

double pct_change_ticker_D = calculate_pct_change(
    merged_data_h.back().at(ticker_D),
    merged_data_h[merged_data_h.size() - 2].at(ticker_D),
    ticker_D,
    date
);

    std::vector<double> pct_changes = {pct_change_ticker1, pct_change_ticker_B, pct_change_ticker_C, pct_change_ticker_D};
    std::sort(pct_changes.begin(), pct_changes.end(), std::greater<double>()); // Sort descending
    double pct_change_diff = pct_changes.front() - pct_changes.back();


    
    // Calculate the 5-day percent changes for the 4 tickers
    double pct_change_5d_ticker1 = calculate_pct_change_5_day(merged_data_h.back()[ticker1], merged_data_h[merged_data_h.size() - 6][ticker1]);
    double pct_change_5d_ticker_B = calculate_pct_change_5_day(merged_data_h.back()[ticker_B], merged_data_h[merged_data_h.size() - 6][ticker_B]);
    double pct_change_5d_ticker_C = calculate_pct_change_5_day(merged_data_h.back()[ticker_C], merged_data_h[merged_data_h.size() - 6][ticker_C]);
    double pct_change_5d_ticker_D = calculate_pct_change_5_day(merged_data_h.back()[ticker_D], merged_data_h[merged_data_h.size() - 6][ticker_D]);
    std::vector<double> pct_changes_5d = {pct_change_5d_ticker1, pct_change_5d_ticker_B, pct_change_5d_ticker_C, pct_change_5d_ticker_D};
    std::sort(pct_changes_5d.begin(), pct_changes_5d.end(), std::greater<double>()); // Sort descending
    double pct_change_5d_diff = pct_changes_5d.front() - pct_changes_5d.back();



double drawdown_250 = calculate_drawdown(ratio_values);

// Calculate daily momentum indicators
std::vector<int> momentum_indicators;
for (size_t i = 2; i < ratio_values.size(); ++i) {
    momentum_indicators.push_back(calculate_momentum_indicator(std::vector<double>(ratio_values.begin(), ratio_values.begin() + i + 1)));
}

// Calculate average momentum for the last 10 days
double avg_momentum_10 = calculate_average_momentum(momentum_indicators);

// Calculate percent change in 20-day volatility
double pct_change_volatility_20 = calculate_volatility_pct_change(ratio_values);

// Add the new variables to numeric_map


// Assuming 'Date' is a custom type, use it accordingly
struct Date {
    int day, month, year;
};

// Define the type for numeric and string maps
// Define the type for numeric and string maps
using NumericMap = std::map<std::string, double>;
using StringMap = std::map<std::string, std::string>;

// Define the results vector as a vector of pairs of maps
std::vector<std::pair<NumericMap, StringMap>> results;
// Map for numeric values
std::map<std::string, double> numeric_map;
numeric_map["Ratio_Average_5_Years"] = avg;
numeric_map["Ratio_0_Percentile"] = ratio_0_percentile;
numeric_map["Ratio_1_Percentile"] = ratio_1_percentile;
numeric_map["Ratio_99_Percentile"] = ratio_99_percentile;
numeric_map["Ratio_100_Percentile"] = ratio_100_percentile;
numeric_map["Ratio_Variance"] = ratio_variance;
numeric_map["Ratio"] = end_date_ratio;
numeric_map["Ratio_pct_change"] = end_date_ratio_pct_change;
numeric_map["Ratio/hist_avg"] = end_date_ratio_hist_avg;
numeric_map["10 Day Percent Change (%)"] = percentage_change;
numeric_map["last_day_volatility_250 (%)"] = last_day_volatility_250;
numeric_map["last_day_volatility_20 (%)"] = last_day_volatility_20;
numeric_map["Distance from Low (%)"] = dist_from_low;
numeric_map["Distance from High (%)"] = dist_from_high;
numeric_map["Ratio_Next_day"] = predict_date_ratio;
numeric_map["Next_10_Day_Ratio_pct_change"] = end_date_ratio_pct_change_10;
numeric_map["Count_20_High"] = count_high;
numeric_map["Count_20_Low"] = count_low;
numeric_map["Ratio_5_Day_Avg"] = avg_5_day;
numeric_map["Ratio/hist_5_Day_Avg"] = end_date_ratio_5_day_avg;
numeric_map["Ratio/hist_20_Day_Avg"] = end_date_ratio_20_day_avg;
numeric_map["Ratio Comp 1D"] = pct_change_diff;
numeric_map["Ratio Comp 5D"] = pct_change_5d_diff;
numeric_map["Drawdown_250 (%)"] = drawdown_250;
numeric_map["Momentum_Indicator_Avg_10"] = avg_momentum_10;
numeric_map["Pct_Change_20d_Volatility(%)"] = pct_change_volatility_20;


// Map for string values
std::map<std::string, std::string> string_map;
string_map["Ticker1"] = ticker1;
string_map["Ticker_B"] = ticker_B;
string_map["Ticker_C"] = ticker_C;
string_map["Ticker_D"] = ticker_D;
string_map["Date"] = start_date_s; // Assuming you can convert 'Date' to string or store separately

// Map for 'tag_rv' as string
string_map["Ratio_Variance_Tag"] = tag_rv;
// Example of storing both maps in a result vector (if needed)
results.push_back(std::make_pair(numeric_map, string_map));




    return results;
}


int main() {
    std::cout << "1" << std::endl;
    auto data = load_data_from_sqlite();
    std::cout << "1.5" << std::endl;
    auto filtered_data = prefilter_data(data);

    std::string end_date_h = "2020-09-09";
    std::string final_end_date_s = "2020-12-30";
    // std::string end_date_h = "2023-09-11";
    // std::string final_end_date_s = "2024-09-10";
    double correlation_diff = 0.0;

    //std::ifstream file("inputs/all_ticker_split_10.csv");
    std::ifstream file("inputs/all_tickers_1621_t6_batch2-2.csv");
    std::vector<std::string> headers = {"Ticker1", "Ticker_B", "Ticker_C", "Ticker_D"};
    std::vector<std::map<std::string, std::string>> df_all;


    std::string line, word;
    while (std::getline(file, line)) {
        std::stringstream s(line);
        std::map<std::string, std::string> row;
        int col_index = 0;
        while (std::getline(s, word, ',')) {
            if (col_index < headers.size()) {
                row[headers[col_index]] = word;
                col_index++;
            }
        }
        df_all.push_back(row);
    }
    file.close();

    // Open output file in append mode and write the headers initially
    std::ofstream outfile("outputs/insample_1621_t6_batch2-2-leftover.csv");
    outfile << "Date,Ticker1,Ticker_B,Ticker_C,Ticker_D,Ratio_Average_5_Years,"
            << "Ratio_0_Percentile,Ratio_1_Percentile,Ratio_99_Percentile,Ratio_100_Percentile,Ratio_Variance,"
            << "Ratio_Variance_Tag,Ratio,Ratio_pct_change,Ratio/hist_avg,10 Day Percent Change (%),"
            << "last_day_volatility_250 (%),last_day_volatility_20 (%),Distance from Low (%),Distance from High (%),Ratio_Next_day,Next_10_Day_Ratio_pct_change,Count_20_High,Count_20_Low,Ratio/hist_5_Day_Avg,Ratio/hist_20_Day_Avg,Ratio Comp 1D,Ratio Comp 5D,"
            << "Drawdown_250 (%),Momentum_Indicator_Avg_10,Pct_Change_20d_Volatility(%)\n";
    outfile.close();
           

    while (end_date_h <= final_end_date_s) {
        size_t end_date_index = 0;
        for (size_t i = 0; i < filtered_data.size(); ++i) {
            if (filtered_data[i]["Date"] == end_date_h) {     
                end_date_index = i;
                break;
            }
        }

        if (end_date_index >= 250) {
            std::string start_date_h = filtered_data[end_date_index - 250]["Date"];
            std::string next_date = (end_date_index + 10 < filtered_data.size()) ? filtered_data[end_date_index + 10]["Date"] : "";
            std::string start_date_s = end_date_h;
            std::string end_date_s = next_date;

            std::cout << "End Date: " << end_date_h << std::endl;
             std::cout << "250th Day Prior: " << start_date_h << std::endl;
             std::cout << "Next Date 10: " << next_date << std::endl;
             std::cout << "----------" << std::endl;

            bool is_header = true;
            for (const auto& row : df_all) {
                if (is_header) {
                    is_header = false;
                    continue;
                }

                auto ticker1_iter = row.find("Ticker1");
                auto ticker_B_iter = row.find("Ticker_B");
                auto ticker_C_iter = row.find("Ticker_C");
                auto ticker_D_iter = row.find("Ticker_D");

                if (ticker1_iter != row.end() && ticker_B_iter != row.end() && ticker_C_iter != row.end() && ticker_D_iter != row.end()) {
                    std::string ticker1 = ticker1_iter->second;
                    std::string ticker_B = ticker_B_iter->second;
                    std::string ticker_C = ticker_C_iter->second;
                    std::string ticker_D = ticker_D_iter->second;

                    auto result_df = calculate_for_combinations(filtered_data, ticker1, ticker_B, ticker_C, ticker_D, start_date_h, end_date_h, start_date_s, end_date_s, correlation_diff);

                    // Append results to the CSV file
                    std::ofstream outfile("outputs/insample_1621_t6_batch2-2-leftover.csv", std::ios_base::app);
                    for (const auto& result_row : result_df) {
                        const auto& numeric_map = result_row.first;
                        const auto& string_map = result_row.second;

                        auto date_iter = string_map.find("Date");
                        auto ticker1_iter = string_map.find("Ticker1");
                        auto ticker_B_iter = string_map.find("Ticker_B");
                        auto ticker_C_iter = string_map.find("Ticker_C");
                        auto ticker_D_iter = string_map.find("Ticker_D");
                        auto ratio_avg_5_iter = numeric_map.find("Ratio_Average_5_Years");
                        auto ratio_0_iter = numeric_map.find("Ratio_0_Percentile");
                        auto ratio_1_iter = numeric_map.find("Ratio_1_Percentile");
                        auto ratio_99_iter = numeric_map.find("Ratio_99_Percentile");
                        auto ratio_100_iter = numeric_map.find("Ratio_100_Percentile");
                        auto ratio_variance_iter = numeric_map.find("Ratio_Variance");
                        auto ratio_variance_tag_iter = string_map.find("Ratio_Variance_Tag");
                        auto ratio_iter = numeric_map.find("Ratio");
                        auto ratio_pct_change_iter = numeric_map.find("Ratio_pct_change");
                        auto ratio_hist_avg_iter = numeric_map.find("Ratio/hist_avg");
                        auto pct_change_iter = numeric_map.find("10 Day Percent Change (%)");
                        auto vol_250_iter = numeric_map.find("last_day_volatility_250 (%)");
                        auto vol_20_iter = numeric_map.find("last_day_volatility_20 (%)");
                        auto dist_low_iter = numeric_map.find("Distance from Low (%)");
                        auto dist_high_iter = numeric_map.find("Distance from High (%)");
                        auto ratio_next_day_iter = numeric_map.find("Ratio_Next_day");
                        auto ratio_10_pct_iter = numeric_map.find("Next_10_Day_Ratio_pct_change");
                        auto count_20_high_iter = numeric_map.find("Count_20_High");
                        auto count_20_low_iter = numeric_map.find("Count_20_Low");
                        auto ratio_hist_5d_iter = numeric_map.find("Ratio/hist_5_Day_Avg");
                        auto ratio_hist_20d_iter = numeric_map.find("Ratio/hist_20_Day_Avg");
                        auto pct_change_diff_iter = numeric_map.find("Ratio Comp 1D");
                        auto pct_change_5d_diff_iter = numeric_map.find("Ratio Comp 5D");
                        auto drawdown_250_iter = numeric_map.find("Drawdown_250 (%)");
                        auto avg_momentum_10_iter = numeric_map.find("Momentum_Indicator_Avg_10");
                        auto pct_change_volatility_20_iter = numeric_map.find("Pct_Change_20d_Volatility(%)");

                
                        if (date_iter != string_map.end() && ticker1_iter != string_map.end() && ticker_B_iter != string_map.end() &&
                            ticker_C_iter != string_map.end() && ticker_D_iter != string_map.end() && ratio_avg_5_iter != numeric_map.end() &&
                            ratio_0_iter != numeric_map.end() && ratio_1_iter != numeric_map.end() && ratio_99_iter != numeric_map.end() &&
                            ratio_100_iter != numeric_map.end() && ratio_variance_iter != numeric_map.end() && 
                            ratio_variance_tag_iter != string_map.end() && ratio_iter != numeric_map.end() &&
                            ratio_pct_change_iter != numeric_map.end() && ratio_hist_avg_iter != numeric_map.end() &&
                            pct_change_iter != numeric_map.end() && vol_250_iter != numeric_map.end() && vol_20_iter != numeric_map.end() &&
                            dist_low_iter != numeric_map.end() && dist_high_iter != numeric_map.end() && ratio_next_day_iter != numeric_map.end() && ratio_10_pct_iter != numeric_map.end() && 
                            count_20_high_iter != numeric_map.end() && count_20_low_iter != numeric_map.end() && ratio_hist_5d_iter != numeric_map.end() && ratio_hist_20d_iter != numeric_map.end() &&
                            pct_change_diff_iter != numeric_map.end() && pct_change_5d_diff_iter != numeric_map.end() &&
                            drawdown_250_iter != numeric_map.end() && avg_momentum_10_iter != numeric_map.end() && pct_change_volatility_20_iter != numeric_map.end()) {

                            outfile << date_iter->second << "," << ticker1_iter->second << "," << ticker_B_iter->second << "," 
                                    << ticker_C_iter->second << "," << ticker_D_iter->second << ","
                                    << ratio_avg_5_iter->second << "," << ratio_0_iter->second << "," << ratio_1_iter->second << ","
                                    << ratio_99_iter->second << "," << ratio_100_iter->second << "," << ratio_variance_iter->second << ","
                                    << ratio_variance_tag_iter->second << "," << ratio_iter->second << "," << ratio_pct_change_iter->second << ","
                                    << ratio_hist_avg_iter->second << "," << pct_change_iter->second << "," << vol_250_iter->second << ","
                                    << vol_20_iter->second << "," << dist_low_iter->second << "," << dist_high_iter->second << ","
                                    << ratio_next_day_iter->second << "," << ratio_10_pct_iter->second << "," << count_20_high_iter->second << "," << count_20_low_iter->second <<","
                                    << ratio_hist_5d_iter->second << "," << ratio_hist_20d_iter->second << "," << pct_change_diff_iter->second << "," << pct_change_5d_diff_iter->second << ","
                                    << drawdown_250_iter->second << "," << avg_momentum_10_iter->second << "," << pct_change_volatility_20_iter->second << "\n";
                        }
             
                        else 
                        {
                            std::cerr << "Error: Missing keys in result_row!" << std::endl;
                        }
                    }
                    outfile.close();
                } else {
                    std::cerr << "Error: Missing keys in row!" << std::endl;
                }
            }
        } else {
            std::cout << "Not enough data to go back 250 business days from " << end_date_h << "." << std::endl;
        }

        end_date_index++;
        if (end_date_index < filtered_data.size()) {
            end_date_h = filtered_data[end_date_index]["Date"];
        } else {
            break;
        }
    }

    return 0;
}
