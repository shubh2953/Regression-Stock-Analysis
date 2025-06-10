#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <time.h>

#define min(a,b) ((a) < (b) ? (a) : (b))
#define MAX_LINE_LENGTH 8192
#define LEARNING_RATE 0.1
#define MAX_ITERATIONS 100
#define BATCH_SIZE 32
#define CONVERGENCE_THRESHOLD 1e-4
#define DEFAULT_INPUT_FILENAME "regression_input.csv"  // Default input filename
#define DEFAULT_TRAINING_WINDOW 100  // Default training window size
#define DEFAULT_REGRESSION_TYPE LINEAR_REGRESSION  // Default regression type
#define EPSILON 1e-10



// Add regression type enum
typedef enum {
    LINEAR_REGRESSION,
    LOGISTIC_REGRESSION
} RegressionType;


// Function to execute Python script
int execute_python_script(const char* gains_csv, const char* output_png) {
    char command[1024];
    
    // Construct command to run Python script
    snprintf(command, sizeof(command), 
             "python gains_chart.py \"%s\" \"%s\"", 
             gains_csv, output_png);
    
    // Execute command
    return system(command);
}



int execute_python_script_2(const char* nav_csv, const char* output_png, const char* output_metrics_csv) {
    char command[1024];
    
    // Construct command to run Python script
    snprintf(command, sizeof(command), 
             "python nav_plot.py \"%s\" \"%s\" \"%s\"", 
             nav_csv, output_png, output_metrics_csv);
    
    // Execute command
    return system(command);
}

typedef struct {
    double* features;
    double target;
    char* ticker_concat;
    char date[11];  // Add date field
} DataPoint;

typedef struct {
    DataPoint* data;
    int total_samples;
    int num_features;
    int date_column_present;
    int num_ticker_columns;  // Number of ticker columns
    const char* input_filename;
} Dataset;
typedef struct {
    char date[11];  // DD-MM-YYYY format
    double investment;
    double gain;
    double normalized_gain;
    double summed_gain;
    double nav;
    double drawdown;
    double max_drawdown;
    double annual_return;
    double ar_md_ratio;
    double dpv_new;
} NAVMetrics;

typedef struct {
    double predicted;
    double target;
    int target_label;
    double row_weight;
    double cumulative_gain;
    double random_line;
    double auc;
} GainsRow;


void multiply_matrix(double* A, double* B, double* C, int m, int n, int p);
void transpose_matrix(double* A, double* B, int m, int n);
void inverse_matrix(double* A, int n);
void train_model_ols(DataPoint* training_data, int start_idx, int num_samples, int num_features, double* weights);
void train_model_logistic(DataPoint* training_data, int start_idx, int num_samples, int num_features, double* weights);
double predict(double* features, double* weights, int num_features, RegressionType reg_type);
double transform_dpv(double dpv, double cutoff);
double transform_dpv(double dpv, double cutoff) {
    if (dpv < -cutoff) return 0.0;
    if (dpv > cutoff) return 1.0;
    return 0.5;
}


int double_compare(const void* a, const void* b) {
    double diff = *(const double*)a - *(const double*)b;
    return (diff > 0) - (diff < 0);
}

// Function to get median for a specific feature on a specific date
double get_median_for_date(DataPoint* data, int total_samples, const char* date, int feature_index) {
    double* values = malloc(total_samples * sizeof(double));
    int count = 0;
    
    // First try to get values from the same date
    for (int i = 0; i < total_samples; i++) {
        if (strcmp(data[i].date, date) == 0 && !isnan(data[i].features[feature_index])) {
            values[count++] = data[i].features[feature_index];
        }
    }
    
    // If no values found for this date, use all non-NaN values
    if (count == 0) {
        for (int i = 0; i < total_samples; i++) {
            if (!isnan(data[i].features[feature_index])) {
                values[count++] = data[i].features[feature_index];
            }
        }
    }
    
    double median = 0.0;
    if (count > 0) {
        qsort(values, count, sizeof(double), double_compare);
        if (count % 2 == 0) {
            median = (values[count/2 - 1] + values[count/2]) / 2.0;
        } else {
            median = values[count/2];
        }
    }
    
    free(values);
    return median;
}

// Complete updated read_csv function
Dataset read_csv(const char* filename) {
    Dataset dataset = {NULL, 0, 0, 0, 4, filename};
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file\n");
        return dataset;
    }

    char line[MAX_LINE_LENGTH];
    char first_data_line[MAX_LINE_LENGTH];
    
    // Read header line
    if (!fgets(line, MAX_LINE_LENGTH, file)) {
        printf("Error reading header\n");
        fclose(file);
        return dataset;
    }
    
    // Read first data line to check for date
    if (!fgets(first_data_line, MAX_LINE_LENGTH, file)) {
        printf("Error reading first data line\n");
        fclose(file);
        return dataset;
    }

    // Count columns
    int num_columns = 0;
    char* header_copy = strdup(line);
    char* token = strtok(header_copy, ",");
    while (token) {
        num_columns++;
        token = strtok(NULL, ",");
    }
    free(header_copy);

    dataset.num_features = num_columns - dataset.num_ticker_columns - 2;

    // Count total lines
    rewind(file);
    fgets(line, MAX_LINE_LENGTH, file);  // Skip header
    int total_lines = 0;
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        total_lines++;
    }

    // Allocate memory
    dataset.data = malloc(total_lines * sizeof(DataPoint));
    dataset.total_samples = total_lines;

    for (int i = 0; i < total_lines; i++) {
        dataset.data[i].features = malloc((dataset.num_features + 1) * sizeof(double));
        dataset.data[i].ticker_concat = malloc(50 * sizeof(char));
        memset(dataset.data[i].date, 0, 11);
    }

    // Reset file pointer and skip header
    rewind(file);
    fgets(line, MAX_LINE_LENGTH, file);

    // Read data
    int row = 0;
    while (fgets(line, MAX_LINE_LENGTH, file) && row < total_lines) {
        char* line_copy = strdup(line);
        char* token = strtok(line_copy, ",");
        
        // Store date
        strncpy(dataset.data[row].date, token, 10);
        dataset.data[row].date[10] = '\0';
        
        // Process ticker columns
        char ticker_concat[50] = "";
        for (int i = 0; i < dataset.num_ticker_columns; i++) {
            token = strtok(NULL, ",");
            strcat(ticker_concat, token);
        }
        strcpy(dataset.data[row].ticker_concat, ticker_concat);

        // Process feature columns
        int col = 0;
        while (col < dataset.num_features && (token = strtok(NULL, ","))) {
            if (strlen(token) == 0 || strcmp(token, "NA") == 0 || strcmp(token, "nan") == 0) {
                dataset.data[row].features[col] = NAN;
            } else {
                dataset.data[row].features[col] = atof(token);
            }
            col++;
        }

        // Process target value
        if ((token = strtok(NULL, ","))) {
            dataset.data[row].target = atof(token);
        }

        dataset.data[row].features[dataset.num_features] = 1.0;  // Bias term
        free(line_copy);
        row++;
    }

    // Handle missing values and add epsilon
    for (int i = 0; i < dataset.total_samples; i++) {
        for (int j = 0; j < dataset.num_features; j++) {
            if (isnan(dataset.data[i].features[j])) {
                dataset.data[i].features[j] = get_median_for_date(
                    dataset.data, dataset.total_samples, 
                    dataset.data[i].date, j
                );
            }
            dataset.data[i].features[j] += EPSILON;
        }
    }

    fclose(file);
    return dataset;
}

// Complete updated inverse_matrix function
void inverse_matrix(double* A, int n) {
    // Add small values to diagonal for stability
    for (int i = 0; i < n; i++) {
        A[i * n + i] += EPSILON;
    }

    double* augmented = malloc(n * 2 * n * sizeof(double));
    if (!augmented) {
        printf("Memory allocation failed in matrix inversion\n");
        return;
    }

    // Create augmented matrix [A|I]
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            augmented[i * (2 * n) + j] = A[i * n + j];
            augmented[i * (2 * n) + (j + n)] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Gauss-Jordan elimination with partial pivoting
    for (int i = 0; i < n; i++) {
        // Find maximum pivot in current column
        int max_row = i;
        double max_val = fabs(augmented[i * (2 * n) + i]);
        
        for (int k = i + 1; k < n; k++) {
            double curr_val = fabs(augmented[k * (2 * n) + i]);
            if (curr_val > max_val) {
                max_val = curr_val;
                max_row = k;
            }
        }

        // Swap rows if necessary
        if (max_row != i) {
            for (int j = 0; j < 2 * n; j++) {
                double temp = augmented[i * (2 * n) + j];
                augmented[i * (2 * n) + j] = augmented[max_row * (2 * n) + j];
                augmented[max_row * (2 * n) + j] = temp;
            }
        }

        // Check for near-zero pivot
        double pivot = augmented[i * (2 * n) + i];
        if (fabs(pivot) < EPSILON) {
            pivot = (pivot >= 0) ? EPSILON : -EPSILON;
            augmented[i * (2 * n) + i] = pivot;
        }

        // Normalize row
        for (int j = 0; j < 2 * n; j++) {
            augmented[i * (2 * n) + j] /= pivot;
        }

        // Eliminate column
        for (int k = 0; k < n; k++) {
            if (k != i) {
                double factor = augmented[k * (2 * n) + i];
                for (int j = 0; j < 2 * n; j++) {
                    augmented[k * (2 * n) + j] -= factor * augmented[i * (2 * n) + j];
                }
            }
        }
    }

    // Copy result back to A
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = augmented[i * (2 * n) + (j + n)];
            
            // Check for numerical instability
            if (isnan(A[i * n + j]) || isinf(A[i * n + j])) {
                A[i * n + j] = 0.0;
                printf("Warning: Numerical instability detected in matrix inversion\n");
            }
        }
    }

    free(augmented);
}

// Complete updated train_model_ols function
void train_model_ols(DataPoint* training_data, int start_idx, int num_samples, int num_features, double* weights) {
    double* X = malloc(num_samples * (num_features + 1) * sizeof(double));
    double* y = malloc(num_samples * sizeof(double));
    double* X_transpose = malloc((num_features + 1) * num_samples * sizeof(double));
    double* X_transpose_X = malloc((num_features + 1) * (num_features + 1) * sizeof(double));
    double* X_transpose_y = malloc((num_features + 1) * sizeof(double));

    if (!X || !y || !X_transpose || !X_transpose_X || !X_transpose_y) {
        printf("Memory allocation failed in OLS training\n");
        goto cleanup;
    }

    // Initialize matrices with regularization
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < (num_features + 1); j++) {
            double feature_val = training_data[start_idx + i].features[j];
            // Add small noise to prevent perfect collinearity
            X[i * (num_features + 1) + j] = feature_val + ((j < num_features) ? EPSILON : 0);
        }
        y[i] = training_data[start_idx + i].target;
    }

    // Compute X^T
    transpose_matrix(X, X_transpose, num_samples, num_features + 1);

    // Compute X^T * X with regularization
    multiply_matrix(X_transpose, X, X_transpose_X, num_features + 1, num_samples, num_features + 1);
    
    // Add ridge regularization term
    for (int i = 0; i < num_features + 1; i++) {
        X_transpose_X[i * (num_features + 1) + i] += EPSILON;
    }

    // Compute X^T * y
    multiply_matrix(X_transpose, y, X_transpose_y, num_features + 1, num_samples, 1);

    // Compute inverse and final weights
    inverse_matrix(X_transpose_X, num_features + 1);
    multiply_matrix(X_transpose_X, X_transpose_y, weights, num_features + 1, num_features + 1, 1);

    // Check for numerical instability in weights
    for (int i = 0; i < num_features + 1; i++) {
        if (isnan(weights[i]) || isinf(weights[i])) {
            printf("Warning: Numerical instability detected in weight %d, setting to 0\n", i);
            weights[i] = 0.0;
        }
    }

cleanup:
    free(X);
    free(y);
    free(X_transpose);
    free(X_transpose_X);
    free(X_transpose_y);
}

double sigmoid(double z) {
    
    return 1.0 / (1.0 + exp(-z));
}


void multiply_vector_matrix(const double* v, const double* M, double* result, int rows, int cols) {
    for (int j = 0; j < cols; j++) {
        result[j] = 0.0;
        for (int i = 0; i < rows; i++) {
            result[j] += v[i] * M[i * cols + j];
        }
    }
}

void multiply_matrix_vector(const double* M, const double* v, double* result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        result[i] = 0.0;
        for (int j = 0; j < cols; j++) {
            result[i] += M[i * cols + j] * v[j];
        }
    }
}

void train_model_logistic(DataPoint* training_data, int start_idx, int num_samples, int num_features, double* weights) {
    const int max_iter = 10;  // Reduced iterations since we're using direct matrix operations
    const double tolerance = 1e-6;
    const int features_plus_bias = num_features + 1;
    
    // Allocate memory for matrices
    double* X = malloc(num_samples * features_plus_bias * sizeof(double));
    double* y = malloc(num_samples * sizeof(double));
    double* prob = malloc(num_samples * sizeof(double));
    double* W = malloc(num_samples * sizeof(double));  // Diagonal weight matrix
    double* X_transpose = malloc(features_plus_bias * num_samples * sizeof(double));
    double* X_transpose_W = malloc(features_plus_bias * num_samples * sizeof(double));
    double* X_transpose_W_X = malloc(features_plus_bias * features_plus_bias * sizeof(double));
    double* X_transpose_W_z = malloc(features_plus_bias * sizeof(double));
    
    // Initialize weights to small random values
    for (int i = 0; i < features_plus_bias; i++) {
        weights[i] = (rand() / (double)RAND_MAX - 0.5) * 0.01;
    }
    
    // Build design matrix X and transform targets
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < num_features; j++) {
            // Add small epsilon to features for stability
            X[i * features_plus_bias + j] = training_data[start_idx + i].features[j] + EPSILON;
        }
        X[i * features_plus_bias + num_features] = 1.0 + EPSILON;  // Bias term
        y[i] = transform_dpv(training_data[start_idx + i].target, 0.005);
    }
    
    // IRLS iterations
    for (int iter = 0; iter < max_iter; iter++) {
        // Calculate probabilities and weights
        double max_diff = 0.0;
        
        for (int i = 0; i < num_samples; i++) {
            double z = 0;
            for (int j = 0; j < features_plus_bias; j++) {
                z += X[i * features_plus_bias + j] * weights[j];
            }
            
            // Clip z to prevent overflow
            z = fmax(-20.0, fmin(20.0, z));
            prob[i] = sigmoid(z);
            
            // Add stability term to weights
            W[i] = fmax(prob[i] * (1 - prob[i]), EPSILON);
            
            // Calculate working response
            double old_weight = weights[0];  // Store for convergence check
            double working_response = z + (y[i] - prob[i]) / W[i];
            
            // Store working response in y array for matrix operations
            y[i] = working_response;
            
            if (i == 0) max_diff = fabs(weights[0] - old_weight);
        }
        
        // Calculate X transpose
        transpose_matrix(X, X_transpose, num_samples, features_plus_bias);
        
        // Calculate X^T * W
        for (int i = 0; i < features_plus_bias; i++) {
            for (int j = 0; j < num_samples; j++) {
                X_transpose_W[i * num_samples + j] = X_transpose[i * num_samples + j] * W[j];
            }
        }
        
        // Calculate X^T * W * X with stability term
        multiply_matrix(X_transpose_W, X, X_transpose_W_X, 
                       features_plus_bias, num_samples, features_plus_bias);
        
        // Add ridge regularization and stability terms
        for (int i = 0; i < features_plus_bias; i++) {
            X_transpose_W_X[i * features_plus_bias + i] += EPSILON;
        }
        
        // Calculate X^T * W * z
        multiply_matrix(X_transpose_W, y, X_transpose_W_z,
                       features_plus_bias, num_samples, 1);
        
        // Solve system using matrix inversion
        inverse_matrix(X_transpose_W_X, features_plus_bias);
        multiply_matrix(X_transpose_W_X, X_transpose_W_z, weights,
                       features_plus_bias, features_plus_bias, 1);
        
        // Check for convergence
        if (max_diff < tolerance) {
            break;
        }
        
        // Check for NaN values and reset if necessary
        int has_nan = 0;
        for (int j = 0; j < features_plus_bias; j++) {
            if (isnan(weights[j])) {
                has_nan = 1;
                break;
            }
        }
        
        if (has_nan) {
            // Reset weights to small random values
            for (int j = 0; j < features_plus_bias; j++) {
                weights[j] = (rand() / (double)RAND_MAX - 0.5) * 0.01;
            }
            continue;
        }
    }
    
    // Clean up
    free(X);
    free(y);
    free(prob);
    free(W);
    free(X_transpose);
    free(X_transpose_W);
    free(X_transpose_W_X);
    free(X_transpose_W_z);
}

int is_date(const char* str) {
    // Skip leading whitespace
    while (isspace(*str)) str++;
    
    // Check length (should be exactly 10 characters for DD-MM-YYYY)
    if (strlen(str) != 10) return 0;
    
    // Check format DD-MM-YYYY
    for (int i = 0; i < 10; i++) {
        switch(i) {
            case 2:
            case 5:
                if (str[i] != '-') return 0;
                break;
            default:
                if (!isdigit(str[i])) return 0;
                break;
        }
    }
    
    // Extract and validate day, month, year
    int day = (str[0] - '0') * 10 + (str[1] - '0');
    int month = (str[3] - '0') * 10 + (str[4] - '0');
    int year = (str[6] - '0') * 1000 + (str[7] - '0') * 100 + 
               (str[8] - '0') * 10 + (str[9] - '0');
    
    // Basic date validation
    if (day < 1 || day > 31) return 0;
    if (month < 1 || month > 12) return 0;
    if (year < 1900 || year > 2100) return 0;
    
    return 1;
}


// Helper function to convert DD-MM-YYYY to a comparable value
static int date_to_int(const char* date) {
    int day = (date[0] - '0') * 10 + (date[1] - '0');
    int month = (date[3] - '0') * 10 + (date[4] - '0');
    int year = (date[6] - '0') * 1000 + (date[7] - '0') * 100 + 
               (date[8] - '0') * 10 + (date[9] - '0');
    return year * 10000 + month * 100 + day;
}

// Comparison function for qsort
static int compare_dates(const void* a, const void* b) {
    const NAVMetrics* metrics_a = (const NAVMetrics*)a;
    const NAVMetrics* metrics_b = (const NAVMetrics*)b;
    int date_a = date_to_int(metrics_a->date);
    int date_b = date_to_int(metrics_b->date);
    return date_a - date_b;  // Sort in ascending order
}
void multiply_matrix_scalar(double* matrix, double scalar, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] *= scalar;
    }
}

void add_matrices(double* A, double* B, double* result, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        result[i] = A[i] + B[i];
    }
}

void calculate_nav_metrics(const char* regression_output, const char* nav_output) {
    FILE* input = fopen(regression_output, "r");
    if (!input) {
        printf("Error opening regression output file\n");
        return;
    }

    char header[MAX_LINE_LENGTH];
    if (!fgets(header, MAX_LINE_LENGTH, input)) {
        printf("Error reading header\n");
        fclose(input);
        return;
    }

    // Count columns in header and detect regression type
    int total_cols = 1;
    int is_logistic = 0;
    for (char* ptr = header; *ptr; ptr++) {
        if (*ptr == ',') total_cols++;
        if (strstr(header, "DPV_new") != NULL) {
            is_logistic = 1;
        }
    }

    // Pre-allocate memory
    size_t capacity = 1000;
    size_t total_rows = 0;
    NAVMetrics* metrics = malloc(capacity * sizeof(NAVMetrics));
    if (!metrics) {
        printf("Memory allocation failed\n");
        fclose(input);
        return;
    }

    // Read data
    char line[MAX_LINE_LENGTH];
    while (fgets(line, MAX_LINE_LENGTH, input)) {
        // Reallocate if needed
        if (total_rows >= capacity) {
            capacity *= 2;
            NAVMetrics* temp = realloc(metrics, capacity * sizeof(NAVMetrics));
            if (!temp) {
                printf("Memory reallocation failed\n");
                free(metrics);
                fclose(input);
                return;
            }
            metrics = temp;
        }

        // Process line
        char* line_copy = strdup(line);
        char* token = strtok(line_copy, ",");
        
        // Store date
        strncpy(metrics[total_rows].date, token, 10);
        metrics[total_rows].date[10] = '\0';

        // Skip to target and predicted values
        double target = 0.0;
        double predicted = 0.0;
        int col = 1;
        
        token = strtok(NULL, ",");
        while (token && col < total_cols) {
            if (is_logistic) {
                // For logistic regression:
                // Target is third from last (before DPV_new and Predicted)
                if (col == total_cols - 3) {
                    target = atof(token);
                } else if (col == total_cols - 1) { // Predicted is last column
                    predicted = atof(token);
                }
            } else {
                // For linear regression:
                // Target is second from last (before Predicted)
                if (col == total_cols - 2) {
                    target = atof(token);
                } else if (col == total_cols - 1) { // Predicted is last column
                    predicted = atof(token);
                }
            }
            token = strtok(NULL, ",");
            col++;
        }

        if (is_logistic) {
            // Logistic regression: threshold at 0.5
            metrics[total_rows].investment = predicted > 0.5 ? 1.0 : -1.0;
        } else {
            // Linear regression: three-way threshold at ±0.005
            if (predicted > 0.005) {
                metrics[total_rows].investment = 1.0;
            } else if (predicted < -0.005) {
                metrics[total_rows].investment = -1.0;
            } else {
                metrics[total_rows].investment = 0.0;
            }
        }

        // Calculate gain metrics
        metrics[total_rows].gain = metrics[total_rows].investment * target;
        metrics[total_rows].normalized_gain = metrics[total_rows].gain * 0.01;
        
        free(line_copy);
        total_rows++;
    }

    // Sort by date
    qsort(metrics, total_rows, sizeof(NAVMetrics), compare_dates);

    // Calculate summed gains by date
    char current_date[11] = "";
    double current_sum = 0.0;
    int date_start = 0;
    
    for (size_t i = 0; i <= total_rows; i++) {
        if (i == total_rows || strcmp(metrics[i].date, current_date) != 0) {
            // Assign previous date's summed gain
            if (date_start < i && strcmp(current_date, "") != 0) {
                for (size_t j = date_start; j < i; j++) {
                    metrics[j].summed_gain = current_sum;
                }
            }
            
            if (i < total_rows) {
                strcpy(current_date, metrics[i].date);
                current_sum = metrics[i].normalized_gain;
                date_start = i;
            }
        } else {
            current_sum += metrics[i].normalized_gain;
        }
    }

    // Calculate NAV and drawdown - Modified to use first day's summed gain
    double nav = 1.0 + metrics[0].summed_gain; // Initialize with first day's gain
    double max_nav = nav;
    int unique_dates = 0;
    strcpy(current_date, "");

    // First date's NAV includes the first day's gain
    for (size_t i = 0; i < total_rows && (i == 0 || strcmp(metrics[i].date, metrics[0].date) == 0); i++) {
        metrics[i].nav = nav;
        metrics[i].drawdown = 0.0;
    }

    // Calculate for remaining dates
    for (size_t i = 0; i < total_rows; i++) {
        if (strcmp(metrics[i].date, current_date) != 0) {
            unique_dates++;
            strcpy(current_date, metrics[i].date);
            
            if (i > 0) { // Skip first date as it's already handled
                nav *= (1.0 + metrics[i].summed_gain);
                max_nav = fmax(nav, max_nav);
            }

            // Set values for all rows of this date
            size_t j = i;
            while (j < total_rows && strcmp(metrics[j].date, current_date) == 0) {
                metrics[j].nav = nav;
                metrics[j].drawdown = (max_nav - nav) / max_nav;
                j++;
            }
        }
    }

    // Calculate final metrics
    double days_elapsed = unique_dates;
    double annual_return = pow(nav, 250.0/days_elapsed) - 1.0;
    
    double max_drawdown = 0.0;
    for (size_t i = 0; i < total_rows; i++) {
        max_drawdown = fmax(max_drawdown, metrics[i].drawdown);
    }
    
    double ar_md_ratio = (max_drawdown > 0) ? annual_return / max_drawdown : 0.0;

    // Write results
    FILE* output = fopen(nav_output, "w");
    if (!output) {
        printf("Error creating NAV metrics output file\n");
        free(metrics);
        fclose(input);
        return;
    }

    fprintf(output, "Date,Investment,Gain,Normalized_Gain,Summed_Gain,NAV,Drawdown,Max_Drawdown,Annual_Return,AR_MD_Ratio\n");
    
    for (size_t i = 0; i < total_rows; i++) {
        fprintf(output, "%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                metrics[i].date,
                metrics[i].investment,
                metrics[i].gain,
                metrics[i].normalized_gain,
                metrics[i].summed_gain,
                metrics[i].nav,
                metrics[i].drawdown,
                max_drawdown,
                annual_return,
                ar_md_ratio);
    }

    fclose(output);
    fclose(input);
    free(metrics);
}
// Matrix operations
void multiply_matrix(double* A, double* B, double* C, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            C[i * p + j] = 0;
            for (int k = 0; k < n; k++) {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}

void transpose_matrix(double* A, double* B, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            B[j * m + i] = A[i * n + j];
        }
    }
}



double predict(double* features, double* weights, int num_features, RegressionType reg_type) {
    double z = 0;
    for (int i = 0; i < (num_features + 1); i++) {
        z += (features[i] + EPSILON) * weights[i];
    }
    
    if (reg_type == LOGISTIC_REGRESSION) {
        // Clip z to prevent overflow
        z = fmax(-20.0, fmin(20.0, z));
        double pred = sigmoid(z);
        // Clip prediction to valid probability range
        return fmax(1e-15, fmin(1 - 1e-15, pred));
    }
    return z;
}
int compare_dates_descending(const void* a, const void* b) {
    const DataPoint* point_a = (const DataPoint*)a;
    const DataPoint* point_b = (const DataPoint*)b;
    
    int date_val_a = date_to_int(point_a->date);
    int date_val_b = date_to_int(point_b->date);
    
    return date_val_b - date_val_a;
}
void sliding_window_regression(Dataset dataset, int training_window, const char* output_file, RegressionType reg_type) {
    // Sort dataset by date in descending order (for each ticker)
    qsort(dataset.data, dataset.total_samples, sizeof(DataPoint), compare_dates_descending);

    double* weights = malloc((dataset.num_features + 1) * sizeof(double));
    FILE* file = fopen(output_file, "w");

    if (!file) {
        printf("Error creating file\n");
        free(weights);
        return;
    }


    fprintf(file, "Date,Ticker_Concat,");
    for (int i = 0; i < dataset.num_features; i++) {
        fprintf(file, "F%d,", i + 1);
    }
    for (int i = 0; i < dataset.num_features; i++) {
        fprintf(file, "C%d,", i + 1);
    }
    if (reg_type == LINEAR_REGRESSION) {
        fprintf(file, "Intercept,Target,Predicted\n");
    } else {
        fprintf(file, "Intercept,Target,DPV_new,Predicted\n");
    }

    // Collect unique tickers
    char** unique_tickers = malloc(dataset.total_samples * sizeof(char*));
    int unique_ticker_count = 0;
    int* ticker_counts = calloc(dataset.total_samples, sizeof(int));

    // Find unique tickers and their first appearances
    for (int i = 0; i < dataset.total_samples; i++) {
        int found = 0;
        for (int j = 0; j < unique_ticker_count; j++) {
            if (strcmp(dataset.data[i].ticker_concat, unique_tickers[j]) == 0) {
                found = 1;
                break;
            }
        }
        if (!found) {
            unique_tickers[unique_ticker_count] = dataset.data[i].ticker_concat;
            unique_ticker_count++;
        }
    }

    // Process each unique ticker
    for (int ticker_idx = 0; ticker_idx < unique_ticker_count; ticker_idx++) {
        for (int i = 0; i < dataset.total_samples; i++) {
            // Match current ticker
            if (strcmp(dataset.data[i].ticker_concat, unique_tickers[ticker_idx]) != 0) continue;

            // Find training samples for this ticker
            int training_samples = 0;
            int* training_indices = malloc(training_window * sizeof(int));
            
            // Collect training samples going backwards
            for (int j = i + 1; j < dataset.total_samples && training_samples < training_window; j++) {
                if (strcmp(dataset.data[j].ticker_concat, unique_tickers[ticker_idx]) == 0) {
                    training_indices[training_samples++] = j;
                }
            }

            // Skip if not enough training samples
            if (training_samples < training_window) {
                free(training_indices);
                continue;
            }

            // Create temporary training dataset
            DataPoint* training_data = malloc(training_window * sizeof(DataPoint));
            for (int j = 0; j < training_window; j++) {
                training_data[j] = dataset.data[training_indices[j]];
            }

            // Train model
            if (reg_type == LINEAR_REGRESSION) {
                train_model_ols(training_data, 0, training_window, dataset.num_features, weights);
            } else {
                train_model_logistic(training_data, 0, training_window, dataset.num_features, weights);
            }

            // Write results
            // fprintf(file, "%s,%s,", dataset.data[i].ticker_concat, dataset.data[i].ticker_concat);
            fprintf(file, "%s,%s,", dataset.data[i].date, dataset.data[i].ticker_concat);

            // Write features
            for (int j = 0; j < dataset.num_features; j++) {
                fprintf(file, "%.6f,", dataset.data[i].features[j]);
            }

            // Write coefficients
            for (int j = 0; j < dataset.num_features + 1; j++) {
                fprintf(file, "%.4f,", weights[j]);
            }

            // Calculate and write prediction
            double prediction = predict(dataset.data[i].features, weights, dataset.num_features, reg_type);
            double target = dataset.data[i].target;

            if (reg_type == LINEAR_REGRESSION) {
                fprintf(file, "%.4f,%.4f\n", target, prediction);
            } else {
                double dpv_new = transform_dpv(target, 0.005);
                fprintf(file, "%.4f,%.4f,%.4f\n", target, dpv_new, prediction);
            }

            free(training_data);
            free(training_indices);
        }
    }

    fclose(file);
    free(weights);
    free(unique_tickers);
    free(ticker_counts);
}

void generate_feature_summary(Dataset dataset, int training_window, const char* base_output_file, RegressionType reg_type, const char* summary_output_file) {
    // Open summary output file
    FILE* summary_file = fopen(summary_output_file, "w");
    if (!summary_file) {
        printf("Error creating feature summary file\n");
        return;
    }

    // Write header for summary file
    fprintf(summary_file, "Feature_Number,Feature_Name,Total_Samples,Regression_Type,Training_Window,Final_AUC,Final_NAV,Max_Drawdown,Annual_Return,AR_MD_Ratio\n");

    // Process each feature
    for (int feature_idx = 0; feature_idx < dataset.num_features; feature_idx++) {
        // Generate filenames for this feature
        char output_file[MAX_LINE_LENGTH];
        char gains_output_file[MAX_LINE_LENGTH];
        char nav_output_file[MAX_LINE_LENGTH];
        
        snprintf(output_file, MAX_LINE_LENGTH, "feature%d_%s", feature_idx + 1, base_output_file);
        snprintf(gains_output_file, MAX_LINE_LENGTH, "gains_feature%d_%s", feature_idx + 1, base_output_file);
        snprintf(nav_output_file, MAX_LINE_LENGTH, "nav_feature%d_%s", feature_idx + 1, base_output_file);

        // Read gains chart to get AUC
        double final_auc = 0.0;
        FILE* gains_file = fopen(gains_output_file, "r");
        if (gains_file) {
            char line[MAX_LINE_LENGTH];
            while (fgets(line, MAX_LINE_LENGTH, gains_file)) {
                if (strstr(line, "FINAL AUC SCORE") != NULL) {
                    sscanf(line, "FINAL AUC SCORE,,,,,,,%lf", &final_auc);
                    break;
                }
            }
            fclose(gains_file);
        }

        // Read NAV metrics to get final NAV, max drawdown, annual return, etc.
        double final_nav = 1.0;
        double max_drawdown = 0.0;
        double annual_return = 0.0;
        double ar_md_ratio = 0.0;
        FILE* nav_file = fopen(nav_output_file, "r");
        if (nav_file) {
            char line[MAX_LINE_LENGTH];
            char* last_line = NULL;
            
            // Read all lines to get the last line
            while (fgets(line, MAX_LINE_LENGTH, nav_file)) {
                if (last_line) free(last_line);
                last_line = strdup(line);
            }

            // Parse last line for metrics
            if (last_line) {
                char* token = strtok(last_line, ",");
                int col = 0;
                while (token != NULL) {
                    switch(col) {
                        case 5: final_nav = atof(token); break;
                        case 7: max_drawdown = atof(token); break;
                        case 8: annual_return = atof(token); break;
                        case 9: ar_md_ratio = atof(token); break;
                    }
                    token = strtok(NULL, ",");
                    col++;
                }
                free(last_line);
            }
            fclose(nav_file);
        }

        // Write summary for this feature
        fprintf(summary_file, "%d,F%d,%d,%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n", 
            feature_idx + 1,          // Feature Number
            feature_idx + 1,          // Feature Name
            dataset.total_samples,    // Total Samples
            (reg_type == LINEAR_REGRESSION) ? "Linear" : "Logistic", // Regression Type
            training_window,          // Training Window
            final_auc,                // Final AUC Score
            final_nav,                // Final NAV
            max_drawdown,             // Max Drawdown
            annual_return,            // Annual Return
            ar_md_ratio               // AR/MD Ratio
        );
    }

    fclose(summary_file);
    printf("Feature summary created in %s\n", summary_output_file);
}

void sliding_window_regression_individual(Dataset dataset, int training_window, const char* base_output_file, RegressionType reg_type) {
    // Sort dataset by date in descending order
    qsort(dataset.data, dataset.total_samples, sizeof(DataPoint), compare_dates_descending);

    // Process each feature individually
    for (int feature_idx = 0; feature_idx < dataset.num_features; feature_idx++) {
        // Create output filename for current feature
        char output_file[MAX_LINE_LENGTH];
        snprintf(output_file, MAX_LINE_LENGTH, "feature%d_%s", feature_idx + 1, base_output_file);
        
        FILE* file = fopen(output_file, "w");
        if (!file) {
            printf("Error creating file for feature %d\n", feature_idx + 1);
            continue;
        }

        // Allocate weights for single feature (plus bias term)
        double* weights = malloc(2 * sizeof(double));

        // Write header
        fprintf(file, "Date,Ticker_Concat,F%d,C%d,", feature_idx + 1, feature_idx + 1);
        if (reg_type == LINEAR_REGRESSION) {
            fprintf(file, "Intercept,Target,Predicted\n");
        } else {
            fprintf(file, "Intercept,Target,DPV_new,Predicted\n");
        }

        // Process each unique ticker
        char** unique_tickers = malloc(dataset.total_samples * sizeof(char*));
        int unique_ticker_count = 0;

        // Find unique tickers
        for (int i = 0; i < dataset.total_samples; i++) {
            int found = 0;
            for (int j = 0; j < unique_ticker_count; j++) {
                if (strcmp(dataset.data[i].ticker_concat, unique_tickers[j]) == 0) {
                    found = 1;
                    break;
                }
            }
            if (!found) {
                unique_tickers[unique_ticker_count] = dataset.data[i].ticker_concat;
                unique_ticker_count++;
            }
        }

        // Process each ticker
        for (int ticker_idx = 0; ticker_idx < unique_ticker_count; ticker_idx++) {
            for (int i = 0; i < dataset.total_samples; i++) {
                if (strcmp(dataset.data[i].ticker_concat, unique_tickers[ticker_idx]) != 0) continue;

                // Create training data for single feature
                int training_samples = 0;
                DataPoint* training_data = malloc(training_window * sizeof(DataPoint));
                
                // Allocate feature array for single feature plus bias
                for (int k = 0; k < training_window; k++) {
                    training_data[k].features = malloc(2 * sizeof(double));
                }

                // Collect training samples
                for (int j = i + 1; j < dataset.total_samples && training_samples < training_window; j++) {
                    if (strcmp(dataset.data[j].ticker_concat, unique_tickers[ticker_idx]) == 0) {
                        // Copy only the current feature and add bias term
                        training_data[training_samples].features[0] = dataset.data[j].features[feature_idx];
                        training_data[training_samples].features[1] = 1.0; // Bias term
                        training_data[training_samples].target = dataset.data[j].target;
                        training_samples++;
                    }
                }

                // Skip if not enough training samples
                if (training_samples < training_window) {
                    for (int k = 0; k < training_window; k++) {
                        free(training_data[k].features);
                    }
                    free(training_data);
                    continue;
                }

                // Train model for single feature
                if (reg_type == LINEAR_REGRESSION) {
                    train_model_ols(training_data, 0, training_window, 1, weights);
                } else {
                    train_model_logistic(training_data, 0, training_window, 1, weights);
                }

                // Prepare current sample's feature vector
                double current_features[2] = {
                    dataset.data[i].features[feature_idx],
                    1.0 // Bias term
                };

                // Calculate prediction
                double prediction = predict(current_features, weights, 1, reg_type);
                double target = dataset.data[i].target;

                // Write results
                fprintf(file, "%s,%s,%.6f,%.4f,%.4f,", 
                    dataset.data[i].date,
                    dataset.data[i].ticker_concat,
                    dataset.data[i].features[feature_idx],
                    weights[0],
                    weights[1]
                );

                if (reg_type == LINEAR_REGRESSION) {
                    fprintf(file, "%.4f,%.4f\n", target, prediction);
                } else {
                    double dpv_new = transform_dpv(target, 0.005);
                    fprintf(file, "%.4f,%.4f,%.4f\n", target, dpv_new, prediction);
                }

                // Cleanup training data
                for (int k = 0; k < training_window; k++) {
                    free(training_data[k].features);
                }
                free(training_data);
            }
        }

        // Cleanup
        free(weights);
        free(unique_tickers);
        fclose(file);

        // Generate gains chart for this feature
        char gains_output_file[MAX_LINE_LENGTH];
        snprintf(gains_output_file, MAX_LINE_LENGTH, "gains_feature%d_%s", feature_idx + 1, base_output_file);
        create_gains_chart(output_file, gains_output_file);

        // Generate NAV metrics for this feature
        char nav_output_file[MAX_LINE_LENGTH];
        snprintf(nav_output_file, MAX_LINE_LENGTH, "nav_feature%d_%s", feature_idx + 1, base_output_file);
        calculate_nav_metrics(output_file, nav_output_file);

        // Generate plots
        char gains_chart_png[MAX_LINE_LENGTH];
        char nav_chart_png[MAX_LINE_LENGTH];
        char datewise_metrics[MAX_LINE_LENGTH];
        
        snprintf(gains_chart_png, MAX_LINE_LENGTH, "%s.png", gains_output_file);
        snprintf(nav_chart_png, MAX_LINE_LENGTH, "%s.png", nav_output_file);
        snprintf(datewise_metrics, MAX_LINE_LENGTH, "datewise_%s", nav_output_file);

        execute_python_script(gains_output_file, gains_chart_png);
        execute_python_script_2(nav_output_file, nav_chart_png, datewise_metrics);
    }
}

void generate_feature_performance_summary(const char* input_file, const char* output_summary_file, int num_features) {
    // Open input file to get feature names
    FILE* input = fopen(input_file, "r");
    if (!input) {
        printf("Error opening input file for feature names\n");
        return;
    }

    // Read header to get feature names
    char header[MAX_LINE_LENGTH];
    if (!fgets(header, MAX_LINE_LENGTH, input)) {
        printf("Error reading header\n");
        fclose(input);
        return;
    }
    fclose(input);

    // Split header to extract feature names
    char* feature_names[MAX_LINE_LENGTH];
    int feature_name_count = 0;
    char* token = strtok(header, ",");
    
    // Skip date and ticker columns
    token = strtok(NULL, ",");  // Skip date
    for (int i = 0; i < 4; i++) {  // Skip 4 ticker columns
        token = strtok(NULL, ",");
    }

    // Collect feature names
    while ((token = strtok(NULL, ",")) != NULL && feature_name_count < num_features) {
        feature_names[feature_name_count] = strdup(token);
        feature_name_count++;
    }

    // Open summary output file
    FILE* summary_file = fopen(output_summary_file, "w");
    if (!summary_file) {
        printf("Error creating feature performance summary file\n");
        // Free allocated feature names
        for (int i = 0; i < feature_name_count; i++) {
            free(feature_names[i]);
        }
        return;
    }

    // Write header
    fprintf(summary_file, "Feature_Number,Feature_Name,Gains_Chart_AUC,NAV_Annual_Return,NAV_Max_Drawdown,NAV_AR_MD_Ratio\n");

    // Process each feature's gains and NAV files
    for (int i = 0; i < feature_name_count; i++) {
        // Construct filenames
        char gains_file[MAX_LINE_LENGTH];
        char nav_file[MAX_LINE_LENGTH];
        
        snprintf(gains_file, MAX_LINE_LENGTH, "gains_feature%d_%s", i + 1, strrchr(input_file, '/') ? strrchr(input_file, '/') + 1 : input_file);
        snprintf(nav_file, MAX_LINE_LENGTH, "nav_feature%d_%s", i + 1, strrchr(input_file, '/') ? strrchr(input_file, '/') + 1 : input_file);

        // Open gains file to extract AUC
        FILE* gains_input = fopen(gains_file, "r");
        double auc = 0.0;
        if (gains_input) {
            char line[MAX_LINE_LENGTH];
            while (fgets(line, MAX_LINE_LENGTH, gains_input)) {
                if (strstr(line, "FINAL AUC SCORE") != NULL) {
                    sscanf(line, "FINAL AUC SCORE,,,,,,,%lf", &auc);
                    break;
                }
            }
            fclose(gains_input);
        }

        // Open NAV file to extract performance metrics
        FILE* nav_input = fopen(nav_file, "r");
        double annual_return = 0.0;
        double max_drawdown = 0.0;
        double ar_md_ratio = 0.0;
        
        if (nav_input) {
            char line[MAX_LINE_LENGTH];
            // Skip header
            fgets(line, MAX_LINE_LENGTH, nav_input);
            
            // Last line contains summary metrics
            while (fgets(line, MAX_LINE_LENGTH, nav_input)) {
                if (strstr(line, "FINAL AUC SCORE") == NULL) continue;
            }
            
            // Extract last line metrics
            sscanf(line, "FINAL AUC SCORE,,,,,,,,,,,,,,%lf,%lf,%lf", 
                   &annual_return, &max_drawdown, &ar_md_ratio);
            
            fclose(nav_input);
        }

        // Write summary for this feature
        fprintf(summary_file, "%d,%s,%.6f,%.6f,%.6f,%.6f\n", 
                i + 1, 
                feature_names[i], 
                auc, 
                annual_return, 
                max_drawdown, 
                ar_md_ratio);

        // Free feature name
        free(feature_names[i]);
    }

    fclose(summary_file);
    printf("Feature performance summary created in %s\n", output_summary_file);
}

void free_dataset(Dataset dataset) {
    for (int i = 0; i < dataset.total_samples; i++) {
        free(dataset.data[i].features);
        free(dataset.data[i].ticker_concat);
        // No need to free date as it's part of the struct
    }
    free(dataset.data);
}
int compare_rows(const void* a, const void* b) {
    GainsRow* row1 = (GainsRow*)a;
    GainsRow* row2 = (GainsRow*)b;
    if (row1->predicted < row2->predicted) return 1;
    if (row1->predicted > row2->predicted) return -1;
    return 0;
}

int count_rows(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) return 0;
    
    int count = 0;
    char line[MAX_LINE_LENGTH];
    
    fgets(line, MAX_LINE_LENGTH, file);
    
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        count++;
    }
    
    fclose(file);
    return count;
}

void create_gains_chart(const char* input_file, const char* output_file) {
    int total_rows = count_rows(input_file);
    if (total_rows == 0) {
        printf("Error: Empty input file or file not found\n");
        return;
    }
    
    printf("Total rows found: %d\n", total_rows);
    
    GainsRow* data = malloc(total_rows * sizeof(GainsRow));
    if (!data) {
        printf("Error: Memory allocation failed\n");
        return;
    }

    FILE* input = fopen(input_file, "r");
    if (!input) {
        printf("Error opening input file\n");
        free(data);
        return;
    }

    char line[MAX_LINE_LENGTH];
    char header[MAX_LINE_LENGTH];
    fgets(header, MAX_LINE_LENGTH, input);  // Read header
    
    // Determine if it's logistic regression by checking for DPV_new in header
    int is_logistic = (strstr(header, "DPV_new") != NULL);
    
    int row_count = 0;
    int total_positive_labels = 0;

    // Process data rows
    while (fgets(line, MAX_LINE_LENGTH, input) && row_count < total_rows) {
        char* line_copy = strdup(line);
        char* token = strtok(line_copy, ",");
        int col_count = 0;
        
        // Count total columns
        while (token != NULL) {
            col_count++;
            token = strtok(NULL, ",");
        }
        free(line_copy);
        
        // Process columns
        token = strtok(line, ",");
        int current_col = 1;
        
        while (token != NULL) {
            if (is_logistic) {
                // For logistic regression:
                // Target is second from last (before DPV_new and Predicted)
                if (current_col == col_count - 2) {
                    data[row_count].target = atof(token);
                }
                // Predicted is last column
                else if (current_col == col_count) {
                    data[row_count].predicted = atof(token);
                }
            } else {
                // For linear regression:
                // Target is second from last (before Predicted)
                if (current_col == col_count - 1) {
                    data[row_count].target = atof(token);
                }
                // Predicted is last column
                else if (current_col == col_count) {
                    data[row_count].predicted = atof(token);
                }
            }
            token = strtok(NULL, ",");
            current_col++;
        }
        
        // Use original target for label determination
        data[row_count].target_label = (data[row_count].target > 0) ? 1 : 0;
        total_positive_labels += data[row_count].target_label;
        row_count++;
    }
    fclose(input);

    printf("Read %d rows, processing...\n", row_count);

    // Sort and calculate gains
    qsort(data, row_count, sizeof(GainsRow), compare_rows);

    int cumulative_positives = 0;
    for (int i = 0; i < row_count; i++) {
        data[i].row_weight = (double)(i + 1) / row_count;
        cumulative_positives += data[i].target_label;
        data[i].cumulative_gain = (double)cumulative_positives / total_positive_labels;
        data[i].random_line = data[i].row_weight;
    }

    // Calculate AUC
    double total_auc = 0.0;
    data[0].auc = 0.0;
    
    for (int i = 1; i < row_count; i++) {
        data[i].auc = (data[i].cumulative_gain + data[i-1].cumulative_gain) / 2.0;
        total_auc += data[i].auc;
    }

    double final_auc = total_auc / (row_count - 1);

    // Write results
    FILE* output = fopen(output_file, "w");
    if (!output) {
        printf("Error creating output file\n");
        free(data);
        return;
    }

    fprintf(output, "Predicted,Target,Target_Label,Row_Weight,Model,Random_Line,AUC,Final_AUC\n");

    for (int i = 0; i < row_count; i++) {
        if (i == 0) {
            fprintf(output, "%.6f,%.6f,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                data[i].predicted, data[i].target, data[i].target_label,
                data[i].row_weight, data[i].cumulative_gain, data[i].random_line,
                data[i].auc, final_auc);
        } else {
            fprintf(output, "%.6f,%.6f,%d,%.6f,%.6f,%.6f,%.6f,\n",
                data[i].predicted, data[i].target, data[i].target_label,
                data[i].row_weight, data[i].cumulative_gain, data[i].random_line,
                data[i].auc);
        }
    }

    fprintf(output, "FINAL AUC SCORE,,,,,,,%.6f\n", final_auc);

    fclose(output);
    free(data);

    printf("Gains chart created successfully in %s\n", output_file);
    printf("Total rows processed: %d\n", row_count);
    printf("Total positive labels: %d\n", total_positive_labels);
    printf("Final AUC Score: %.6f\n", final_auc);
}
int main() {
    char input_file[MAX_LINE_LENGTH];
    char regression_output_file[MAX_LINE_LENGTH];
    char gains_chart_output_file[MAX_LINE_LENGTH];
    char gains_chart_png[MAX_LINE_LENGTH];
    char nav_metrics_file[MAX_LINE_LENGTH];
    char nav_chart_png[MAX_LINE_LENGTH];
    char datewise_metrics[MAX_LINE_LENGTH];
    char buffer[MAX_LINE_LENGTH];
    char feature_summary_file[MAX_LINE_LENGTH];
    RegressionType reg_type = DEFAULT_REGRESSION_TYPE;
    int training_window = DEFAULT_TRAINING_WINDOW;

    // Get input filename or use default
    printf("Enter the name of the input file (press Enter to use default '%s'): ", DEFAULT_INPUT_FILENAME);
    if (!fgets(input_file, MAX_LINE_LENGTH, stdin) || input_file[0] == '\n') {
        strcpy(input_file, DEFAULT_INPUT_FILENAME);
        printf("Using default input file: %s\n", DEFAULT_INPUT_FILENAME);
    } else {
        input_file[strcspn(input_file, "\n")] = '\0';
    }

    // Get regression type or use default
    printf("Select regression type (0 for Linear, 1 for Logistic, press Enter for default [%s]): ",
           (DEFAULT_REGRESSION_TYPE == LINEAR_REGRESSION) ? "Linear" : "Logistic");
    
    if (!fgets(buffer, MAX_LINE_LENGTH, stdin) || buffer[0] == '\n') {
        reg_type = DEFAULT_REGRESSION_TYPE;
        printf("Using default regression type: %s\n", 
               (DEFAULT_REGRESSION_TYPE == LINEAR_REGRESSION) ? "Linear" : "Logistic");
    } else {
        int type_choice = atoi(buffer);
        reg_type = (type_choice == 1) ? LOGISTIC_REGRESSION : LINEAR_REGRESSION;
    }

    // Get training window size or use default
    printf("Enter training window size (press Enter for default [%d]): ", DEFAULT_TRAINING_WINDOW);
    if (!fgets(buffer, MAX_LINE_LENGTH, stdin) || buffer[0] == '\n') {
        training_window = DEFAULT_TRAINING_WINDOW;
        printf("Using default training window size: %d\n", DEFAULT_TRAINING_WINDOW);
    } else {
        training_window = atoi(buffer);
    }

    // Generate output filename based on input filename
    snprintf(regression_output_file, MAX_LINE_LENGTH, "results_%s", input_file);

    // Run regression
    Dataset dataset = read_csv(input_file);
    if (dataset.total_samples == 0) {
        printf("Error: No data found in input file\n");
        return 1;
    }

    // Print regression type being used
    printf("Running %s regression with training window size: %d\n", 
           (reg_type == LINEAR_REGRESSION) ? "linear" : "logistic",
           training_window);

    // Validate training window size
    if (training_window <= 0 || training_window >= dataset.total_samples) {
        printf("Error: Invalid training window size\n");
        free_dataset(dataset);
        return 1;
    }

    // Run regression with selected type
    sliding_window_regression_individual(dataset, training_window, regression_output_file, reg_type);
    free_dataset(dataset);

    snprintf(feature_summary_file, MAX_LINE_LENGTH, "feature_summary_%s", regression_output_file);
    
    generate_feature_summary(dataset, training_window, regression_output_file, reg_type, feature_summary_file);
    // snprintf(feature_summary_file, MAX_LINE_LENGTH, "feature_performance_summary_%s", regression_output_file);
    
    // generate_feature_performance_summary(input_file, feature_summary_file, dataset.num_features);

    printf("Individual feature regression analysis complete.\n");

    // Create gains chart CSV with standardized name
    snprintf(gains_chart_output_file, MAX_LINE_LENGTH, "gains_%s", regression_output_file);
    create_gains_chart(regression_output_file, gains_chart_output_file);
    printf("Gains chart CSV created in %s\n", gains_chart_output_file);

    // Create PNG file name and generate the plot
    snprintf(gains_chart_png, MAX_LINE_LENGTH, "%s.png", gains_chart_output_file);
    if (execute_python_script(gains_chart_output_file, gains_chart_png) == 0) {
        printf("Gains chart plot created in %s\n", gains_chart_png);
    } else {
        printf("Error creating gains chart plot\n");
    }

    // Generate NAV metrics CSV
    snprintf(nav_metrics_file, MAX_LINE_LENGTH, "nav_%s", regression_output_file);
    calculate_nav_metrics(regression_output_file, nav_metrics_file);
    
    // Create NAV chart PNG filename
    snprintf(nav_chart_png, MAX_LINE_LENGTH, "%s.png", nav_metrics_file);
    snprintf(datewise_metrics, MAX_LINE_LENGTH, "datewise_%s", nav_metrics_file);
    
    // Generate NAV chart
    if (execute_python_script_2(nav_metrics_file, nav_chart_png, datewise_metrics) == 0) {
        printf("NAV chart created in %s\n", nav_chart_png);
    } else {
        printf("Error creating NAV chart\n");
    }

    return 0;
}