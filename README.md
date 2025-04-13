## Implementation Guide


## 1. Feature Extraction Methods

1. **Deep Scattering Network**:
   - Implement using ScatNet with Morlet wavelets
   - **Specific Configuration Parameters**:
     - Use 2 network layers
     - Set time support of low-pass filter to 2^8
     - Set Q-Factor (quality factor of wavelets) to 1
     - Set number of wavelets per octave to 8
     - Use PCA to reduce feature dimensionality to 100 components
   - Input: 2ms sound segment at 500 kHz (1000 sample points)
   - Output: Feature vector of transformed coefficients

2. **MFCC (Mel-frequency cepstral coefficients)**:
   - Use librosa, python_speech_features, or Dan Ellis implementation
   - **Specific Configuration**:
     - Use 40 mel filter banks
     - Extract 13 MFCC coefficients
     - Calculate first and second derivatives of MFCCs (delta and delta-delta)
     - Resulting in 39 total features per frame
   - Apply windowing with 1ms Hamming windows and 0.5ms overlap
   - Normalize features using Z-score normalization

3. **Basic Features**:
   - Extract the following time and frequency domain features:
     - **Energy**: Sum of squared amplitude values
     - **Energy Entropy**: Entropy of normalized energy in sub-frames (divide the 2ms into 8 sub-frames)
     - **Spectral Entropy**: Entropy of the normalized FFT magnitude spectrum
     - **Maximum Frequency**: Frequency with highest amplitude in spectrum
   - Calculate FFT with 1024 points for frequency analysis
   - Ensure all features are normalized to be on comparable scales

## 2. Machine Learning Model Implementation

### A. SVM Classifier (Acoustic Box Recordings)
1. **Data Preparation**:
   - **Input Format**: Each sample should be preprocessed as described in section 2
   - **Sample Structure**: For each plant and condition, create a table with:
     - Row: One processed sound event
     - Columns: Extracted features + metadata (plant ID, species, condition)
   - **Train-Test Split**:
     - Implement leave-one-plant-out cross-validation (LOPO-CV)
     - Create a function that takes a plant ID as input and returns training and testing sets
     - For each fold, exclude all sounds from one specific plant for testing
   - **Balancing**: For each binary comparison, randomly undersample the larger class to match the smaller class size
   - **Normalization**: Apply StandardScaler (Z-score normalization) to all features
   - **Dimensionality Reduction**: Apply PCA, retaining components that explain 95% of variance

2. **Model Training**:
   - **SVM Configuration**:
     - Kernel: Radial basis function (RBF)
     - C parameter: Search in range [0.1, 1, 10, 100]
     - Gamma parameter: Search in range ['scale', 'auto', 0.001, 0.01, 0.1]
     - Optimization: Use 5-fold cross-validation on training data to find optimal hyperparameters
   - Implement separate models for each comparison pair
   - **Library**: Use sklearn.svm.SVC or LIBSVM Python bindings

3. **Model Evaluation**:
   - **Metrics to Calculate**:
     - Balanced accuracy (average of sensitivity and specificity)
     - Precision, recall, F1-score for each class
     - Confusion matrix
   - **Statistical Testing**:
     - Use Wilcoxon rank-sum test to compare model performance against random (50%)
     - Apply Holm-Bonferroni correction for multiple comparisons
     - Report p-values and significance levels

### B. CNN Classifier (Greenhouse Recordings)
1. **Network Architecture**:
   - **Input Layer**: Raw or normalized waveform (1000 time points from 2ms samples)
   - **Three CNN Blocks, each containing**:
     - Conv1D layers: 32 and 64 filters (first block), 64 and 128 (second block), 128 and 256 (third block)
     - Filter size: 5 for all Conv1D layers
     - Activation: ReLU for all convolutional layers
     - MaxPooling1D with pool size of 2
     - Dropout with rate of 0.25
   - **Final Layers**:
     - Flatten layer
     - Dense layer with 128 units and ReLU activation
     - Dropout layer with rate 0.5
     - Output layer: Dense with 1 unit (sigmoid activation) for binary classification
   - **Compilation**:
     - Loss: Binary cross-entropy
     - Optimizer: Adam with learning rate of 0.001
     - Metrics: Accuracy, precision, recall
   - **Training**:
     - Batch size: 32
     - Epochs: 50
     - Early stopping: Monitor validation loss with patience of 5
     - Class weights: Balanced to handle any class imbalance

2. **Implementation of Cross-Validation**:
   - Create a custom LOPO-CV generator function that:
     - Takes the complete dataset as input
     - For each fold, returns a training set (all plants except one) and test set (the excluded plant)
     - Maintains the mapping between sounds and their source plants
     - Ensures each plant is used exactly once for testing

## 3. Evaluation and Analysis

1. **Classification Performance Metrics**:
   - **Function Implementation**:
     - Create a function to calculate balanced accuracy as (sensitivity + specificity)/2
     - Create a function to generate and visualize confusion matrices
     - Implement statistical significance testing using scipy.stats
   - **Visualization**:
     - Plot ROC curves with AUC scores
     - Create bar charts comparing accuracy across different feature extraction methods
     - Generate precision-recall curves where appropriate

2. **Temporal and Environmental Analysis**:
   - **Time Series Analysis**:
     - Create daily aggregation function that counts sounds per day during dehydration
     - Implement hourly aggregation function to analyze patterns throughout the day
     - Use smoothing techniques (e.g., moving average) for visualization
   - **Environmental Correlation**:
     - Create binning function to group sounds by VWC ranges (e.g., 0.01 increments)
     - Implement correlation analysis between sound count and transpiration rate
     - Calculate cross-correlation with various time lags (0 to 24 hours, 1-hour steps)

3. **Cross-Validation Implementation Details**:
   - **LOPO-CV Function Specifications**:
     - Input: Complete dataset with plant identifiers
     - Process: Generate n folds where n = number of unique plants
     - For each fold, separate data into:
       - Training data: All sounds except those from plant i
       - Test data: All sounds from plant i
     - Output: List of (train_indices, test_indices) tuples
   - **Ensuring Test Set Independence**:
     - Verify that preprocessing and feature scaling are fit only on training data
     - Apply the same transformations to test data using parameters from training data

## 4. Additional Analyses

1. **Sound Normalization Tests**:
   - **Implementation Details**:
     - Create two parallel pipelines: one for raw data, one for normalized data
     - Normalization method: Divide each sample by its maximum absolute value
     - Run identical models on both datasets and compare performance
     - Implement statistical comparison of performance differences (paired t-test)

2. **Background Noise Correction**:
   - **Noise Superimposition Procedure**:
     - For each acoustic box sound: Add a randomly selected greenhouse background noise at SNR of -10 to 10 dB
     - For each greenhouse sound: Add a randomly selected acoustic box background noise at same SNR range
     - Implementation requires:
       - Function to calculate Signal-to-Noise Ratio (SNR)
       - Function to add noise at specified SNR
       - Method to randomly select background noise samples
     - Create range of SNR values to test robustness

3. **Rate-Based Analysis**:
   - **Threshold Determination**:
     - Implement a function that:
       - Takes classified sounds per hour as input
       - Tests various thresholds (1-10 sounds per hour)
       - Calculates classification metrics at each threshold
       - Returns optimal threshold based on balanced accuracy
     - Generate ROC curve for this rate-based classifier
     - Calculate AUC to quantify overall performance

4. **Specific Data Requirements and Formats**:
   - **Sound Data Structure**:
     - Format: 2ms waveform segments (1000 time points at 500kHz)
     - Organization: Python dictionary or pandas DataFrame with columns:
       - 'waveform': Raw sound array
       - 'plant_id': Unique identifier for the source plant
       - 'species': Plant species ('tomato' or 'tobacco')
       - 'condition': Treatment condition ('dry', 'cut', or 'control')
       - 'vwc': Volumetric water content (for greenhouse recordings)
       - 'recording_time': Timestamp when sound was recorded
   - **Metadata Requirements**:
     - Plant tracking table linking plant_id to all experimental conditions
     - Soil moisture timeseries data for each plant
     - Transpiration rate data (where available)

