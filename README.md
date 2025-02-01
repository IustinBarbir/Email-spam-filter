# Email Spam Filter

This is a Python-based email spam filter that utilizes a Naïve Bayes classifier with TF-IDF vectorization to classify emails as either spam or clean. The model is trained on a dataset of emails and can scan a folder of new emails to classify them accordingly.

## Features
- Uses **TF-IDF vectorization** for text processing.
- Implements **Multinomial Naïve Bayes** for classification.
- Supports training on custom datasets.
- Scans new emails and classifies them as spam or clean.
- Outputs results in a structured format.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/spam-filter.git
   cd spam-filter
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
### Train the model
To train the spam filter on available email datasets, run:
```sh
python main.py -train
```
This will train the model using emails stored in `data/clean` (for non-spam) and `data/spam` (for spam). The trained model will be saved as `trained_model.pkl`.

### Scan new emails
To classify emails in a specific folder, run:
```sh
python main.py -scan <input_folder> <output_file>
```
Example:
```sh
python main.py -scan test_emails/ output_results.txt
```
The output file will contain the classification results for each email.

### Generate project information
To generate project metadata in JSON format, run:
```sh
python main.py -info project_info.json
```

## Folder Structure
```
spam-filter/
│-- main.py               # Main script
│-- requirements.txt      # Dependencies
│-- trained_model.pkl     # Pre-trained model (generated after training)
│-- data/
│   ├── clean/           # Folder containing non-spam emails
│   ├── spam/            # Folder containing spam emails
│-- test_emails/          # Folder with emails for testing
│-- output_results.txt    # Scan results
```

## Training and Evaluation Results
The model was trained on **7,615 clean emails** and **8,709 spam emails**. It was then tested on an unseen dataset with the following results:

| Detection Rate | False Positive Rate | Time (seconds) |
|---------------|---------------------|----------------|
| 0.9775        | 0.0611               | 7.2250         |

## Notes
- The `trained_model.pkl` file must be present for scanning; otherwise, retrain the model.



