
# AI/ML-Based Ransomware Detection in IoT Network Traffic

This project focuses on detecting ransomware activity within IoT network traffic using machine learning (ML) techniques. It uses flow-level features extracted from the CICIoT2023 dataset and applies classification models including Random Forest, XGBoost, and LightGBM. The solution also integrates with VirusTotal's Threat Intelligence API for optional post-detection verification.

---

## 📚 Table of Contents

- [Project Motivation](#-project-motivation)  
- [Objectives](#-objectives)  
- [Dataset Information](#-dataset-information)  
- [Dataset Included](#-dataset-included)  
- [Technologies Used](#-technologies-used)  
- [ML Models Implemented](#-ml-models-implemented)  
- [VirusTotal Integration (Optional)](#-virustotal-integration-optional)  
- [How to Run](#-how-to-run)  
- [Archive ZIP](#-archive-zip)  
- [Project Structure](#-project-structure)  
- [License](#-license)  
- [Author](#-author)  
- [Acknowledgements](#-acknowledgements)

---

## 📌 Project Motivation

Ransomware attacks pose a serious and increasing threat to both traditional IT and IoT infrastructures. Most signature-based or rule-based systems fail to detect unknown ransomware variants. This project leverages data-driven AI/ML models to proactively detect ransomware traffic flows before damage occurs.

---

## 🧠 Objectives

- Filter and preprocess network traffic logs to identify ransomware patterns.
- Apply feature engineering and handle class imbalance.
- Train and evaluate multiple ML classifiers on labeled data.
- Use ensemble learning to improve detection performance.
- Integrate with VirusTotal API to query URLs or hashes for real-time threat intelligence.

---

## 📁 Dataset Information

- **Source**: [CICIoT2023 Dataset](https://www.unb.ca/cic/datasets/iot2023.html)
- **Malware samples**: Extracted from `Backdoor_Malware` folder.
- **Benign samples**: From multiple clean `.pcap.csv` files (e.g., `BenignTraffic.pcap.csv`).
- **Final Dataset**: 24,100 total records (3,218 ransomware, 20,882 benign)

Each record includes protocol metadata and statistical flow features such as:

- Packet size statistics
- TCP flag ratios
- Protocol usage (HTTP, DNS, SSH)
- Inter-arrival time (IAT)
- Entropy, header length, and more

---

## 📂 Dataset Included

The repository includes the final cleaned dataset used for model training and evaluation:

- **File**: `final_combined_dataset.csv`
- **Size**: ~4.9 MB
- **Contents**: Preprocessed IoT network traffic records, with labeled benign and ransomware samples.
- **Source**: Derived from the CICIoT2023 public dataset and post-processed for academic research.

Each row represents flow-based metadata features engineered for machine learning models.

Note: This CSV was included for academic reproducibility and is safe for lightweight GitHub hosting.

---

## ⚙️ Technologies Used

- **Python 3.10+**
- **Pandas**, **NumPy** for data handling
- **Scikit-learn** for ML pipeline and metrics
- **XGBoost**, **LightGBM** for boosting algorithms
- **SMOTE** (from imblearn) for balancing data
- **Requests** for VirusTotal API interaction

---

## 🚀 ML Models Implemented

| Model             | Accuracy   | Description                                   |
|-------------------|------------|-----------------------------------------------|
| RandomForest      | 93.10%     | Good baseline performance with high precision |
| LightGBM          | 93.78%     | Best performing model, fast and efficient     |
| XGBoost           | 93.57%     | Competitive boosting model with lower recall  |
| Ensemble (Voting) | 93.72%     | Combines strengths of all three classifiers   |

Evaluation metrics include:
- Accuracy, Precision, Recall, F1-Score
- Confusion matrix analysis

---

## 🔐 VirusTotal Integration (Optional)

The final model includes a function that allows URL-based scans via [VirusTotal Public API](https://developers.virustotal.com/reference). This adds contextual intelligence to flagged traffic.

Example:
```python
submit_url_to_virustotal("http://testphp.vulnweb.com", api_key)
```

---

## 🏁 How to Run

1. Clone this repo and set up a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```bash
   python train_final_models.py
   ```
4. (Optional) To enable VirusTotal API:
   - Replace the placeholder API key in `train_final_models.py`
   - Set `run_virustotal_check = True`

---

## 📦 Archive ZIP

This repository includes a compressed archive of related datasets and scripts for reproducibility:

**File:** `dataset_archive.zip`

**Contents:**
- `Archive/`: Older scripts and versions
- `Backdoor_Malware/`: Malware flow samples
- `Benign_Final/`: Clean flow samples

Unzip this file in the root project directory to access raw data and intermediate steps.

---

## 📄 Project Structure

```plaintext
AI-ML/
├── train_final_models.py            # Main script for training + evaluation + VirusTotal scan
├── final_combined_dataset.csv       # Final preprocessed dataset
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git exclusions
├── .gitattributes                   # Git attributes normalization
├── README.md                        # Project documentation
├── dataset_archive.zip              # Full dataset and scripts in compressed form
```

---

## 📜 License

This project is part of an academic MSc Cybersecurity AI/ML Project and is shared for educational purposes. Please do not use or republish without proper attribution.

---

## 👤 Author

**Yashwant Suryakant Salunkhe**  
MSc Cybersecurity, National College of Ireland  
Email: x23284811@student.ncirl.ie

---

## 🧾 Acknowledgements

- CICIoT2023 dataset authors
- VirusTotal for API access
- National College of Ireland faculty guidance
