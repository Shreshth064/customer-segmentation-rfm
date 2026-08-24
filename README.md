# Customer Segmentation Platform: RFM Clustering

A customer segmentation project that uses **RFM (Recency, Frequency, Monetary) analysis** and machine learning techniques to identify meaningful customer groups from e-commerce transaction data.

The project processes customer transaction history, constructs RFM features, applies data transformations and standardization, and segments customers based on their purchasing behavior.

## 📌 Project Overview

Understanding customer behavior is important for targeted marketing, customer retention, and improving business decisions.

This project uses the **RFM framework** to categorize customers based on three key behavioral dimensions:

* **Recency** — How recently a customer made a purchase
* **Frequency** — How often a customer makes purchases
* **Monetary** — How much a customer spends

The resulting customer segments can be used to identify groups such as:

* High-value customers
* Loyal customers
* At-risk customers
* Dormant customers
* New customers

## 📊 Dataset

The project uses the **UCI Online Retail II dataset**, which contains e-commerce transaction records collected over approximately two years.

The dataset contains **1M+ transactions** and is transformed into an RFM dataset containing approximately **5,900 customers**.

### RFM Feature Matrix

Each customer is represented using three features:

| Feature   | Description                                              |
| --------- | -------------------------------------------------------- |
| Recency   | Number of days since the customer's most recent purchase |
| Frequency | Number of purchases/orders made by the customer          |
| Monetary  | Total amount spent by the customer                       |

The resulting dataset has approximately:

```text
5,900 customers × 3 RFM features
```

## 🔄 Project Workflow

```text
Raw Transaction Data
        │
        ▼
Data Cleaning
        │
        ▼
Customer-Level Aggregation
        │
        ▼
RFM Feature Engineering
        │
        ▼
Log Transformation
        │
        ▼
Z-Score Standardization
        │
        ▼
Customer Clustering
        │
        ▼
Cluster Profiling
        │
        ▼
Customer Segmentation
```

## 🧮 Data Preprocessing

Transaction-level data is aggregated at the customer level to construct the RFM features.

Because transaction data can contain highly skewed purchasing behavior, **log transformation** is applied before standardization.

The transformed features are then standardized using the z-score:

```text
Z = (X - μ) / σ
```

where:

* `X` = original feature value
* `μ` = feature mean
* `σ` = feature standard deviation

This places the RFM variables on a comparable scale before clustering.

## 🤖 Customer Segmentation

The processed RFM features are used to identify groups of customers with similar purchasing behavior.

The clustering analysis focuses on the three RFM dimensions:

```text
Recency
Frequency
Monetary
```

After clustering, the groups are profiled according to their RFM characteristics to understand the behavior represented by each segment.

Example business-oriented segments include:

| Segment       | Typical Characteristics                              |
| ------------- | ---------------------------------------------------- |
| High Value    | Recent purchases, frequent orders, high spending     |
| Loyal         | Frequent purchases with consistent engagement        |
| At Risk       | Previously valuable customers with declining recency |
| Dormant       | Low recent activity and low engagement               |
| New Customers | Recent activity with limited purchase history        |

> Segment labels depend on the resulting cluster characteristics and should be interpreted from the actual cluster statistics.

## 📈 Analysis

The project includes analysis and visualization of customer segments across the RFM dimensions.

The analysis helps answer questions such as:

* Which customers generate the most revenue?
* Which customers purchase most frequently?
* Which customers have not purchased recently?
* Which customer groups are potentially at risk of churn?
* Which segments could be targeted with retention campaigns?
* How do customer groups differ in spending behavior?

## 🖥️ Streamlit Application

The repository includes a **Streamlit-based customer segmentation application** for presenting the analysis interactively.

The application is located in:

```text
customer_segmentation_streamlit/
```

The exploratory and analytical notebooks are located in:

```text
notebooks/
```

## 🗂️ Project Structure

```text
customer-segmentation-rfm/
│
├── customer_segmentation_streamlit/
│   └── Streamlit application files
│
├── notebooks/
│   └── Data analysis and customer segmentation notebooks
│
├── .gitignore
│
└── README.md
```

## 🛠️ Tech Stack

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Machine Learning

* Scikit-learn

### Visualization

* Matplotlib
* Seaborn

### Application

* Streamlit

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Shreshth064/customer-segmentation-rfm.git
```

Navigate into the project:

```bash
cd customer-segmentation-rfm
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

Otherwise, install the core libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit
```

### 4. Run the Streamlit Application

Navigate to the application directory if required:

```bash
cd customer_segmentation_streamlit
```

Then run:

```bash
streamlit run app.py
```

> Replace `app.py` with the actual Streamlit entry-point filename if it is named differently in the repository.

## 📚 Key Concepts

### RFM Analysis

RFM analysis is a customer analytics technique that evaluates customers using:

```text
Recency + Frequency + Monetary Value
```

It provides a simple way to quantify customer engagement and purchasing behavior.

### Customer Clustering

Clustering groups customers based on similarities in their RFM characteristics.

This allows businesses to move beyond aggregate sales metrics and understand **different types of customers individually**.

### Feature Scaling

Standardization prevents features with larger numerical ranges from dominating the clustering process.

## 💡 Business Applications

Customer segmentation can support:

* Targeted marketing campaigns
* Customer retention strategies
* Churn-risk identification
* Personalized offers
* Customer lifetime value analysis
* Loyalty programs
* Revenue optimization
* Marketing budget allocation

## 🔍 Key Takeaways

This project demonstrates an end-to-end customer analytics workflow:

1. Processing large-scale transaction data
2. Cleaning and transforming raw data
3. Engineering customer-level RFM features
4. Handling skewed distributions
5. Standardizing machine-learning features
6. Performing customer clustering
7. Profiling customer segments
8. Presenting results through an interactive application

## 🔗 Repository

**GitHub:** https://github.com/Shreshth064/customer-segmentation-rfm

## 👨‍💻 Author

**Shreshth Garg**

GitHub: https://github.com/Shreshth064

LinkedIn: https://www.linkedin.com/in/shreshth-garg/

---

⭐ If you found this project useful, consider giving the repository a star.
