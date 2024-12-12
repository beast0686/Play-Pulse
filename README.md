# Sales Analytics Dashboard

## Overview
The **Sales Analytics Dashboard** is an interactive platform designed to provide in-depth insights into sales performance across three major product categories:

- **Personal Computers (PCs)**
- **Laptops**
- **Gaming Consoles**

This dashboard leverages **Streamlit** for the front-end interface and **Plotly** for interactive visualizations, offering users an intuitive and engaging way to explore sales data.

---

## Key Features

### 1. **Landing Page**
- A visually appealing introduction with:
  - Dashboard overview.
  - Key features description.
  - Insights teaser.
  - Call-to-action for navigation.

### 2. **Sales Analytics Section**
- Interactive visualizations:
  - **Monthly Components Sold**: Line charts showing the total quantity of components sold monthly, segmented by brand.
  - **Sales Trends**: 
    - Bar charts for monthly sales overview.
    - Pie charts for brand sales distribution.
  - **Profitability Analysis**: Bar charts illustrating profit margins by brand.
  - **Market Share Analysis**: Pie charts highlighting the top 10 brands by sales.
  - **Sales Distribution**: Histograms with boxplot overlays for sales distribution.
  - **Sales by Brand**: Box plots comparing gross sales across brands.

### 3. **Summary Page**
- A comprehensive performance summary for each product category:
  - **Total Sales**.
  - **Total Quantity Sold**.
  - **Peak Sales Month**.
  - **Top Performing Brand**.
  - **Most Profitable Brand**.
  - **Average Discount Offered**.
- Cross-category comparison with:
  - Bar charts for total sales and quantity comparisons.
  - Key observations and insights.

---

## Data Loading and Preprocessing

### Datasets
- Data is sourced from three CSV files:
  - `laptops.csv`
  - `consoles.csv`
  - `pcs.csv`

### Preprocessing Steps
- **Data Type Conversion**: Ensures numeric consistency by converting sales and quantity columns to appropriate data types.
- **Cleaning Sales Data**: Removes commas and formats numeric columns for seamless aggregation and analysis.

---

## How to Run the Dashboard

### Prerequisites
1. Python installed (version 3.7 or later recommended).
2. Required libraries:
   - `streamlit`
   - `pandas`
   - `plotly`
   - `scikit-learn`

### Installation
1. Clone the repository or download the script.
2. Install dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
1. Navigate to the project directory.
2. Execute the Streamlit application:
   ```bash
   streamlit run app.py
   ```
3. Open the URL displayed in your terminal to interact with the dashboard.

---

## File Structure
```
.
|-- app.py                    # Main application script
|-- datasets/
|   |-- laptops.csv           # Laptops dataset
|   |-- consoles.csv          # Gaming consoles dataset
|   |-- pcs.csv               # PCs dataset
|-- requirements.txt          # Required libraries
```

---

## Application Structure

### Main Functions
1. **Data Loading**:
   - `load_data()`: Loads and preprocesses the datasets.

2. **Analytics Functions**:
   - `monthly_components_sold()`: Plots monthly sales trends by brand.
   - `analyze_sales_trends()`: Analyzes monthly sales and brand performance.
   - `analyze_profitability()`: Evaluates profit margins by brand.
   - `analyze_market_share()`: Computes and visualizes brand market share.
   - `analyze_sales_distribution()`: Explores sales distribution using histograms.
   - `analyze_sales_by_brand()`: Compares sales performance across brands.

3. **Page Rendering**:
   - `show_landing_page()`: Displays the introductory landing page.
   - `generate_summary_analysis()`: Computes summary statistics for each category.
   - `show_summary_page()`: Displays the detailed summary page.

### Navigation
- The sidebar allows users to:
  - Choose between "Home," "Sales Analytics," and "Summary" pages.
  - Select product categories (PCs, Laptops, Consoles).
  - Pick desired visualizations.

---

## Future Enhancements
1. Add predictive modeling for sales forecasts using machine learning.
2. Integrate real-time data updates.
3. Expand visualizations to include advanced analytics (e.g., customer segmentation).
4. Add export functionality for visualizations and reports.

---

## Acknowledgments
- Developed with Python and supported by open-source libraries.
- Special thanks to the Streamlit and Plotly communities for their extensive documentation and support.
