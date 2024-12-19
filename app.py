from math import sqrt

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


# Load the datasets
@st.cache_data
def load_data():
    df_laptop = pd.read_csv("datasets/laptops.csv")
    df_console = pd.read_csv("datasets/consoles.csv")
    df_pcs = pd.read_csv("datasets/pcs.csv")

    # Data type conversion function
    def set_types(dataframe: pd.DataFrame) -> None:
        dataframe["Qty"] = dataframe["Qty"].astype(int)
        dataframe["Gross Sales"] = (dataframe["Gross Sales"].replace({",": ""}, regex=True).astype(float))
        dataframe["Discount"] = (dataframe["Discount"].replace({",": ""}, regex=True).astype(float))
        dataframe["Net Sales With Tax"] = (dataframe["Net Sales With Tax"].replace({",": ""}, regex=True).astype(float))
        dataframe["Tax Amount"] = (dataframe["Tax Amount"].replace({",": ""}, regex=True).astype(float))
        dataframe["Net Sales Without Tax"] = (
            dataframe["Net Sales Without Tax"].replace({",": ""}, regex=True).astype(float))
        dataframe["Target Sales Amount"] = (
            dataframe["Target Sales Amount"].replace({",": ""}, regex=True).astype(float))

    set_types(df_laptop)
    set_types(df_console)
    set_types(df_pcs)

    return df_laptop, df_console, df_pcs


def monthly_components_sold(dataframe: pd.DataFrame) -> None:
    grouped = dataframe.groupby(["Month", "Brand"], as_index=False)["Qty"].sum()
    grouped.rename(columns={"Qty": "Total Components Sold"}, inplace=True)

    top_brands = (
        grouped.groupby("Brand", as_index=False)["Total Components Sold"].sum().nlargest(10, "Total Components Sold")[
            "Brand"])

    filtered_grouped = grouped[grouped["Brand"].isin(top_brands)]

    fig = px.line(filtered_grouped, x="Month", y="Total Components Sold", color="Brand",
                  title="Number of Components Sold Monthly", labels={"Total Components Sold": "Total Quantity"},
                  markers=True, color_discrete_sequence=px.colors.qualitative.Set2, )

    fig.update_layout(xaxis_title="Month", yaxis_title="Number of Components Sold", legend_title="Brand", )

    return fig


# Sales Trends
def analyze_sales_trends(dataframe: pd.DataFrame):
    sales_summary = dataframe.groupby("Month", as_index=False).agg(Total_Sales=("Gross Sales", "sum"),
                                                                   Total_Quantity=("Qty", "sum"),
                                                                   Avg_Discount=("Discount", "mean"), )
    sales_summary = sales_summary.nlargest(10, "Total_Sales")

    fig = px.bar(sales_summary, x="Month", y="Total_Sales", text="Total_Sales", title="Monthly Sales Overview",
                 labels={"Total_Sales": "Total Sales", "Month": "Month"}, color="Total_Sales",
                 color_continuous_scale=px.colors.sequential.Blues, )
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title="Month", yaxis_title="Total Sales", showlegend=False)

    brand_performance = (dataframe.groupby("Brand", as_index=False).agg(Total_Sales=("Gross Sales", "sum"),
                                                                        Total_Discount=("Discount", "sum"),
                                                                        Total_Quantity=("Qty", "sum"), ).sort_values(
        by="Total_Sales", ascending=False))
    brand_performance = brand_performance.nlargest(10, "Total_Sales")

    fig2 = px.pie(brand_performance, names="Brand", values="Total_Sales", title="Sales Distribution by Brand",
                  labels={"Total_Sales": "Total Sales", "Brand": "Brand"}, )
    fig2.update_traces(textinfo="percent+label")

    return fig, fig2


#  Seasonal Trends Analysis
def analyze_seasonal_trends(dataframe: pd.DataFrame):
    dataframe["Month"] = pd.Categorical(dataframe["Month"],
                                        categories=["January", "February", "March", "April", "May", "June", "July",
                                                    "August", "September", "October", "November", "December", ],
                                        ordered=True, )

    sales_trends = dataframe.groupby("Month", as_index=False).agg(Total_Sales=("Gross Sales", "sum"),
                                                                  Total_Quantity=("Qty", "sum"))

    fig = px.line(sales_trends, x="Month", y="Total_Sales", markers=True, title="Seasonal Sales Trends",
                  labels={"Total_Sales": "Total Sales", "Month": "Month"}, )
    fig.update_layout(xaxis_title="Month", yaxis_title="Total Sales")
    return fig


# Market Share
def analyze_market_share(dataframe: pd.DataFrame):
    # Group by Brand and calculate total sales
    brand_share = dataframe.groupby("Brand", as_index=False).agg(Total_Sales=("Gross Sales", "sum"))

    # Sort by total sales and select the top 10 brands
    top_brands = brand_share.nlargest(10, "Total_Sales")

    # Create a pie chart for market share
    fig = px.pie(top_brands, names="Brand", values="Total_Sales", title="Market Share by Top 10 Brands",
                 labels={"Total_Sales": "Total Sales"}, )
    return fig


# Sales Distribution
def analyze_sales_distribution(dataframe: pd.DataFrame):
    fig = px.histogram(dataframe, x="Gross Sales", marginal="box", title="Sales Distribution",
                       labels={"Gross Sales": "Sales", "count": "Number of Sales"}, )
    return fig


# Sales by Brand Boxplot
def analyze_sales_by_brand(dataframe: pd.DataFrame):
    fig = px.box(dataframe, x="Brand", y="Gross Sales", title="Sales by Brand",
                 labels={"Gross Sales": "Sales", "Brand": "Brand"}, )
    return fig


# Landing Page
def show_landing_page():
    # Custom CSS for styling
    st.markdown("""
    <style>
    .landing-title {
        font-size: 48px;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 20px;
    }
    .landing-subtitle {
        font-size: 24px;
        color: #4B5563;
        text-align: center;
        margin-bottom: 30px;
    }
    .feature-box {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True, )

    # Title and Introduction
    st.markdown('<h1 class="landing-title">Play Pulse</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="landing-subtitle">Comprehensive Sales Insights Across Product Categories</h2>',
                unsafe_allow_html=True, )

    # Overview Section
    st.markdown('<div class="feature-box">', unsafe_allow_html=True)
    st.markdown("### 📊 Dashboard Overview")
    st.write("""
    Welcome to our comprehensive Sales Analytics Dashboard! 
    This interactive platform provides deep insights into sales performance across three key product categories:
    - Personal Computers (PCs)
    - Laptops
    - Gaming Consoles
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # Key Features
    st.markdown('<div class="feature-box">', unsafe_allow_html=True)
    st.markdown("### 🚀 Key Features")
    st.write("""
    - **Interactive Visualizations**: Explore sales data through multiple chart types
    - **Product Category Selection**: Switch between PCs, Laptops, and Consoles
    - **Comprehensive Analytics**:
        - Monthly Sales Trends
        - Brand Performance
        - Market Share Analysis
        - Profitability Insights
        - Sales Distribution
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # Data Insights Teaser
    st.markdown('<div class="feature-box">', unsafe_allow_html=True)
    st.markdown("### 💡 What You'll Discover")
    st.write("""
    Our dashboard helps you uncover:
    - Which brands are performing best
    - Monthly sales fluctuations
    - Impact of discounts on sales
    - Profit margins across different product lines
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # Call to Action
    st.markdown('<div class="feature-box">', unsafe_allow_html=True)
    st.markdown("### 🔍 Ready to Explore?")
    st.write("""
    Click on the menu to the left to start your data exploration journey. 
    Select a product category and choose from various visualization options to gain valuable insights!
    """)
    st.markdown("</div>", unsafe_allow_html=True)


# Summary Analysis Function
def generate_summary_analysis(df_laptop, df_console, df_pcs):
    def analyze_dataset(dataframe, category_name):
        # Sales Trends Analysis
        sales_summary = dataframe.groupby("Month", as_index=False).agg(Total_Sales=("Gross Sales", "sum"),
                                                                       Total_Quantity=("Qty", "sum"),
                                                                       Avg_Discount=("Discount", "mean"), )
        max_sales_month = sales_summary.loc[sales_summary["Total_Sales"].idxmax(), "Month"]
        total_sales = sales_summary["Total_Sales"].sum()
        total_quantity = sales_summary["Total_Quantity"].sum()

        # Brand Performance
        brand_performance = dataframe.groupby("Brand", as_index=False).agg(Total_Sales=("Gross Sales", "sum"),
                                                                           Total_Quantity=("Qty", "sum"))
        top_brand = brand_performance.loc[brand_performance["Total_Sales"].idxmax(), "Brand"]

        # Profitability Analysis
        dataframe["Profit Margin (%)"] = (dataframe["Net Sales Without Tax"] / dataframe["Gross Sales"] * 100)
        profitability = dataframe.groupby("Brand", as_index=False).agg(Avg_Profit_Margin=("Profit Margin (%)", "mean"))
        most_profitable_brand = profitability.loc[profitability["Avg_Profit_Margin"].idxmax(), "Brand"]

        # Discount Analysis
        avg_discount = dataframe["Discount"].mean()

        return {"category": category_name, "max_sales_month": max_sales_month, "total_sales": total_sales,
                "total_quantity": total_quantity, "top_brand": top_brand,
                "most_profitable_brand": most_profitable_brand, "avg_discount": avg_discount, }

    # Generate summaries for each dataset
    pc_summary = analyze_dataset(df_pcs, "Personal Computers")
    laptop_summary = analyze_dataset(df_laptop, "Laptops")
    console_summary = analyze_dataset(df_console, "Consoles")

    return pc_summary, laptop_summary, console_summary


# Comprehensive Summary Page
def show_summary_page(pc_summary, laptop_summary, console_summary):
    st.markdown("""
    <style>
    .summary-title {
        font-size: 36px;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 20px;
    }
    .summary-box {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .category-header {
        color: #2563EB;
        font-size: 24px;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True, )

    st.markdown('<h1 class="summary-title">Play Pulse</h1>', unsafe_allow_html=True)

    # Summaries for each category
    categories = [("Personal Computers", pc_summary), ("Laptops", laptop_summary), ("Consoles", console_summary), ]

    for category_name, summary in categories:
        st.markdown(f'<div class="summary-box">', unsafe_allow_html=True)
        st.markdown(f'<h2 class="category-header">{category_name} Sales Analysis</h2>', unsafe_allow_html=True, )

        st.markdown(f"""
        ### Key Insights:
        - **Total Sales**: {summary['total_sales']:,.2f}
        - **Total Quantity Sold**: {summary['total_quantity']:,}
        - **Peak Sales Month**: {summary['max_sales_month']}
        - **Top Performing Brand**: {summary['top_brand']}
        - **Most Profitable Brand**: {summary['most_profitable_brand']}
        - **Average Discount**: {summary['avg_discount']:.2f}
        """)

        # Detailed Interpretation
        st.markdown(f"""
        ### Performance Interpretation:
        The {category_name} market shows dynamic sales patterns with several key observations:

        1. **Sales Performance**: 
           The category generated total sales of {summary['total_sales']:,.2f}, 
           with the peak sales month being {summary['max_sales_month']}. 
           A total of {summary['total_quantity']:,} units were sold.

        2. **Brand Dynamics**:
           {summary['top_brand']} emerged as the top-performing brand in terms of total sales, 
           while {summary['most_profitable_brand']} demonstrated the highest profit margins.

        3. **Discount Strategy**:
           The average discount across this category was {summary['avg_discount']:.2f}, 
           indicating the pricing and promotional strategies employed.
        """)

        st.markdown("</div>", unsafe_allow_html=True)

    # Overall Market Comparison
    st.markdown('<div class="summary-box">', unsafe_allow_html=True)
    st.markdown('<h2 class="category-header">Cross-Category Market Comparison</h2>', unsafe_allow_html=True, )

    # Comparative Analysis
    comparison_data = pd.DataFrame([{"Category": "Personal Computers", "Total Sales": pc_summary["total_sales"],
                                     "Total Quantity": pc_summary["total_quantity"], },
                                    {"Category": "Laptops", "Total Sales": laptop_summary["total_sales"],
                                     "Total Quantity": laptop_summary["total_quantity"], },
                                    {"Category": "Consoles", "Total Sales": console_summary["total_sales"],
                                     "Total Quantity": console_summary["total_quantity"], }, ])

    # Create two columns for visualizations
    col1, col2 = st.columns(2)

    # Sales Comparison Bar Chart
    with col1:
        fig1 = px.bar(comparison_data, x="Category", y="Total Sales", title="Total Sales Comparison",
                      labels={"Total Sales": "Total Sales"}, )
        st.plotly_chart(fig1)

    # Quantity Comparison Bar Chart
    with col2:
        fig2 = px.bar(comparison_data, x="Category", y="Total Quantity", title="Total Quantity Sold Comparison",
                      labels={"Total Quantity": "Total Quantity"}, )
        st.plotly_chart(fig2)

    st.markdown("""
    ### Overall Market Insights:
    - Comparative analysis reveals the sales performance and market dynamics across different product categories
    - The visualizations provide a quick overview of total sales and quantities sold
    - Each category shows unique characteristics in terms of sales volume and market penetration
    """)

    st.markdown("</div>", unsafe_allow_html=True)


# ARIMA Forecasting with Brand Filter and Error Metrics
def forecast_sales_arima(dataframe: pd.DataFrame, brand: str, steps: int):
    brand_data = dataframe[dataframe["Brand"] == brand]
    sales_data = brand_data.groupby("Month")["Gross Sales"].sum()
    sales_data = sales_data.reset_index()
    sales_data["Month"] = pd.to_datetime("2024 " + sales_data["Month"], format="%Y %B")
    sales_data.set_index("Month", inplace=True)

    model = ARIMA(sales_data["Gross Sales"], order=(5, 1, 0))
    model_fit = model.fit()

    # Forecasting future sales
    forecast = model_fit.forecast(steps=steps)
    future_months = pd.date_range(start="2025-01-01", periods=steps + 1, freq="M")[1:]

    # Evaluation Metrics (if enough data is available)
    actuals = sales_data["Gross Sales"].iloc[-steps:].values if len(sales_data) > steps else None
    mae, mse, rmse = None, None, None
    if actuals is not None:
        mae = mean_absolute_error(actuals, forecast[:len(actuals)])
        mse = mean_squared_error(actuals, forecast[:len(actuals)])
        rmse = sqrt(mse)
        st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
        st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")

    forecast_df = pd.DataFrame({"Month": future_months, "Forecasted Sales": forecast})

    # Plotting
    fig = px.line(sales_data.reset_index(), x="Month", y="Gross Sales", title=f"ARIMA Forecasting of Sales for {brand}",
                  labels={"Gross Sales": "Total Sales", "Month": "Month"}, markers=True)

    fig.add_scatter(x=forecast_df["Month"], y=forecast_df["Forecasted Sales"], mode="lines+markers",
                    name="Forecasted Sales", line=dict(dash="dash", color="red"))

    fig.update_layout(xaxis_title="Month", yaxis_title="Total Sales", showlegend=True)
    return fig


# Sidebar for selecting brand and forecast period
def show_forecasts(df):
    brand = st.sidebar.selectbox("Select Brand for Sales Forecast", df["Brand"].unique())
    steps = st.sidebar.slider("Select Forecast Period", 1, 12, 6)
    method = st.sidebar.selectbox("Select Forecasting Method", ["ARIMA", "SARIMA"])

    if method == "ARIMA":
        st.plotly_chart(forecast_sales_arima(df, brand, steps))
    elif method == "SARIMA":
        st.plotly_chart(forecast_sales_sarima(df, brand,
                                              steps))  # elif method == "Exponential Smoothing":  #     st.plotly_chart(forecast_sales_ets(df, brand, steps))


def forecast_sales_sarima(dataframe: pd.DataFrame, brand: str, steps: int):
    brand_data = dataframe[dataframe["Brand"] == brand]
    sales_data = brand_data.groupby("Month")["Gross Sales"].sum()
    sales_data = sales_data.reset_index()
    sales_data["Month"] = pd.to_datetime("2024 " + sales_data["Month"], format="%Y %B")
    sales_data.set_index("Month", inplace=True)

    # Fit SARIMA model
    model = SARIMAX(sales_data["Gross Sales"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit()

    # Forecast future sales
    forecast = model_fit.forecast(steps=steps)
    future_months = pd.date_range(start="2025-01-01", periods=steps + 1, freq="M")[1:]

    forecast_df = pd.DataFrame({"Month": future_months, "Forecasted Sales": forecast})

    # Plotting
    fig = px.line(sales_data.reset_index(), x="Month", y="Gross Sales",
                  title=f"SARIMA Forecasting of Sales for {brand}",
                  labels={"Gross Sales": "Total Sales", "Month": "Month"}, markers=True)

    fig.add_scatter(x=forecast_df["Month"], y=forecast_df["Forecasted Sales"], mode="lines+markers",
                    name="Forecasted Sales", line=dict(dash="dash", color="green"))

    return fig


def forecast_sales_ets(dataframe: pd.DataFrame, brand: str, steps: int):
    # Filter data for the selected brand
    brand_data = dataframe[dataframe["Brand"] == brand]

    # Aggregate sales data by month
    sales_data = brand_data.groupby("Month")["Gross Sales"].sum().reset_index()

    # Ensure the Month column is correctly formatted
    sales_data["Month"] = pd.to_datetime("2024 " + sales_data["Month"], format="%Y %B", errors="coerce")
    sales_data = sales_data.dropna(subset=["Month"])  # Drop rows with invalid dates
    sales_data.set_index("Month", inplace=True)

    # Fit Exponential Smoothing model
    try:
        model = ExponentialSmoothing(sales_data["Gross Sales"], trend="add", seasonal="add", seasonal_periods=12)
        model_fit = model.fit()
    except ValueError as e:
        st.error(f"Error fitting Exponential Smoothing model: {e}")
        return None

    # Forecast future sales
    forecast = model_fit.forecast(steps=steps)
    future_months = pd.date_range(start="2025-01-01", periods=steps + 1, freq="M")[1:]

    # Create a dataframe for forecasted sales
    forecast_df = pd.DataFrame({"Month": future_months, "Forecasted Sales": forecast})

    # Plot the actual and forecasted sales
    fig = px.line(sales_data.reset_index(), x="Month", y="Gross Sales",
                  title=f"Exponential Smoothing Forecasting of Sales for {brand}",
                  labels={"Gross Sales": "Total Sales", "Month": "Month"}, markers=True)

    fig.add_scatter(x=forecast_df["Month"], y=forecast_df["Forecasted Sales"], mode="lines+markers",
                    name="Forecasted Sales", line=dict(dash="dash", color="orange"))

    return fig


# Kmeans Clustering
def kmeans_clustering(dataframe: pd.DataFrame, num_clusters: int):
    # Select relevant columns for clustering (e.g., 'Gross Sales' and 'Qty')
    data = dataframe[["Gross Sales", "Qty"]].dropna()

    # Scale the data to normalize the feature ranges
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    dataframe["Cluster"] = kmeans.fit_predict(scaled_data)

    # Create a scatter plot to visualize the clusters
    fig = px.scatter(dataframe, x="Gross Sales", y="Qty", color="Cluster", title="KMeans Clustering of Sales Data",
                     labels={"Gross Sales": "Total Sales", "Qty": "Quantity Sold"}, color_continuous_scale="Viridis", )
    fig.update_layout(xaxis_title="Total Sales", yaxis_title="Quantity Sold", showlegend=True)
    return fig


# Elbow Method to find the optimal number of clusters with missing values handling
def plot_elbow_method(df):
    # Select numeric columns for clustering (assuming df contains numerical data)
    X = df.select_dtypes(include=["float64", "int64"]).values

    # Handle missing values by imputing with the mean of each column
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    # List to hold the inertia values for different k
    inertia = []

    # Range of k values to try (2 to 10 clusters, you can modify this range)
    k_range = range(1, 11)

    # Fit KMeans for different k values and record the inertia (within-cluster sum of squared distances)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_imputed)
        inertia.append(kmeans.inertia_)

    # Plotting the Elbow Method
    plt.figure(figsize=(8, 6))
    plt.plot(k_range, inertia, marker="o", color="b", linestyle="--")
    plt.title("Elbow Method for Optimal k")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (Sum of Squared Distances)")
    plt.xticks(k_range)
    plt.grid(True)

    # Show the plot in the Streamlit app
    st.pyplot(plt)


# KMeans Clustering with Silhouette Score and 3D Visualization
def kmeans_clustering_with_metrics(dataframe: pd.DataFrame, num_clusters: int):
    # Select relevant columns
    data = dataframe[["Gross Sales", "Qty", "Discount"]].dropna()

    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    dataframe["Cluster"] = kmeans.fit_predict(scaled_data)

    # Calculate Silhouette Score
    silhouette_avg = silhouette_score(scaled_data, dataframe["Cluster"])
    print(f"Silhouette Score: {silhouette_avg:.3f}")

    # 3D Scatter Plot for Clusters
    fig = px.scatter_3d(dataframe, x="Gross Sales", y="Qty", z="Discount", color="Cluster",
                        title=f"KMeans 3D Clustering with {num_clusters} Clusters",
                        labels={"Gross Sales": "Total Sales", "Qty": "Quantity Sold", "Discount": "Discount", },
                        color_continuous_scale="Viridis", )

    fig.update_layout(scene={"xaxis_title": "Total Sales", "yaxis_title": "Quantity Sold", "zaxis_title": "Discount", },
                      showlegend=True, )
    return fig


# Modify the main function to include the summary page
def main():
    # Load the data
    df_laptop, df_console, df_pcs = load_data()

    # Generate summary analysis
    pc_summary, laptop_summary, console_summary = generate_summary_analysis(df_laptop, df_console, df_pcs)

    # Sidebar for page selection
    page = st.sidebar.radio("Navigate", ["Home", "Sales Analytics", "Advanced Analysis", "Summary"])

    # Conditional page rendering
    if page == "Home":
        show_landing_page()
    elif page == "Summary":
        # Show summary page with generated summaries
        show_summary_page(pc_summary, laptop_summary, console_summary)
    elif page == "Advanced Analysis":
        # Existing Sales Analytics Page (remains the same as in previous artifact)
        st.title("Advanced Analytics Dashboard")

        # Sidebar for selecting dataset
        dataset_selection = st.sidebar.radio("Select Product Category", ["PCs", "Laptops", "Consoles"])

        # Choose the appropriate dataframe based on selection
        if dataset_selection == "PCs":
            df = df_pcs
            st.sidebar.write("PC Advanced Analytics")
        elif dataset_selection == "Laptops":
            df = df_laptop
            st.sidebar.write("Laptop Advanced Analytics")
        else:
            df = df_console
            st.sidebar.write("Console Advanced Analytics")

        visualization = st.sidebar.selectbox("Select Visualization",
                                             ["Forecast Sales", "Clustering", "Elbow Method for Optimal k",
                                              "3D KMeans Clustering"], )

        # Display selected visualization
        if visualization == "Forecast Sales":
            show_forecasts(df)
        elif visualization == "Clustering":
            num_clusters = st.sidebar.slider("Select Number of Clusters", 2, 10, 3)
            st.plotly_chart(kmeans_clustering(df, num_clusters))
        elif visualization == "Elbow Method for Optimal k":
            st.sidebar.write("Elbow Method: Visualize the optimal number of clusters")
            st.write("This graph helps you identify the best number of clusters (k) for your data.")
            plot_elbow_method(df)  # Plot the Elbow Method graph
        elif visualization == "3D KMeans Clustering":
            num_clusters = st.sidebar.slider("Select Number of Clusters", 2, 10, 3)
            fig = kmeans_clustering_with_metrics(df, num_clusters)  # Use the KMeans 3D clustering function
            st.plotly_chart(fig)

    elif page == "Sales Analytics":
        # Existing Sales Analytics Page (remains the same as in previous artifact)
        st.title("Sales Analytics Dashboard")

        # Sidebar for selecting dataset
        dataset_selection = st.sidebar.radio("Select Product Category", ["PCs", "Laptops", "Consoles"])

        # Choose the appropriate dataframe based on selection
        if dataset_selection == "PCs":
            df = df_pcs
            st.sidebar.write("PC Sales Analytics")
        elif dataset_selection == "Laptops":
            df = df_laptop
            st.sidebar.write("Laptop Sales Analytics")
        else:
            df = df_console
            st.sidebar.write("Console Sales Analytics")

        visualization = st.sidebar.selectbox("Select Visualization",
                                             ["Monthly Components Sold", "Sales Trends", "Market Share",
                                              "Sales Distribution", "Sales by Brand", "Seasonal Trends",
                                              "Forecast Sales", "Clustering", "Elbow Method for Optimal k",
                                              "3D KMeans Clustering", ], )

        # Display selected visualization
        if visualization == "Monthly Components Sold":
            st.plotly_chart(monthly_components_sold(df))

        elif visualization == "Sales Trends":
            fig1, fig2 = analyze_sales_trends(df)
            st.plotly_chart(fig1)
            st.plotly_chart(fig2)

        elif visualization == "Seasonal Trends":
            st.plotly_chart(analyze_seasonal_trends(df))

        elif visualization == "Market Share":
            st.plotly_chart(analyze_market_share(df))

        elif visualization == "Sales Distribution":
            st.plotly_chart(analyze_sales_distribution(df))

        elif visualization == "Sales by Brand":
            st.plotly_chart(analyze_sales_by_brand(df))

        elif visualization == "Forecast Sales":
            show_forecasts(df)

        elif visualization == "Clustering":
            num_clusters = st.sidebar.slider("Select Number of Clusters", 2, 10, 3)
            st.plotly_chart(kmeans_clustering(df, num_clusters))

        elif visualization == "Elbow Method for Optimal k":
            st.sidebar.write("Elbow Method: Visualize the optimal number of clusters")
            st.write("This graph helps you identify the best number of clusters (k) for your data.")
            plot_elbow_method(df)  # Plot the Elbow Method graph

        elif visualization == "3D KMeans Clustering":
            num_clusters = st.sidebar.slider("Select Number of Clusters", 2, 10, 3)
            fig = kmeans_clustering_with_metrics(df, num_clusters)  # Use the KMeans 3D clustering function
            st.plotly_chart(fig)


# Run the app
if __name__ == "__main__":
    main()
