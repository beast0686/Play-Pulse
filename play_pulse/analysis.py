from typing import Protocol
from dataclasses import dataclass

import numpy as np
from numpy import format_float_scientific
import pandas as pd
import streamlit as st
import plotly.express as px
from plotly.graph_objs import Figure
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go


class MonthlyComponentsSoldAnalysis:
    name: str = "Monthly Components Sold"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare the DataFrame by ensuring necessary columns are present and grouping data by month and component."""
        if "Component" in df.columns:
            df["Component"] = df["Component"].astype(str)
        return df

    def plot(self, df: pd.DataFrame) -> Figure:
        """Generate and return a Plotly line graph for monthly sales analysis."""
        if "Component" in df.columns:
            components = df["Component"].unique()
            metric = "Qty"  # Default metric for plotting

            # Generate the first component's plot as an example
            component_data = df[df["Component"] == components[0]]
            fig = px.line(
                component_data,
                x="Month",
                y=metric,
                color="Brand",
                title=f"{components[0]} - {metric} Analysis",
                markers=True,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )

            fig.update_layout(
                xaxis_title="Month",
                yaxis_title=metric,
                legend_title="Brand",
                height=500,
            )

            return fig
        else:
            # Default case for non-PC datasets
            grouped = df.groupby(["Month", "Brand"], as_index=False)["Qty"].sum()
            grouped.rename(columns={"Qty": "Total Components Sold"}, inplace=True)

            top_brands = (
                grouped.groupby("Brand", as_index=False)["Total Components Sold"]
                .sum()
                .nlargest(10, "Total Components Sold")["Brand"]
            )

            filtered_grouped = grouped[grouped["Brand"].isin(top_brands)]

            fig = px.line(
                filtered_grouped,
                x="Month",
                y="Total Components Sold",
                color="Brand",
                title="Number of Components Sold Monthly",
                labels={"Total Components Sold": "Total Quantity"},
                markers=True,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )

            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Number of Components Sold",
                legend_title="Brand",
            )

            return fig

    def display(self, df: pd.DataFrame) -> None:
        """Display the analysis in the Streamlit app."""
        df = self.transform(df)

        if "Component" in df.columns:
            st.header("Monthly Sales Analysis by PC Components")

            components = df["Component"].unique()
            metric_tabs = st.tabs(["Quantity Sold", "Gross Sales", "Discount"])

            metrics = {
                "Quantity Sold": "Qty",
                "Gross Sales": "Gross Sales",
                "Discount": "Discount",
            }

            for idx, (tab, metric) in enumerate(metrics.items()):
                with metric_tabs[idx]:
                    for component in components:
                        st.subheader(f"{component} Analysis")
                        component_data = df[df["Component"] == component]

                        # Create line graph
                        fig = px.line(
                            component_data,
                            x="Month",
                            y=metric,
                            color="Brand",
                            title=f"{component} - {metric} Analysis",
                            markers=True,
                            color_discrete_sequence=px.colors.qualitative.Set2,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Add summary statistics
                        total = component_data[metric].sum()
                        avg = component_data[metric].mean()
                        top_brand = (
                            component_data.groupby("Brand")[metric].sum().idxmax()
                        )

                        col1, col2, col3 = st.columns(3)
                        col1.metric(
                            f"Total {metric.replace('_', ' ')}", f"{total:,.2f}"
                        )
                        col2.metric(
                            f"Average Monthly {metric.replace('_', ' ')}", f"{avg:,.2f}"
                        )
                        col3.metric(f"Top {metric.replace('_', ' ')} Brand", top_brand)
                        st.divider()
        else:
            st.plotly_chart(self.plot(df))


class SalesTrendsAnalysis:
    name: str = "Sales Trends"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare the DataFrame by summarizing sales and brand performance."""
        # Group sales data by Month
        sales_summary = df.groupby("Month", as_index=False).agg(
            Total_Sales=("Gross Sales", "sum"),
            Total_Quantity=("Qty", "sum"),
            Avg_Discount=("Discount", "mean"),
        )

        # Get the top 10 months with the highest sales
        sales_summary = sales_summary.nlargest(10, "Total_Sales")

        # Group sales data by Brand
        brand_performance = (
            df.groupby("Brand", as_index=False)
            .agg(
                Total_Sales=("Gross Sales", "sum"),
                Total_Discount=("Discount", "sum"),
                Total_Quantity=("Qty", "sum"),
            )
            .sort_values(by="Total_Sales", ascending=False)
        )

        # Get the top 10 brands with the highest sales
        brand_performance = brand_performance.nlargest(10, "Total_Sales")

        return sales_summary, brand_performance

    def plot(self, df: pd.DataFrame) -> Figure:
        """Create and return two Plotly figures for sales trends."""
        sales_summary, brand_performance = self.transform(df)

        # Create bar chart for monthly sales overview
        fig1 = px.bar(
            sales_summary,
            x="Month",
            y="Total_Sales",
            text="Total_Sales",
            title="Monthly Sales Overview",
            labels={"Total_Sales": "Total Sales", "Month": "Month"},
            color="Total_Sales",
            color_continuous_scale=px.colors.sequential.Blues,
        )
        fig1.update_traces(textposition="outside")
        fig1.update_layout(
            xaxis_title="Month", yaxis_title="Total Sales", showlegend=False
        )

        # Create pie chart for sales distribution by brand
        fig2 = px.pie(
            brand_performance,
            names="Brand",
            values="Total_Sales",
            title="Sales Distribution by Brand",
            labels={"Total_Sales": "Total Sales", "Brand": "Brand"},
        )
        fig2.update_traces(textinfo="percent+label")

        return fig1, fig2

    def display(self, df: pd.DataFrame) -> None:
        """Display the sales trends analysis in the Streamlit app."""
        fig1, fig2 = self.plot(df)

        st.subheader("Monthly Sales Overview")
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Sales Distribution by Brand")
        st.plotly_chart(fig2, use_container_width=True)


# Market Share Analysis
class MarketShareAnalysis:
    name: str = "Market Share"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate total sales by brand and select the top 10 brands."""
        # Group by Brand and calculate total sales
        brand_share = df.groupby("Brand", as_index=False).agg(
            Total_Sales=("Gross Sales", "sum")
        )

        # Sort by total sales and select the top 10 brands
        top_brands = brand_share.nlargest(10, "Total_Sales")

        return top_brands

    def plot(self, df: pd.DataFrame) -> Figure:
        """Create and return a Plotly pie chart for market share by top 10 brands."""
        top_brands = self.transform(df)

        # Create a pie chart for market share
        fig = px.pie(
            top_brands,
            names="Brand",
            values="Total_Sales",
            title="Market Share by Top 10 Brands",
            labels={"Total_Sales": "Total Sales"},
        )

        return fig

    def display(self, df: pd.DataFrame) -> None:
        """Display the market share analysis in the Streamlit app."""
        st.subheader("Market Share Analysis")
        fig = self.plot(df)
        st.plotly_chart(fig, use_container_width=True)


# Sales Distribution Analysis
class SalesDistributionAnalysis:
    name: str = "Sales Distribution"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare the DataFrame for plotting by ensuring 'Gross Sales' is present and valid."""
        # Ensure 'Gross Sales' is a numeric column
        df = df.dropna(subset=["Gross Sales"])
        df["Gross Sales"] = pd.to_numeric(df["Gross Sales"], errors="coerce")
        return df

    def plot(self, df: pd.DataFrame) -> Figure:
        """Create and return a Plotly histogram for sales distribution."""
        df = self.transform(df)

        # Create a histogram with a box plot marginal
        fig = px.histogram(
            df,
            x="Gross Sales",
            marginal="box",
            title="Sales Distribution",
            labels={"Gross Sales": "Sales", "count": "Number of Sales"},
        )

        return fig

    def display(self, df: pd.DataFrame) -> None:
        """Display the sales distribution analysis in the Streamlit app."""
        st.subheader("Sales Distribution Analysis")
        fig = self.plot(df)
        st.plotly_chart(fig, use_container_width=True)


# Sales by Brand Analysis
class SalesByBrandAnalysis:
    name: str = "Sales By Brand"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare the DataFrame by ensuring necessary columns are present and valid."""
        # Ensure 'Gross Sales' is numeric and drop rows with missing values in 'Brand' or 'Gross Sales'
        df = df.dropna(subset=["Brand", "Gross Sales"])
        df["Gross Sales"] = pd.to_numeric(df["Gross Sales"], errors="coerce")
        return df

    def plot(self, df: pd.DataFrame) -> Figure:
        """Create and return a Plotly box plot for sales by brand."""
        df = self.transform(df)

        # Create the box plot for sales by brand
        fig = px.box(
            df,
            x="Brand",
            y="Gross Sales",
            title="Sales by Brand",
            labels={"Gross Sales": "Sales", "Brand": "Brand"},
        )

        return fig

    def display(self, df: pd.DataFrame) -> None:
        """Display the sales by brand analysis in the Streamlit app."""
        st.subheader("Sales by Brand Analysis")
        fig = self.plot(df)
        st.plotly_chart(fig, use_container_width=True)


#  Seasonal Trends Analysis
class SeasonalTrendsAnalysis:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare the DataFrame by ensuring the 'Month' column is ordered and aggregating sales data."""
        # Ensure 'Month' is a categorical variable with a specific order
        df["Month"] = pd.Categorical(
            df["Month"],
            categories=[
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ],
            ordered=True,
        )

        # Aggregate total sales and quantity by month
        sales_trends = df.groupby("Month", as_index=False).agg(
            Total_Sales=("Gross Sales", "sum"), Total_Quantity=("Qty", "sum")
        )

        return sales_trends

    def plot(self, df: pd.DataFrame) -> Figure:
        """Create and return a Plotly line graph for seasonal sales trends."""
        sales_trends = self.transform(df)

        # Create the line chart
        fig = px.line(
            sales_trends,
            x="Month",
            y="Total_Sales",
            markers=True,
            title="Seasonal Sales Trends",
            labels={"Total_Sales": "Total Sales", "Month": "Month"},
        )

        fig.update_layout(xaxis_title="Month", yaxis_title="Total Sales")

        return fig

    def display(self, df: pd.DataFrame) -> None:
        """Display the seasonal sales trends in the Streamlit app."""
        st.subheader("Seasonal Sales Trends")
        fig = self.plot(df)
        st.plotly_chart(fig, use_container_width=True)


@dataclass
class ForecastSales:
    name: str = "Forecast Sales"

    def split_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Splits the data into training and testing sets."""
        train_size = int(len(df) * (8 / 12))  # 8-month training, 4-month testing
        train_data = df.iloc[:train_size]
        test_data = df.iloc[train_size:]
        return train_data, test_data

    def evaluate_forecast(
            self, actual: pd.Series, predicted: pd.Series
    ) -> dict[str, float | str]:
        """Calculates MAE and MAPE to evaluate model performance."""
        mae = mean_absolute_error(actual, predicted)
        mape = mean_absolute_percentage_error(actual, predicted) * 100
        return {"MAE": mae, "MAPE (%)": mape}

    def arima(self, df: pd.DataFrame, brand: str, steps: int) -> dict[str, pd.DataFrame]:
        brand_data = df[df["Brand"] == brand]
        sales_data = brand_data.groupby("Month")["Gross Sales"].sum()
        sales_data = sales_data.reset_index()
        sales_data["Month"] = pd.to_datetime(
            "2024 " + sales_data["Month"], format="%Y %B"
        )
        sales_data = sales_data.sort_values(by="Month")
        sales_data.set_index("Month", inplace=True)

        train_data, test_data = self.split_data(sales_data)

        # ARIMA model training and forecasting
        model = ARIMA(train_data["Gross Sales"], order=(5, 1, 1))
        model_fit = model.fit()

        # Forecasting
        forecast = model_fit.forecast(steps=len(test_data))
        forecast_accuracy = self.evaluate_forecast(test_data["Gross Sales"], forecast)

        # Extending the forecast
        extended_forecast = model_fit.forecast(steps=steps)
        future_months = pd.date_range(start="2025-01-01", periods=steps + 1, freq="ME")[1:]
        forecast_df = pd.DataFrame(
            {"Month": future_months, "Forecasted Sales": extended_forecast}
        )

        return {
            "sales_data": sales_data,
            "forecast_data": forecast_df,
            "accuracy": forecast_accuracy,
        }

    def sarima(
            self, df: pd.DataFrame, brand: str, steps: int
    ) -> dict[str, pd.DataFrame]:
        brand_data = df[df["Brand"] == brand]
        sales_data = brand_data.groupby("Month")["Gross Sales"].sum()
        sales_data = sales_data.reset_index()
        sales_data["Month"] = pd.to_datetime(
            "2024 " + sales_data["Month"], format="%Y %B"
        )
        sales_data = sales_data.sort_values(by="Month")
        sales_data.set_index("Month", inplace=True)

        train_data, test_data = self.split_data(sales_data)

        # SARIMA model training and forecasting
        model = SARIMAX(
            train_data["Gross Sales"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)
        )
        model_fit = model.fit(disp=False)

        # Forecasting
        forecast = model_fit.forecast(steps=len(test_data))
        forecast_accuracy = self.evaluate_forecast(test_data["Gross Sales"], forecast)

        # Extending the forecast
        extended_forecast = model_fit.forecast(steps=steps)
        future_months = pd.date_range(start="2025-01-01", periods=steps + 1, freq="M")[1:]
        forecast_df = pd.DataFrame(
            {"Month": future_months, "Forecasted Sales": extended_forecast}
        )

        return {
            "sales_data": sales_data,
            "forecast_data": forecast_df,
            "accuracy": forecast_accuracy,
        }

    def create_lagged_features(self, data: pd.DataFrame, lags: int = 3) -> pd.DataFrame:
        """Creates lagged features for the dataset."""
        df = data.copy()
        for lag in range(1, lags + 1):
            df[f"Lag_{lag}"] = df["Gross Sales"].shift(lag)
        return df.dropna()

    def random_forest(
            self, df: pd.DataFrame, brand: str, steps: int
    ) -> dict[str, pd.DataFrame]:
        brand_data = df[df["Brand"] == brand]
        sales_data = brand_data.groupby("Month")["Gross Sales"].sum()
        sales_data = sales_data.reset_index()
        sales_data["Month"] = pd.to_datetime(
            "2024 " + sales_data["Month"], format="%Y %B"
        )
        sales_data = sales_data.sort_values(by="Month")
        sales_data.set_index("Month", inplace=True)

        # Create lagged features
        sales_data = self.create_lagged_features(sales_data)

        # Train-test split
        X = sales_data.drop(columns=["Gross Sales"])
        y = sales_data["Gross Sales"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=4, shuffle=False
        )

        # Train the Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Forecasting on test data
        y_pred_test = model.predict(X_test)
        forecast_accuracy = {
            "MAE": mean_absolute_error(y_test, y_pred_test),
            "MAPE (%)": mean_absolute_percentage_error(y_test, y_pred_test) * 100,
        }

        # Extend forecast iteratively
        last_known_data = X.iloc[-1].values
        forecast_values = []
        for _ in range(steps):
            next_forecast = model.predict([last_known_data])[0]
            forecast_values.append(next_forecast)

            # Update lagged values
            last_known_data = np.roll(last_known_data, shift=-1)
            last_known_data[-1] = next_forecast

        # Generate future months
        future_months = pd.date_range(start="2025-01-01", periods=steps + 1, freq="M")[1:]
        forecast_df = pd.DataFrame(
            {"Month": future_months, "Forecasted Sales": forecast_values}
        )

        return {
            "sales_data": sales_data,
            "forecast_data": forecast_df,
            "accuracy": forecast_accuracy,
        }

    def plot(self, df: dict[str, pd.DataFrame]) -> Figure:
        sales_data = df["sales_data"]
        forecast_df = df["forecast_data"]

        fig = px.line(
            sales_data.reset_index(),
            x="Month",
            y="Gross Sales",
            title=f"Forecasting of Sales",
            labels={"Gross Sales": "Total Sales", "Month": "Month"},
            markers=True,
        )

        fig.add_scatter(
            x=forecast_df["Month"],
            y=forecast_df["Forecasted Sales"],
            mode="lines+markers",
            name="Forecasted Sales",
            line=dict(dash="dash", color="red"),
        )

        fig.update_layout(
            xaxis_title="Month", yaxis_title="Total Sales", showlegend=True
        )

        return fig


    def display(self, df: pd.DataFrame) -> None:
        brand = st.sidebar.selectbox(
            "Select Brand for Sales Forecast", df["Brand"].unique()
        )
        steps = st.sidebar.slider("Select Forecast Period", 1, 12, 6)
        method = st.sidebar.selectbox("Select Forecasting Method", ["ARIMA", "SARIMA", "Random Forest"])

        accuracy_metrics = None
        fig = None

        if method == "ARIMA":
            result = self.arima(df, brand, steps)
            fig = self.plot(result)
            accuracy_metrics = result["accuracy"]

        elif method == "SARIMA":
            result = self.sarima(df, brand, steps)
            fig = self.plot(result)
            accuracy_metrics = result["accuracy"]

        elif method == "Random Forest":
            result = self.random_forest(df, brand, steps)
            fig = self.plot(result)
            accuracy_metrics = result["accuracy"]

        # Display the forecast plot
        st.plotly_chart(fig)

        # Display the accuracy metrics in a table format
        if accuracy_metrics:
            metrics_data = {
                "Metric": ["Mean Absolute Error (MAE)", "Mean Absolute Percentage Error (MAPE)"],
                "Value": [accuracy_metrics["MAE"], accuracy_metrics["MAPE (%)"]],
                "Unit": ["Units", "%"],
            }
            metrics_df = pd.DataFrame(metrics_data)

            st.markdown(f"### {method} Model Accuracy")
            st.table(metrics_df.style.format({"Value": "{:,.2f}"}))


@dataclass
class Clustering:
    name: str = "Clustering"

    def kmeans(self, df: pd.DataFrame, num_clusters: int) -> dict[str, pd.DataFrame]:
        # Select relevant columns for clustering (e.g., 'Gross Sales' and 'Qty')
        data = df[["Gross Sales", "Qty"]].dropna()

        # Scale the data to normalize the feature ranges
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df["Cluster"] = kmeans.fit_predict(scaled_data)

        return df

    def plot_kmeans(self, df: pd.DataFrame) -> Figure:
        fig = px.scatter(
            df,
            x="Gross Sales",
            y="Qty",
            color="Cluster",
            title="KMeans Clustering of Sales Data",
            labels={"Gross Sales": "Total Sales", "Qty": "Quantity Sold"},
            color_continuous_scale="Viridis",
        )
        fig.update_layout(
            xaxis_title="Total Sales", yaxis_title="Quantity Sold", showlegend=True
        )
        return fig

    def plot_elbow(self, df: pd.DataFrame) -> None:
        # Select numeric columns for clustering (assuming df contains numerical data)
        X = df.select_dtypes(include=["float64", "int64"]).values

        # Handle missing values by imputing with the mean of each column
        imputer = SimpleImputer(strategy="mean")
        X_imputed = imputer.fit_transform(X)

        # List to hold the inertia values for different k
        inertia = []

        # Range of k values to try (1 to 10 clusters)
        k_range = range(1, 11)

        # Fit KMeans for different k values and record the inertia (within-cluster sum of squared distances)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_imputed)
            inertia.append(kmeans.inertia_)

        # Create an Elbow Method plot using Plotly
        fig = go.Figure()

        # Add the line plot
        fig.add_trace(
            go.Scatter(
                x=list(k_range),
                y=inertia,
                mode="lines+markers",
                marker=dict(color="blue"),
                line=dict(dash="dash"),
                name="Inertia",
            )
        )

        # Customize the layout
        fig.update_layout(
            title="Elbow Method for Optimal k",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Inertia (Sum of Squared Distances)",
            xaxis=dict(tickmode="linear"),
            template="plotly_white",
        )

        # Show the plot
        st.plotly_chart(fig)

    def plot_kmeans_3d(self, df: pd.DataFrame, num_clusters: int) -> Figure:
        # Select relevant columns
        data = df[["Gross Sales", "Qty", "Discount"]].dropna()

        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df["Cluster"] = kmeans.fit_predict(scaled_data)

        # 3D Scatter Plot for Clusters
        fig = px.scatter_3d(
            df,
            x="Gross Sales",
            y="Qty",
            z="Discount",
            color="Cluster",
            title=f"KMeans 3D Clustering with {num_clusters} Clusters",
            labels={
                "Gross Sales": "Total Sales",
                "Qty": "Quantity Sold",
                "Discount": "Discount",
            },
            color_continuous_scale="Viridis",
        )

        fig.update_layout(
            scene={
                "xaxis_title": "Total Sales",
                "yaxis_title": "Quantity Sold",
                "zaxis_title": "Discount",
            },
            showlegend=True,
        )
        return fig

    def display(self, df: pd.DataFrame) -> None:
        num_clusters = st.sidebar.slider("Select Number of Clusters", 2, 10, 3)
        method = st.sidebar.selectbox(
            "Select Clustering Method", ["KMeans", "Elbow", "3D Clustering"]
        )

        if method == "KMeans":
            df = self.kmeans(df, num_clusters)
            fig = self.plot_kmeans(df)
            st.plotly_chart(fig)

        elif method == "Elbow":
            self.plot_elbow(df)

        elif method == "3D Clustering":
            fig = self.plot_kmeans_3d(df, num_clusters)
            st.plotly_chart(fig)
