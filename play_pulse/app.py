import pandas as pd
import plotly.express as px
import streamlit as st

from .analysis import (MonthlyComponentsSoldAnalysis, SalesTrendsAnalysis, MarketShareAnalysis,
                       SalesDistributionAnalysis, SalesByBrandAnalysis, ForecastSales, Clustering, )
from .dashboard import Dashboard


@st.cache_data
def load_data() -> dict[str, pd.DataFrame]:
    return {"PC": pd.read_csv("datasets/pcs.csv"), "Laptop": pd.read_csv("datasets/laptops.csv"),
        "Console": pd.read_csv("datasets/consoles.csv"), }


class AnalysisDashboard(Dashboard):
    def display(self, dfs: dict[str, pd.DataFrame]) -> None:
        dataset = st.sidebar.radio("Dataset", list(dfs.keys()))
        super().display(dfs[dataset])


# Landing Page Class
class LandingPage:
    def __init__(self):
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

    def show(self):
        # Title and Introduction
        st.markdown('<h1 class="landing-title">Play Pulse</h1>', unsafe_allow_html=True)
        st.markdown('<h2 class="landing-subtitle">Comprehensive Sales Insights Across Product Categories</h2>',
            unsafe_allow_html=True, )

        # Overview Section
        self._feature_box("📊 Dashboard Overview", """
        Welcome to our comprehensive Sales Analytics Dashboard! 
        This interactive platform provides deep insights into sales performance across three key product categories:
        - Personal Computers (PCs)
        - Laptops
        - Gaming Consoles
        """, )

        # Key Features Section
        self._feature_box("🚀 Key Features", """
        - **Interactive Visualizations**: Explore sales data through multiple chart types
        - **Product Category Selection**: Switch between PCs, Laptops, and Consoles
        - **Comprehensive Analytics**:
            - Monthly Sales Trends
            - Brand Performance
            - Market Share Analysis
            - Profitability Insights
            - Sales Distribution
        """, )

        # Data Insights Teaser
        self._feature_box("💡 What You'll Discover", """
        Our dashboard helps you uncover:
        - Which brands are performing best
        - Monthly sales fluctuations
        - Impact of discounts on sales
        - Profit margins across different product lines
        """, )

        # Call to Action
        self._feature_box("🔍 Ready to Explore?", """
        Click on the menu to the left to start your data exploration journey. 
        Select a product category and choose from various visualization options to gain valuable insights!
        """, )

    def _feature_box(self, title, content):
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown(f"### {title}")
        st.write(content)
        st.markdown("</div>", unsafe_allow_html=True)


# Summary Analysis Class
class SummaryAnalysis:
    def __init__(self, data):
        self.data = data

    def _analyze_dataset(self, dataframe, category_name):
        # Sales Trends Analysis
        sales_summary = dataframe.groupby("Month", as_index=False).agg(Total_Sales=("Gross Sales", "sum"),
            Total_Quantity=("Qty", "sum"), Avg_Discount=("Discount", "mean"), )
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
            "total_quantity": total_quantity, "top_brand": top_brand, "most_profitable_brand": most_profitable_brand,
            "avg_discount": avg_discount, }

    def generate_summary_analysis(self):
        # Generate summaries for each dataset
        pc_summary = self._analyze_dataset(self.data["PC"], "Personal Computers")
        laptop_summary = self._analyze_dataset(self.data["Laptop"], "Laptops")
        console_summary = self._analyze_dataset(self.data["Console"], "Consoles")
        return pc_summary, laptop_summary, console_summary

    def show(self, pc_summary, laptop_summary, console_summary):
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
        self._cross_category_comparison(pc_summary, laptop_summary, console_summary)

    def _cross_category_comparison(self, pc_summary, laptop_summary, console_summary):
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


def run() -> None:
    dataframe = load_data()
    page = st.sidebar.radio("Navigate", ["Home", "Analysis", "Summary"])

    match page:
        case "Home":
            landing_page = LandingPage()
            landing_page.show()
        case "Analysis":
            static_analysis_dashboard = Dashboard(name="Sales Analysis")
            static_analysis_dashboard.add_component(MonthlyComponentsSoldAnalysis())
            static_analysis_dashboard.add_component(SalesTrendsAnalysis())
            static_analysis_dashboard.add_component(MarketShareAnalysis())
            static_analysis_dashboard.add_component(SalesDistributionAnalysis())
            static_analysis_dashboard.add_component(SalesByBrandAnalysis())

            advanced_analysis_dashboard = Dashboard(name="Advanced Analysis")
            advanced_analysis_dashboard.add_component(ForecastSales())
            advanced_analysis_dashboard.add_component(Clustering())

            analysis_dashboard = AnalysisDashboard(name="Analysis")
            analysis_dashboard.add_component(static_analysis_dashboard)
            analysis_dashboard.add_component(advanced_analysis_dashboard)
            analysis_dashboard.display(dataframe)
        case "Summary":
            summary_analysis = SummaryAnalysis(dataframe)
            pc_summary, laptop_summary, console_summary = summary_analysis.generate_summary_analysis()
            summary_analysis.show(pc_summary, laptop_summary, console_summary)
