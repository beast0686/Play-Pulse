import streamlit as st
import pandas as pd
import plotly.express as px

# Load the datasets
@st.cache_data
def load_data():
    df_laptop = pd.read_csv("datasets/laptops.csv")
    df_console = pd.read_csv("datasets/consoles.csv")
    df_pcs = pd.read_csv("datasets/pcs.csv")

    # Data type conversion function
    def set_types(dataframe: pd.DataFrame) -> None:
        dataframe['Qty'] = dataframe['Qty'].astype(int)
        dataframe['Gross Sales'] = dataframe['Gross Sales'].replace({',': ''}, regex=True).astype(float)
        dataframe['Discount'] = dataframe['Discount'].replace({',': ''}, regex=True).astype(float)
        dataframe['Net Sales With Tax'] = dataframe['Net Sales With Tax'].replace({',': ''}, regex=True).astype(float)
        dataframe['Tax Amount'] = dataframe['Tax Amount'].replace({',': ''}, regex=True).astype(float)
        dataframe['Net Sales Without Tax'] = dataframe['Net Sales Without Tax'].replace({',': ''}, regex=True).astype(
            float)
        dataframe['Target Sales Amount'] = dataframe['Target Sales Amount'].replace({',': ''}, regex=True).astype(float)

    set_types(df_laptop)
    set_types(df_console)
    set_types(df_pcs)

    return df_laptop, df_console, df_pcs


def monthly_components_sold(dataframe: pd.DataFrame) -> None:
    grouped = dataframe.groupby(["Month", "Brand"], as_index=False)["Qty"].sum()
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

# Sales Trends
def analyze_sales_trends(dataframe: pd.DataFrame):
    sales_summary = dataframe.groupby('Month', as_index=False).agg(
        Total_Sales=('Gross Sales', 'sum'),
        Total_Quantity=('Qty', 'sum'),
        Avg_Discount=('Discount', 'mean')
    )
    sales_summary = sales_summary.nlargest(10, 'Total_Sales')

    fig = px.bar(
        sales_summary,
        x='Month',
        y='Total_Sales',
        text='Total_Sales',
        title='Monthly Sales Overview',
        labels={'Total_Sales': 'Total Sales ($)', 'Month': 'Month'},
        color='Total_Sales',
        color_continuous_scale=px.colors.sequential.Blues
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Total Sales ($)',
        showlegend=False
    )

    brand_performance = dataframe.groupby('Brand', as_index=False).agg(
        Total_Sales=('Gross Sales', 'sum'),
        Total_Discount=('Discount', 'sum'),
        Total_Quantity=('Qty', 'sum')
    ).sort_values(by='Total_Sales', ascending=False)
    brand_performance = brand_performance.nlargest(10, 'Total_Sales')

    fig2 = px.pie(
        brand_performance,
        names='Brand',
        values='Total_Sales',
        title='Sales Distribution by Brand',
        labels={'Total_Sales': 'Total Sales ($)', 'Brand': 'Brand'}
    )
    fig2.update_traces(textinfo='percent+label')

    return fig, fig2

# Profitability Analysis
def analyze_profitability(dataframe: pd.DataFrame):
    dataframe['Profit Margin (%)'] = dataframe['Net Sales Without Tax'] / dataframe['Gross Sales'] * 100

    profitability = dataframe.groupby('Brand', as_index=False).agg(
        Avg_Profit_Margin=('Profit Margin (%)', 'mean')
    ).sort_values(by='Avg_Profit_Margin', ascending=False)

    fig = px.bar(
        profitability,
        x='Brand',
        y='Avg_Profit_Margin',
        text='Avg_Profit_Margin',
        title='Profitability by Brand',
        labels={'Avg_Profit_Margin': 'Average Profit Margin (%)', 'Brand': 'Brand'}
    )
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    return fig

# Market Share
def analyze_market_share(dataframe: pd.DataFrame):
    # Group by Brand and calculate total sales
    brand_share = dataframe.groupby('Brand', as_index=False).agg(
        Total_Sales=('Gross Sales', 'sum')
    )

    # Sort by total sales and select the top 10 brands
    top_brands = brand_share.nlargest(10, 'Total_Sales')

    # Create a pie chart for market share
    fig = px.pie(
        top_brands,
        names='Brand',
        values='Total_Sales',
        title='Market Share by Top 10 Brands',
        labels={'Total_Sales': 'Total Sales ($)'}
    )
    return fig

# Sales Distribution
def analyze_sales_distribution(dataframe: pd.DataFrame):
    fig = px.histogram(
        dataframe,
        x='Gross Sales',
        marginal='box',
        title='Sales Distribution',
        labels={'Gross Sales': 'Sales ($)', 'count': 'Number of Sales'}
    )
    return fig

# Sales by Brand Boxplot
def analyze_sales_by_brand(dataframe: pd.DataFrame):
    fig = px.box(
        dataframe,
        x='Brand',
        y='Gross Sales',
        title='Sales by Brand',
        labels={'Gross Sales': 'Sales ($)', 'Brand': 'Brand'}
    )
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
    """, unsafe_allow_html=True)

    # Title and Introduction
    st.markdown('<h1 class="landing-title">Sales Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="landing-subtitle">Comprehensive Sales Insights Across Product Categories</h2>',
                unsafe_allow_html=True)

    # Overview Section
    st.markdown('<div class="feature-box">', unsafe_allow_html=True)
    st.markdown('### 📊 Dashboard Overview')
    st.write("""
    Welcome to our comprehensive Sales Analytics Dashboard! 
    This interactive platform provides deep insights into sales performance across three key product categories:
    - Personal Computers (PCs)
    - Laptops
    - Gaming Consoles
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Key Features
    st.markdown('<div class="feature-box">', unsafe_allow_html=True)
    st.markdown('### 🚀 Key Features')
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
    st.markdown('</div>', unsafe_allow_html=True)

    # Data Insights Teaser
    st.markdown('<div class="feature-box">', unsafe_allow_html=True)
    st.markdown('### 💡 What You\'ll Discover')
    st.write("""
    Our dashboard helps you uncover:
    - Which brands are performing best
    - Monthly sales fluctuations
    - Impact of discounts on sales
    - Profit margins across different product lines
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Call to Action
    st.markdown('<div class="feature-box">', unsafe_allow_html=True)
    st.markdown('### 🔍 Ready to Explore?')
    st.write("""
    Click on the menu to the left to start your data exploration journey. 
    Select a product category and choose from various visualization options to gain valuable insights!
    """)
    st.markdown('</div>', unsafe_allow_html=True)


# Summary Analysis Function
def generate_summary_analysis(df_laptop, df_console, df_pcs):
    def analyze_dataset(dataframe, category_name):
        # Sales Trends Analysis
        sales_summary = dataframe.groupby('Month', as_index=False).agg(
            Total_Sales=('Gross Sales', 'sum'),
            Total_Quantity=('Qty', 'sum'),
            Avg_Discount=('Discount', 'mean')
        )
        max_sales_month = sales_summary.loc[sales_summary['Total_Sales'].idxmax(), 'Month']
        total_sales = sales_summary['Total_Sales'].sum()
        total_quantity = sales_summary['Total_Quantity'].sum()

        # Brand Performance
        brand_performance = dataframe.groupby('Brand', as_index=False).agg(
            Total_Sales=('Gross Sales', 'sum'),
            Total_Quantity=('Qty', 'sum')
        )
        top_brand = brand_performance.loc[brand_performance['Total_Sales'].idxmax(), 'Brand']

        # Profitability Analysis
        dataframe['Profit Margin (%)'] = dataframe['Net Sales Without Tax'] / dataframe['Gross Sales'] * 100
        profitability = dataframe.groupby('Brand', as_index=False).agg(
            Avg_Profit_Margin=('Profit Margin (%)', 'mean')
        )
        most_profitable_brand = profitability.loc[profitability['Avg_Profit_Margin'].idxmax(), 'Brand']

        # Discount Analysis
        avg_discount = dataframe['Discount'].mean()

        return {
            'category': category_name,
            'max_sales_month': max_sales_month,
            'total_sales': total_sales,
            'total_quantity': total_quantity,
            'top_brand': top_brand,
            'most_profitable_brand': most_profitable_brand,
            'avg_discount': avg_discount
        }

    # Generate summaries for each dataset
    pc_summary = analyze_dataset(df_pcs, 'Personal Computers')
    laptop_summary = analyze_dataset(df_laptop, 'Laptops')
    console_summary = analyze_dataset(df_console, 'Consoles')

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
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="summary-title">Sales Performance Summary</h1>', unsafe_allow_html=True)

    # Summaries for each category
    categories = [
        ('Personal Computers', pc_summary),
        ('Laptops', laptop_summary),
        ('Consoles', console_summary)
    ]

    for category_name, summary in categories:
        st.markdown(f'<div class="summary-box">', unsafe_allow_html=True)
        st.markdown(f'<h2 class="category-header">{category_name} Sales Analysis</h2>', unsafe_allow_html=True)

        st.markdown(f"""
        ### Key Insights:
        - **Total Sales**: ${summary['total_sales']:,.2f}
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
           The category generated total sales of ${summary['total_sales']:,.2f}, 
           with the peak sales month being {summary['max_sales_month']}. 
           A total of {summary['total_quantity']:,} units were sold.

        2. **Brand Dynamics**:
           {summary['top_brand']} emerged as the top-performing brand in terms of total sales, 
           while {summary['most_profitable_brand']} demonstrated the highest profit margins.

        3. **Discount Strategy**:
           The average discount across this category was {summary['avg_discount']:.2f}, 
           indicating the pricing and promotional strategies employed.
        """)

        st.markdown('</div>', unsafe_allow_html=True)

    # Overall Market Comparison
    st.markdown('<div class="summary-box">', unsafe_allow_html=True)
    st.markdown('<h2 class="category-header">Cross-Category Market Comparison</h2>', unsafe_allow_html=True)

    # Comparative Analysis
    comparison_data = pd.DataFrame([
        {'Category': 'Personal Computers', 'Total Sales': pc_summary['total_sales'],
         'Total Quantity': pc_summary['total_quantity']},
        {'Category': 'Laptops', 'Total Sales': laptop_summary['total_sales'],
         'Total Quantity': laptop_summary['total_quantity']},
        {'Category': 'Consoles', 'Total Sales': console_summary['total_sales'],
         'Total Quantity': console_summary['total_quantity']}
    ])

    # Create two columns for visualizations
    col1, col2 = st.columns(2)

    # Sales Comparison Bar Chart
    with col1:
        fig1 = px.bar(
            comparison_data,
            x='Category',
            y='Total Sales',
            title='Total Sales Comparison',
            labels={'Total Sales': 'Total Sales ($)'}
        )
        st.plotly_chart(fig1)

    # Quantity Comparison Bar Chart
    with col2:
        fig2 = px.bar(
            comparison_data,
            x='Category',
            y='Total Quantity',
            title='Total Quantity Sold Comparison',
            labels={'Total Quantity': 'Total Quantity'}
        )
        st.plotly_chart(fig2)

    st.markdown("""
    ### Overall Market Insights:
    - Comparative analysis reveals the sales performance and market dynamics across different product categories
    - The visualizations provide a quick overview of total sales and quantities sold
    - Each category shows unique characteristics in terms of sales volume and market penetration
    """)

    st.markdown('</div>', unsafe_allow_html=True)


# Modify the main function to include the summary page
def main():
    # Load the data
    df_laptop, df_console, df_pcs = load_data()

    # Generate summary analysis
    pc_summary, laptop_summary, console_summary = generate_summary_analysis(df_laptop, df_console, df_pcs)

    # Sidebar for page selection
    page = st.sidebar.radio(
        "Navigate",
        ["Home", "Sales Analytics", "Summary"]
    )

    # Conditional page rendering
    if page == "Home":
        show_landing_page()
    elif page == "Summary":
        # Show summary page with generated summaries
        show_summary_page(pc_summary, laptop_summary, console_summary)
    else:
        # Existing Sales Analytics Page (remains the same as in previous artifact)
        st.title('Sales Analytics Dashboard')

        # Sidebar for selecting dataset
        dataset_selection = st.sidebar.radio(
            "Select Product Category",
            ["PCs", "Laptops", "Consoles"]
        )

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

        # Visualization Selection
        visualization = st.sidebar.selectbox(
            "Select Visualization",
            [
                "Monthly Components Sold",
                "Sales Trends",
                "Market Share",
                "Sales Distribution",
                "Sales by Brand",
                "Profitability Analysis"
            ]
        )

        # Display selected visualization
        if visualization == "Monthly Components Sold":
            st.plotly_chart(monthly_components_sold(df))

        elif visualization == "Sales Trends":
            fig1, fig2 = analyze_sales_trends(df)
            st.plotly_chart(fig1)
            st.plotly_chart(fig2)

        elif visualization == "Profitability Analysis":
            st.plotly_chart(analyze_profitability(df))

        elif visualization == "Market Share":
            st.plotly_chart(analyze_market_share(df))

        elif visualization == "Sales Distribution":
            st.plotly_chart(analyze_sales_distribution(df))

        elif visualization == "Sales by Brand":
            st.plotly_chart(analyze_sales_by_brand(df))


# Run the app
if __name__ == "__main__":
    main()
