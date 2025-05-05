import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Set page config
st.set_page_config(page_title="E-Commerce Dashboard", layout="wide")

# Set color palette
custom_palette = sns.color_palette("crest")
sns.set_palette(custom_palette)

# Load dataset
def load_data():
    df = pd.read_csv("cleaned_ecommerce_data.csv")
    return df

df = load_data()

# --- Intro ---
st.title("Pakistan E-Commerce Analysis Dashboard")

st.markdown("""
Welcome to the **Pakistan E-Commerce Dashboard**, built using real transactional data from Pakistan's largest e-commerce dataset available on [Kaggle](https://www.kaggle.com/datasets/zusmani/pakistans-largest-ecommerce-dataset/data).
This project was developed as a **final personal project** for a Data Analyst course.

The primary goal is to equip you with the skills needed to analyze, interpret, and derive meaningful conclusions from a large dataset.  


The project is divided into two key parts:
- **Data Analysis**: Deep exploration of sales, customers, and order behavior.
- **Market Understanding & Strategic Recommendations**: Actionable insights for marketing and business strategy.
""")

# --- Tabs ---
tabs = st.tabs(["Dashboard", "Insights", "Recommendations"])

# --- Dashboard Tab ---
with tabs[0]:
    st.header("Dashboard Overview")

    # KPIs Calculation
    returned_qty = df[df['status'].isin(['refund', 'order_refunded'])]['qty_ordered'].sum()
    eligible_qty = df[df['status'].isin(['refund', 'order_refunded', 'complete', 'closed'])]['qty_ordered'].sum()
    return_rate = returned_qty / eligible_qty

    non_completed_statuses = ['canceled', 'fraud', 'holded', 'payment_review', 'pending', 'pending_paypal']
    non_completed_qty = df[df['status'].isin(non_completed_statuses)]['qty_ordered'].sum()
    total_orders = df['increment_id'].nunique()
    non_completed_rate = non_completed_qty / total_orders

    total_abs = df['abs_total'].sum()
    aov = total_abs / total_orders
    number_of_orders = total_orders

    # Cards layout
    st.subheader("Key Metrics Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Number of Orders", f"{number_of_orders:,}")
    col2.metric("Return Rate", f"{return_rate:.2%}")
    col3.metric("Non-Completed Orders Rate", f"{non_completed_rate:.2%}")
    col4.metric("Average Order Value (PKR)", f"{aov:,.0f}")


    # Bar Graphs ---
    st.subheader("Top Payment Methods Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top 5 Payment Methods by Median Order Value**")

        if 'payment_method' in df.columns and 'abs_total' in df.columns and 'increment_id' in df.columns:
            median_per_method = (
                df.groupby(['payment_method', 'increment_id'])['abs_total'].sum()
                .groupby('payment_method')
                .median()
                .sort_values(ascending=False)
                .head(5)
                .sort_values()
            )

            fig1, ax1 = plt.subplots(figsize=(5, 3))
            bars1 = ax1.barh(median_per_method.index, median_per_method.values, color=sns.color_palette("crest", n_colors=5))

            for bar in bars1:
                ax1.text(bar.get_width() + 300, bar.get_y() + bar.get_height() / 2,
                         f"{int(bar.get_width()):,}", va='center')

            ax1.set_xlabel("Median Order Value (PKR)")
            ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x / 1000)}K'))
            ax1.set_title("")
            sns.despine(top=True, right=True)
            st.pyplot(fig1)
        else:
            st.warning("Missing required columns for median calc.")

    with col2:
        st.markdown("**Top 5 Payment Methods by Number of Orders**")

        if 'payment_method' in df.columns and 'increment_id' in df.columns:
            orders_per_method = (
                df.groupby(['payment_method', 'increment_id']).size()
                .reset_index(name='order_count')
                .groupby('payment_method')['increment_id']
                .count()
                .sort_values(ascending=False)
                .head(5)
                .sort_values()
            )

            fig2, ax2 = plt.subplots(figsize=(5, 3))
            bars2 = ax2.barh(orders_per_method.index, orders_per_method.values, color=sns.color_palette("crest", n_colors=5))

            for bar in bars2:
                ax2.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
                         f"{int(bar.get_width()):,}", va='center')

            ax2.set_xlabel("Number of Orders")
            ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x / 1000)}K'))

            ax2.set_title("")
            sns.despine(top=True, right=True)
            st.pyplot(fig2)
        else:
            st.warning("Missing required columns for order count.")

    # Line Chart: Non-Completed Orders Rate Over Time
    st.subheader("Non-Completed Orders Rate Over Time")

    if 'status' in df.columns and 'qty_ordered' in df.columns and 'increment_id' in df.columns and 'created_at' in df.columns:

        df['created_at'] = pd.to_datetime(df['created_at'])
        df['order_month'] = df['created_at'].dt.to_period('M').dt.to_timestamp()

        non_completed_statuses = ['canceled', 'fraud', 'holded', 'payment_review', 'pending', 'pending_paypal']

        monthly_data = df.groupby('order_month').apply(
            lambda g: pd.Series({
                'non_completed_qty': g[g['status'].isin(non_completed_statuses)]['qty_ordered'].sum(),
                'total_orders': g['increment_id'].nunique()
            })
        ).reset_index()

        monthly_data['non_completed_rate'] = (monthly_data['non_completed_qty'] / monthly_data['total_orders']) * 100

        monthly_data = monthly_data.sort_values('order_month')
        monthly_data['month_label'] = monthly_data['order_month'].dt.strftime('%b-%y').str.upper()

        median_rate = monthly_data['non_completed_rate'].median()

        z = np.polyfit(range(len(monthly_data)), monthly_data['non_completed_rate'], 1)
        p = np.poly1d(z)

        fig, ax = plt.subplots(figsize=(10, 2.5))
        ax.plot(monthly_data['month_label'], monthly_data['non_completed_rate'], marker='o', label='Non-Completed Rate')
        ax.plot(monthly_data['month_label'], p(range(len(monthly_data))), linestyle='--', color='orange',
                label='Trend Line')
        ax.axhline(median_rate, color='gray', linestyle=':', label=f'Median: {median_rate:.2f}%')

        ax.set_ylabel("Rate (%)")
        ax.set_xlabel("Month")
        ax.set_title("Monthly Non-Completed Orders Rate")
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        sns.despine()

        st.pyplot(fig)

    else:
        st.warning("Missing required columns in dataset.")

    # Heatmap: Non-Completed Orders by Status and Payment Method
    st.subheader("Heatmap of Non-Completed Orders by Status and Payment Method")

    non_completed_statuses = ['canceled', 'fraud', 'holded', 'payment_review', 'pending', 'pending_paypal']
    non_completed_df = df[df['status'].isin(non_completed_statuses)]

    pivot = (
        non_completed_df
        .groupby(['status', 'payment_method'])['increment_id']
        .nunique()
        .reset_index()
        .pivot(index='status', columns='payment_method', values='increment_id')
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(10, 2.5))

    sns.heatmap(
        pivot,
        annot=False,  # בלי מספרים
        cmap="crest",  # הצבעים שלך
        linewidths=0.4,
        linecolor='white',
        cbar=False  # בלי סרגל צבע
    )

    ax.set_title("Non-Completed Orders by Status and Payment Method", fontsize=10)
    ax.set_xlabel("Payment Method", fontsize=9)
    ax.set_ylabel("Order Status", fontsize=9)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(fontsize=7)

    sns.despine()
    st.pyplot(fig)

# Insights Tab
with tabs[1]:
    st.header("Insights")
    # Operational Summary Insights
    st.markdown("""
    ---

    ### Operational Insights Summary

    - **Only 32.1%** of the total 408,619 orders were actually completed.
    - The **return rate** among completed orders stands at **23.07%**, indicating significant post-purchase issues.
    - A noticeable spike in order returns occurred in **late 2018**, suggesting potential fulfillment or product satisfaction concerns.

    ---

    ### Status & Payment Correlation

    - The majority of non-completed orders are associated with the status **`canceled`**.
    - Some statuses like `holded` or `payment_review` appear to be remnants from **legacy flows** or temporary platform logic.

    ---

    ### Payment Method Behavior

    - **COD (Cash on Delivery)** is the most commonly used method — but it's linked to **low median order value** and **high cancellation rates**.
    - **Bank Alfalah** and **MyGateway** are more commonly associated with **higher-value transactions**.
    - **Finance Settlement** stands out with the **highest median order value**, but has very few orders — possibly used for **B2B or bulk transactions**.

    ---

    ### Final Thought

    67.9% of all orders were **not completed**, and nearly a quarter of completed ones were returned.  
    This suggests an urgent need to revisit **order flow**, **payment method design**, and **post-order experiences**.

    ---
    """)

# Recommendations Tab
with tabs[2]:
    st.header("Recommendations")
    # --- Strategic Recommendations ---
    # --- Strategic Recommendations (Full Version) ---
    st.markdown("""
    ---

    ###  Strategic Recommendations

    - **Reassess the use of COD (Cash on Delivery)** for high-risk or high-value products.
    - **Encourage digital payment adoption** by offering incentives such as discounts or loyalty points.
    - **Explore expanding the use of Finance Settlement** for premium segments or B2B customers, given its strong association with high median order value.
    - **Analyze return reasons** to pinpoint root causes — whether related to product issues, delivery delays, or customer expectations.
    - **Introduce a “Try Before You Buy” policy** for specific categories to reduce post-purchase friction and return rates.
    - **Strengthen monitoring of intermediate statuses** like `payment_review` and `holded`, which may reflect abandoned flows or system issues.
    - **Eliminate rarely-used or outdated statuses** from the operational pipeline to simplify tracking and reporting.
    - **Refine fulfillment policies** to reduce time-to-delivery and enhance customer satisfaction.
    - **Implement more precise return reason tagging**, allowing clearer classification and smarter corrective action.
    - **Enhance post-purchase communication** (e.g., SMS/Email confirmations, delivery tracking) to improve confidence and reduce cancellations.
    - **Pilot pre-payment only flows** for high-value, customized, or limited-edition products, reducing the risk of unpaid deliveries.

    ---
    """)

# --- Project Links ---
st.markdown("""
---

### 🔗 Project Links

- 📊 [Tableau Dashboard – Operations Overview](https://public.tableau.com/app/profile/sapir.cardona/viz/E-commerceOperationsPerformanceOverview/E-commerceOperationsPerformanceOverview?publish=yes)
- 🖼️ [Presentation – Executive Overview](https://gamma.app/docs/-e8nim3ymwwd1cbf)
- 📄 [Summary Report – Operational Insights](https://gamma.app/docs/-e5mzjzu6ehx9rkd)
- 🧹 [Kaggle Notebook – Data Cleaning & EDA](https://www.kaggle.com/code/sapircardona/pakistan-s-largest-e-commerce-dataset-eda-sapir)

---
""")

