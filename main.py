import streamlit as st
import joblib
import numpy as np

# Ensure this is the first Streamlit command
st.set_page_config(
    page_title="Danesh Order Prediction App",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS to style the page
st.markdown(
    """
    <style>
    /* Set the background color to black */
    .stApp {
        background-color: black;
    }

    /* Style the buttons with the specific red color */
    .stButton > button {
        background-color: rgb(233,66,85); /* Logo color */
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
    }

    /* Style the input boxes with light gray background and red border */
    input[type=number] {
        background-color: #333333; /* Dark gray background */
        border: 2px solid rgb(233,66,85); /* Logo color */
        border-radius: 5px;
        padding: 5px;
        color: white;
    }

    /* Style the expander headers with black background and white text */
    .stExpander > div > div:first-child {
        background-color: black;
        color: white;
        border: 1px solid rgb(233,66,85); /* Red border */
        border-radius: 5px;
    }

    /* Style the expander content with white background and black text */
    .stExpander > div > div:nth-child(2) {
        background-color: black; /* White background for content */
        color: white !important; /* Black text inside content */
        border: 1px solid rgb(233,66,85); /* Red border */
        border-radius: 5px;
    }

    /* Style the titles, headers, and labels with white text */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown p {
        color: white;
    }

    /* Style for input labels */
    .stMarkdown label {
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Load the model from the pkl file
model = joblib.load('models/model_et_order_danesh.pkl')


# Define the Streamlit app
def main():
    # Add the business logo to the top of the dashboard
    st.image(r"images/logo2-removebg-preview.png", use_column_width=False, width=200)

    # Add the title with white color using HTML
    st.markdown("<h1 style='color: white;'>Danesh Order Prediction App</h1>", unsafe_allow_html=True)

    st.write("<p style='color: white;'>Enter feature values to get the predicted order count.</p>",
             unsafe_allow_html=True)

    # Group 1: Order Details
    with st.expander("Order Details"):
        col1, col2, col3 = st.columns(3)
        with col1:
            voucher_order = st.number_input("Voucher Order", value=0, step=1)
            percentage_discount = st.number_input("Percentage Discount", value=0.0, format="%.5f")
            discount = st.number_input("Discount", value=0 , step=1)
            order_cycle_time = st.number_input("Order Cycle Time (Oct)", value=0.0, format="%.5f")
        with col2:
            credit_bnpl_order = st.number_input("Credit BNPL Order", value=0, step=1)
            voucher_order_free_shipping = st.number_input("Voucher Order Free Shipping", value=0, step=1)
            shipping_fee = st.number_input("Shipping Fee(IRR)", value=0, step=1)
        with col3:
            share_app_customer = st.number_input("Share App Customer Among Customer", value=0.0, format="%.4f")
            active_customer_3_month = st.number_input("Active Customer 3 Month", value=0, step=1)

    # Group 2: Customer Insights
    with st.expander("Customer Insights"):
        col1, col2, col3 = st.columns(3)
        with col1:
            dk_monthly_atp = st.number_input("DK Monthly ATP Short Tail Item", value=0.0, format="%.3f")
            oct_avg_past_30_days = st.number_input("Oct Avg Past 30 Days", value=0.0, format="%.8f")
            campaign_score = st.number_input("Campaign Score", value=0, step=1)
        with col2:
            new_customer_free_shipping = st.number_input("Is New Customer Free Shipping", value=0, step=1)
            holiday = st.number_input("Holiday", value=0, step=1)
            holiday_type_count = st.number_input("Holiday Type Count", value=0, step=1)
        with col3:
            persian_day_number_of_week = st.number_input("Persian Day Number of Week", value=0, step=1)
            persian_decade_of_month = st.number_input("Persian Decade of Month", value=0, step=1)
            persian_semester = st.number_input("Persian Semester", value=0, step=1)

    # Group 3: Environmental Factors
    with st.expander("Environmental Factors"):
        col1, col2, col3 = st.columns(3)
        with col1:
            persian_season = st.number_input("Persian Season", value=0, step=1)
            persian_month_number = st.number_input("Persian Month Number", value=0, step=1)
            corona = st.number_input("Corona", value=0, step=1)
            dollar = st.number_input("Dollar", value=0.0, format="%.2f")
        with col2:
            tv = st.number_input("TV", value=0.0, format="%.2f")
            weather_clear = st.number_input("Is Weather Clear", value=0, step=1)
            return_policy = st.number_input("Return Policy", value=0.0, format="%.2f")
        with col3:
            plus_subscription = st.number_input("Plus Subscription", value=0.0, format="%.2f")
            plus_sub_sum = st.number_input("Plus Sub Sum Past 30 Days", value=0.0, format="%.2f")
            dollar_trend = st.number_input("Dollar Trend", value=0.0, format="%.2f")


    # Prediction button
    if st.button("Predict Order Count"):
        # Gather all the input data in the correct order
        features = [
            voucher_order, percentage_discount, discount, order_cycle_time,
            credit_bnpl_order, voucher_order_free_shipping, shipping_fee,
            share_app_customer, active_customer_3_month, dk_monthly_atp,
            oct_avg_past_30_days, campaign_score, new_customer_free_shipping,
            holiday, holiday_type_count, persian_day_number_of_week,
            persian_decade_of_month, persian_semester, persian_season,
            persian_month_number, corona, tv, weather_clear,
            return_policy, plus_subscription, plus_sub_sum, dollar_trend, dollar
        ]

        # Convert input features into a numpy array
        input_data = np.array(features).reshape(1, -1)

        # Make a prediction
        prediction = model.predict(input_data)

        # Display the prediction
        st.success(f"Predicted Order Count: {prediction[0]}")


if __name__ == "__main__":
    main()
