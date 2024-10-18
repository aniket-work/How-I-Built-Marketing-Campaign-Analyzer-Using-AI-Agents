import streamlit as st
from loguru import logger


def run_streamlit_app(graph):
    st.title("AI Marketing Campaign Analyzer")

    product_name = st.text_input("Product Name:")
    product_type = st.text_input("Product Type:")
    features = st.text_area("Key Features:")
    target_audience = st.text_input("Target Audience:")
    usp = st.text_area("Unique Selling Proposition:")

    if st.button("Analyze"):
        if product_name and product_type and features and target_audience and usp:
            product_info = {
                "product_name": product_name,
                "product_type": product_type,
                "features": features,
                "target_audience": target_audience,
                "usp": usp
            }
            logger.info(f"Analyzing product: {product_info}")
            results = graph.invoke({"product_info": product_info})

            st.subheader("Marketing Expert 1's Strategy:")
            st.write(results["speaker1_strategy"].content)

            st.subheader("Marketing Expert 2's Strategy:")
            st.write(results["speaker2_strategy"].content)

            st.subheader("Judge's Decision:")
            st.write(results["judge_decision"].content)
        else:
            st.error("Please fill in all product details before analyzing.")
            logger.warning("Analyze button clicked without complete product information")