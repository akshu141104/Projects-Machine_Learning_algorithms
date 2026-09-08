
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🏠 House Price Predictor")
st.write(
    "Enter the house details below to predict the estimated house price."
)


# =========================================================
# CSV UPLOAD
# =========================================================

st.sidebar.header("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload train.csv",
    type=["csv"]
)


# =========================================================
# IF CSV IS NOT UPLOADED
# =========================================================

if uploaded_file is None:

    st.info(
        "👆 Please upload your Ames Housing train.csv file "
        "from the sidebar to start prediction."
    )

    st.stop()


# =========================================================
# READ CSV
# =========================================================

try:

    data = pd.read_csv(uploaded_file)

    st.sidebar.success("✅ Dataset uploaded successfully!")

except Exception as e:

    st.error("❌ Error while reading CSV file.")
    st.write(e)
    st.stop()


# =========================================================
# REQUIRED FEATURES
# =========================================================

prediction_features = [

    "OverallQual",
    "GrLivArea",
    "TotalBsmtSF",
    "GarageCars",
    "BedroomAbvGr",
    "FullBath",
    "YearBuilt",
    "LotArea",
    "YearRemodAdd"

]


# =========================================================
# CHECK REQUIRED COLUMNS
# =========================================================

required_columns = prediction_features + ["SalePrice"]

missing_columns = [

    column
    for column in required_columns
    if column not in data.columns

]


if missing_columns:

    st.error("❌ Required columns are missing from the CSV file.")

    st.write("Missing columns:")

    st.write(missing_columns)

    st.stop()


# =========================================================
# HANDLE MISSING VALUES
# =========================================================

for column in prediction_features:

    data[column] = data[column].fillna(
        data[column].median()
    )


# =========================================================
# MACHINE LEARNING MODEL
# =========================================================

X = data[prediction_features]

y = np.log1p(
    data["SalePrice"]
)


# Ridge Regression Model

prediction_model = Ridge(
    alpha=10.0
)


# Train Model

prediction_model.fit(
    X,
    y
)


# =========================================================
# PREDICTION SECTION
# =========================================================

st.header("🏡 Predict House Price")

st.write(
    "Enter the following details about the house."
)


# =========================================================
# INPUT FIELDS
# =========================================================

col1, col2 = st.columns(2)


# ---------------------------------------------------------
# LEFT COLUMN
# ---------------------------------------------------------

with col1:

    overall_qual = st.number_input(
        "⭐ Overall Quality (1-10)",
        min_value=1,
        max_value=10,
        value=5,
        step=1
    )


    gr_liv_area = st.number_input(
        "📐 Living Area (sq ft)",
        min_value=100,
        max_value=10000,
        value=1500,
        step=50
    )


    total_bsmt_sf = st.number_input(
        "🏠 Basement Area (sq ft)",
        min_value=0,
        max_value=5000,
        value=800,
        step=50
    )


    garage_cars = st.number_input(
        "🚗 Garage Cars",
        min_value=0,
        max_value=5,
        value=2,
        step=1
    )


    bedroom_abv_gr = st.number_input(
        "🛏️ Bedrooms",
        min_value=0,
        max_value=10,
        value=3,
        step=1
    )


# ---------------------------------------------------------
# RIGHT COLUMN
# ---------------------------------------------------------

with col2:

    full_bath = st.number_input(
        "🚿 Full Bathrooms",
        min_value=0,
        max_value=5,
        value=2,
        step=1
    )


    year_built = st.number_input(
        "📅 Year Built",
        min_value=1800,
        max_value=2026,
        value=2000,
        step=1
    )


    lot_area = st.number_input(
        "🌳 Lot Area (sq ft)",
        min_value=100,
        max_value=100000,
        value=8000,
        step=100
    )


    year_remod_add = st.number_input(
        "🔨 Year Remodeled",
        min_value=1800,
        max_value=2026,
        value=2000,
        step=1
    )


# =========================================================
# PREDICT BUTTON
# =========================================================

st.divider()


predict_button = st.button(
    "🔮 Predict House Price",
    type="primary",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # Create DataFrame from user input

    input_data = pd.DataFrame(

        [[
            overall_qual,
            gr_liv_area,
            total_bsmt_sf,
            garage_cars,
            bedroom_abv_gr,
            full_bath,
            year_built,
            lot_area,
            year_remod_add
        ]],

        columns=prediction_features

    )


    # Predict log price

    log_prediction = prediction_model.predict(
        input_data
    )[0]


    # Convert log price to original price

    predicted_price = np.expm1(
        log_prediction
    )


    # Make sure price is positive

    predicted_price = max(
        1,
        predicted_price
    )


    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    st.success(
        "🎉 House Price Prediction Completed!"
    )


    st.metric(
        label="🏠 Estimated House Price",
        value=f"${predicted_price:,.2f}"
    )


    st.write(
        "The above value is the estimated price "
        "predicted by the Machine Learning model."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "House Price Prediction | Ames Housing Dataset | "
    "Ridge Regression"
)