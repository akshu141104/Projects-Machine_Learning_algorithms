import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib.pyplot as plt


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction")
st.write("Ames Housing Dataset - Machine Learning Project")


# =========================================================
# UPLOAD DATASET
# =========================================================

st.sidebar.header("Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload train.csv",
    type=["csv"]
)


if uploaded_file is None:
    st.info("👆 Please upload your Ames Housing train.csv file from the sidebar.")
    st.stop()


# Read CSV
df = pd.read_csv(uploaded_file)

st.success("Dataset loaded successfully!")


# =========================================================
# CHECK TARGET
# =========================================================

if "SalePrice" not in df.columns:
    st.error("❌ SalePrice column is not present in the dataset.")
    st.stop()


# =========================================================
# DATASET OVERVIEW
# =========================================================

st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

with col3:
    st.metric(
        "Average Sale Price",
        f"${df['SalePrice'].mean():,.0f}"
    )


with st.expander("View Dataset"):
    st.dataframe(df.head(10), use_container_width=True)


# =========================================================
# DATA PREPROCESSING
# =========================================================

data = df.copy()


# Fill categorical missing values
categorical_cols = data.select_dtypes(include="object").columns

data[categorical_cols] = data[categorical_cols].fillna("None")


# Fill numerical missing values
numerical_cols = data.select_dtypes(include=np.number).columns

data[numerical_cols] = data[numerical_cols].fillna(
    data[numerical_cols].median()
)


# =========================================================
# FEATURE ENGINEERING
# =========================================================

# Total house area
data["TotalSF"] = (
    data["TotalBsmtSF"]
    + data["1stFlrSF"]
    + data["2ndFlrSF"]
)


# House age
data["HouseAge"] = (
    data["YrSold"]
    - data["YearBuilt"]
)


# Total bathrooms
data["TotalBathrooms"] = (
    data["FullBath"]
    + 0.5 * data["HalfBath"]
    + data["BsmtFullBath"]
    + 0.5 * data["BsmtHalfBath"]
)


# =========================================================
# SEPARATE X AND Y
# =========================================================

X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]


# =========================================================
# ONE HOT ENCODING
# =========================================================

X = pd.get_dummies(
    X,
    drop_first=True
)


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# =========================================================
# MODEL TRAINING
# =========================================================

linear_model = LinearRegression()

ridge_model = Ridge(alpha=1.0)

lasso_model = Lasso(
    alpha=1.0,
    max_iter=10000
)


linear_model.fit(X_train, y_train)

ridge_model.fit(X_train, y_train)

lasso_model.fit(X_train, y_train)


# =========================================================
# PREDICTIONS
# =========================================================

linear_pred = linear_model.predict(X_test)

ridge_pred = ridge_model.predict(X_test)

lasso_pred = lasso_model.predict(X_test)


# =========================================================
# MODEL EVALUATION
# =========================================================

def evaluate_model(y_true, prediction):

    rmse = np.sqrt(
        mean_squared_error(y_true, prediction)
    )

    mae = mean_absolute_error(
        y_true,
        prediction
    )

    r2 = r2_score(
        y_true,
        prediction
    )

    return rmse, mae, r2


linear_rmse, linear_mae, linear_r2 = evaluate_model(
    y_test,
    linear_pred
)

ridge_rmse, ridge_mae, ridge_r2 = evaluate_model(
    y_test,
    ridge_pred
)

lasso_rmse, lasso_mae, lasso_r2 = evaluate_model(
    y_test,
    lasso_pred
)


# =========================================================
# MODEL COMPARISON
# =========================================================

results = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression"
    ],

    "RMSE": [
        linear_rmse,
        ridge_rmse,
        lasso_rmse
    ],

    "MAE": [
        linear_mae,
        ridge_mae,
        lasso_mae
    ],

    "R2": [
        linear_r2,
        ridge_r2,
        lasso_r2
    ]

})


st.header("🤖 Model Performance")

st.dataframe(
    results.style.format({
        "RMSE": "{:,.2f}",
        "MAE": "{:,.2f}",
        "R2": "{:.4f}"
    }),
    use_container_width=True
)


# =========================================================
# BEST MODEL
# =========================================================

best_model_name = results.loc[
    results["R2"].idxmax(),
    "Model"
]


st.success(
    f"🏆 Best Model based on R²: {best_model_name}"
)


# =========================================================
# MODEL METRICS
# =========================================================

st.subheader("Linear Regression Performance")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "RMSE",
        f"{linear_rmse:,.2f}"
    )

with c2:
    st.metric(
        "MAE",
        f"{linear_mae:,.2f}"
    )

with c3:
    st.metric(
        "R² Score",
        f"{linear_r2:.4f}"
    )


# =========================================================
# ACTUAL VS PREDICTED
# =========================================================

st.header("📈 Actual vs Predicted Price")

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(
    y_test,
    linear_pred,
    alpha=0.5
)

ax.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

ax.set_xlabel("Actual SalePrice")

ax.set_ylabel("Predicted SalePrice")

ax.set_title(
    "Actual vs Predicted House Prices"
)

st.pyplot(fig)


# =========================================================
# RESIDUAL PLOT
# =========================================================

st.header("📉 Residual Plot")

residuals = y_test - linear_pred

fig2, ax2 = plt.subplots(figsize=(8, 6))

ax2.scatter(
    linear_pred,
    residuals,
    alpha=0.5
)

ax2.axhline(
    y=0,
    linestyle="--"
)

ax2.set_xlabel(
    "Predicted SalePrice"
)

ax2.set_ylabel(
    "Residuals"
)

ax2.set_title(
    "Residual Plot"
)

st.pyplot(fig2)


# =========================================================
# HOUSE PRICE PREDICTION
# =========================================================

st.header("🏡 Predict House Price")

st.write(
    "Enter the important house features below."
)


# =========================================================
# PREDICTION MODEL
# =========================================================
# This model is ONLY for the user prediction section.
# Existing Linear/Ridge/Lasso models above remain unchanged.

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

prediction_model = Ridge(alpha=10.0)

prediction_model.fit(
    data[prediction_features],
    np.log1p(data["SalePrice"])
)


# =========================================================
# USER INPUTS
# =========================================================

input_data = {}


if "OverallQual" in data.columns:

    input_data["OverallQual"] = st.number_input(
        "Overall Quality (1-10)",
        min_value=1,
        max_value=10,
        value=5
    )


if "GrLivArea" in data.columns:

    input_data["GrLivArea"] = st.number_input(
        "Living Area (sq ft)",
        min_value=100,
        max_value=10000,
        value=1500
    )


if "TotalBsmtSF" in data.columns:

    input_data["TotalBsmtSF"] = st.number_input(
        "Basement Area (sq ft)",
        min_value=0,
        max_value=5000,
        value=800
    )


if "GarageCars" in data.columns:

    input_data["GarageCars"] = st.number_input(
        "Garage Cars",
        min_value=0,
        max_value=5,
        value=2
    )


if "BedroomAbvGr" in data.columns:

    input_data["BedroomAbvGr"] = st.number_input(
        "Bedrooms",
        min_value=0,
        max_value=10,
        value=3
    )


if "FullBath" in data.columns:

    input_data["FullBath"] = st.number_input(
        "Full Bathrooms",
        min_value=0,
        max_value=5,
        value=2
    )


if "YearBuilt" in data.columns:

    input_data["YearBuilt"] = st.number_input(
        "Year Built",
        min_value=1800,
        max_value=2026,
        value=2000
    )


if "LotArea" in data.columns:

    input_data["LotArea"] = st.number_input(
        "Lot Area (sq ft)",
        min_value=100,
        max_value=100000,
        value=8000
    )


if "YearRemodAdd" in data.columns:

    input_data["YearRemodAdd"] = st.number_input(
        "Year Remodeled",
        min_value=1800,
        max_value=2026,
        value=2000
    )


# =========================================================
# PREDICT BUTTON
# =========================================================

if st.button(
    "🔮 Predict House Price",
    type="primary"
):

    # Create DataFrame using ONLY the values entered by user
    prediction_input = pd.DataFrame(
        [input_data],
        columns=prediction_features
    )


    # Predict log price
    log_prediction = prediction_model.predict(
        prediction_input
    )[0]


    # Convert log price back to original SalePrice
    predicted_price = np.expm1(
        log_prediction
    )


    # Make sure price is positive
    predicted_price = max(
        1,
        predicted_price
    )


    st.success(
        f"🏠 Estimated House Price: ${predicted_price:,.2f}"
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "House Price Prediction | Ames Housing Dataset | "
    "Linear, Ridge and Lasso Regression"
)