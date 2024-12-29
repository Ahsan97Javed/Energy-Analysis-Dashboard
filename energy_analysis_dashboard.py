# Libraries
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from PIL import Image
import plotly.graph_objects as go
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="Energy Analysis Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Visuals
st.markdown("""
<style>
    .main { background-color: #f0f2f6; }
    .sidebar { background-color: #2c3e50; color: white; }
    .sidebar .css-1lcbmhc { color: white; }
    .sidebar .css-1d391kg { background-color: #2c3e50; color: white; }
</style>
""", unsafe_allow_html=True)

# Load Data and Clean
data = pd.read_csv("Panel format.csv")
data["Year"] = pd.to_datetime(data["Year"], format="%Y")

data_cleaned = data.drop(columns=['coalprod_ej'])  # Dropping 'coalprod_ej' due to missing values
numeric_cols = data_cleaned.select_dtypes(include=['float64']).columns
data_cleaned[numeric_cols] = data_cleaned[numeric_cols].fillna(data_cleaned[numeric_cols].median())
categorical_cols = ['Region', 'SubRegion', 'Country']
for col in categorical_cols:
    data_cleaned[col] = data_cleaned[col].fillna(data_cleaned[col].mode()[0])
data_cleaned = data_cleaned.drop_duplicates()

banner_image = Image.open("Energy.jpg")
st.image(banner_image, use_container_width=True)

with st.sidebar:
    st.title("🔋 Energy Dashboard")
    sidebar_image = Image.open("Energy1.jpeg")
    st.sidebar.image(sidebar_image, use_container_width=True)
    country = st.sidebar.selectbox("Select a country", data_cleaned["Country"].unique())
    st.markdown("---")
           

# Filter data for the selected country
country_data = data_cleaned[data_cleaned["Country"] == country]

st.title(f"Energy and Population Analysis for {country}")

# Add Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Population Analysis", "Energy Analysis", "Correlation Analysis", "Machine Learning", "Global Data"])

# Tab 1: Population Analysis
with tab1:
    st.subheader("📊 Population Analysis Over Time")

    # Population Over Time with Trendline
    fig = px.line(country_data, x="Year", y="pop", title="Population Growth Over Time",
                  labels={'pop': 'Population'}, color_discrete_sequence=["#ad800e"])
    fig.update_traces(line=dict(width=3))
    fig.update_layout(
        title=dict(font=dict(size=20)),
        xaxis=dict(title="Year"),
        yaxis=dict(title="Population")
    )
    st.plotly_chart(fig, use_container_width=True)

    # Population Growth Rate with Highlight
    country_data['pop_growth'] = country_data['pop'].pct_change() * 100
    fig = px.bar(country_data, x="Year", y="pop_growth", title="Population Growth Rate Over Time",
                 labels={'pop_growth': 'Population Growth Rate (%)'}, color_discrete_sequence=["#EF553B"])
    fig.update_layout(
        title=dict(font=dict(size=20)),
        xaxis=dict(title="Year"),
        yaxis=dict(title="Growth Rate (%)")
    )
    st.plotly_chart(fig, use_container_width=True)


# Tab 2: Energy Analysis
with tab2:
    st.subheader("⚡ Energy Analysis (Fossil Fuels vs Renewables)")

    # Fossil Fuels vs Renewable Energy Over Time
    cols = st.columns(2)
    with cols[0]:
        fig = px.line(country_data, x="Year", y=['coalcons_ej', 'ren_power_ej'],
                      title="Fossil Fuels vs Renewable Energy Over Time",
                      labels={'coalcons_ej': 'Fossil Fuel Consumption (Exajoules)',
                              'ren_power_ej': 'Renewable Energy (Exajoules)'},
                      color_discrete_sequence=px.colors.sequential.Sunset)
        fig.update_layout(
            title=dict(font=dict(size=20)),
            xaxis=dict(title="Year"),
            yaxis=dict(title="Energy Consumption (Exajoules)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with cols[1]:
        country_data['renewable_share'] = (country_data['ren_power_ej'] / country_data['primary_ej']) * 100
        fig = px.area(country_data, x="Year", y="renewable_share",
                      title="Renewable Energy Share of Total Consumption",
                      labels={'renewable_share': 'Renewable Share (%)'},
                      color_discrete_sequence=px.colors.sequential.Blues)
        fig.update_layout(
            title=dict(font=dict(size=20)),
            xaxis=dict(title="Year"),
            yaxis=dict(title="Renewable Share (%)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # CO2 Emissions vs Fossil Fuel Consumption
    cols = st.columns(2)
    with cols[0]:
        fig = px.scatter(country_data, x="coalcons_ej", y="co2_combust_per_ej",
                         title="CO2 Emissions vs Fossil Fuel Consumption",
                         labels={'coalcons_ej': 'Fossil Fuel Consumption (Exajoules)',
                                 'co2_combust_per_ej': 'CO2 Emissions (Tonnes per Exajoule)'},
                         color_discrete_sequence=["#FF6692"])
        fig.update_layout(
            title=dict(font=dict(size=20)),
            xaxis=dict(title="Fossil Fuel Consumption"),
            yaxis=dict(title="CO2 Emissions"),
        )
        st.plotly_chart(fig, use_container_width=True)

with cols[1]:
    st.subheader(f"🌍 Energy Mix for {country} (Most Recent Year)")

    # Filter data for the selected country and the most recent year
    most_recent_year = country_data["Year"].dt.year.max()
    country_energy_mix = country_data[country_data["Year"].dt.year == most_recent_year][['oilcons_ej', 'gascons_ej', 'coalcons_ej', 'ren_power_ej']].sum()

    # Check if energy mix data is available
    if country_energy_mix.sum() == 0:
        st.warning(f"No energy mix data available for {country} in {most_recent_year}.")
    else:
        # Convert to a DataFrame for plotting
        energy_mix_df = country_energy_mix.reset_index().rename(columns={0: 'Total Consumption', 'index': 'Energy Source'})

        # Create a pie chart for energy mix
        fig = px.pie(
            energy_mix_df, 
            values='Total Consumption', 
            names='Energy Source', 
            title=f"Energy Mix for {country} in {most_recent_year}", 
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        fig.update_traces(textinfo='percent+label', pull=[0.1, 0, 0, 0])  # Highlight the first slice
        fig.update_layout(
            title=dict(font=dict(size=20))
        )
        st.plotly_chart(fig, use_container_width=True)

    

# Tab 3: Correlation Analysis
with tab3:
    st.subheader("📈 Correlation Analysis: Population vs Energy Use")

    # Correlation Heatmap
    cols_to_analyze = ['pop', 'primary_ej', 'solar_ej', 'hydro_ej', 'nuclear_ej', 'ren_power_ej']
    correlation_data = country_data[cols_to_analyze].dropna()

    if correlation_data.empty:
        st.warning("Not enough data available to generate a correlation matrix.")
    else:
        corr = correlation_data.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, cbar=True, square=True, fmt=".2f",
                    annot_kws={"size": 10})
        ax.set_title("Correlation Matrix", fontsize=16)
        st.pyplot(fig)

    # Add insights about correlation
    st.markdown("""
        ### Insights:
        - Positive correlations indicate that variables tend to increase together.
        - Negative correlations indicate that one variable tends to decrease as the other increases.
        - Values close to **1** or **-1** indicate strong correlations, while values close to **0** indicate weak or no correlation.
    """)


# Tab 4: Country-Specific Machine Learning Models
with tab4:
    st.subheader(f"🔍 Machine Learning Models for {country}")

    # Section 1: Random Forest Regression
    st.markdown("### 🌍 Predicting Energy Consumption (Random Forest Regression)")

    # Feature and target variables
    X = country_data[['pop', 'ren_power_ej', 'co2_combust_per_ej']].dropna()
    y = country_data['primary_ej'].dropna()

    if not X.empty and not y.empty:
        # Align indices
        X = X.loc[y.index]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = rf_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = rf_model.score(X_test, y_test)
        mae = np.mean(np.abs(y_test - y_pred))
        st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
        st.write(f"**R-squared:** {r2:.2f}")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")

        # Visualization: Actual vs Predicted
        results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        fig = px.scatter(results_df, x='Actual', y='Predicted',
                         title=f"Actual vs Predicted Energy Consumption in {country}",
                         labels={'Actual': 'Actual Energy Consumption (Exajoules)', 'Predicted': 'Predicted Energy Consumption (Exajoules)'},
                         trendline="ols")
        st.plotly_chart(fig)

        # Residual Plot
        results_df['Residual'] = results_df['Actual'] - results_df['Predicted']
        fig = px.scatter(results_df, x='Actual', y='Residual',
                         title=f"Residuals (Actual - Predicted) for {country}",
                         labels={'Actual': 'Actual Energy Consumption (Exajoules)', 'Residual': 'Residual (Error)'},
                         trendline="ols")
        st.plotly_chart(fig)

        # Feature Importance
        importances = rf_model.feature_importances_
        feature_names = X.columns
        importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False)

        fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h",
                     title="Feature Importance (Random Forest)")
        st.plotly_chart(fig)
    else:
        st.warning(f"Not enough data to train the Random Forest model for {country}.")

    # Section 2: Logistic Regression Classification
    st.markdown(f"### 🌱 Classifying High vs Low Renewable Energy Producers in {country}")

    # High vs Low renewable energy production
    country_data['high_renewable'] = np.where(country_data['ren_power_ej'] > country_data['ren_power_ej'].median(), 1, 0)

    # Features and target for classification
    X_class = country_data[['pop', 'primary_ej', 'co2_combust_per_ej']].dropna()
    y_class = country_data['high_renewable']

    if not X_class.empty and not y_class.empty:
        # Display class distribution
        st.write("Class Distribution:")
        st.write(y_class.value_counts())

        # Align indices
        X_class = X_class.loc[y_class.index]

        # Train/test split
        X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

        # Logistic Regression model
        logreg_model = LogisticRegression()
        logreg_model.fit(X_train_class, y_train_class)

        # Predict and evaluate
        y_pred_class = logreg_model.predict(X_test_class)
        y_prob_class = logreg_model.predict_proba(X_test_class)[:, 1]
        accuracy = accuracy_score(y_test_class, y_pred_class)
        st.write(f"**Accuracy of Renewable Energy Classification in {country}:** {accuracy:.2f}")

        # Confusion Matrix
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_test_class, y_pred_class)
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # ROC Curve
        st.markdown("#### ROC Curve")
        fpr, tpr, thresholds = roc_curve(y_test_class, y_prob_class)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc="lower right")
        st.pyplot(fig)

        # Precision-Recall Curve
        st.markdown("#### Precision-Recall Curve")
        precision, recall, _ = precision_recall_curve(y_test_class, y_prob_class)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(recall, precision, color='blue', lw=2)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        st.pyplot(fig)
    else:
        st.warning(f"Not enough data to train the Logistic Regression model for {country}.")
    
    # Tab 5: Global Data
with tab5:
    st.markdown("## 🌍 Global Energy and Emissions Analysis")

    # Global Energy Mix
    st.subheader("🔋 Global Energy Mix (Most Recent Year)")
    most_recent_year = data_cleaned["Year"].dt.year.max()
    global_energy_mix = data_cleaned[data_cleaned["Year"].dt.year == most_recent_year][
        ['oilcons_ej', 'gascons_ej', 'coalcons_ej', 'ren_power_ej']
    ].sum().reset_index()
    global_energy_mix.columns = ['Energy Source', 'Consumption (Exajoules)']
    fig = px.pie(global_energy_mix, values='Consumption (Exajoules)', names='Energy Source',
                 title=f"Global Energy Mix ({most_recent_year})", color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig, use_container_width=True)

    # Top Energy Consumers
    st.subheader("🌟 Top 10 Energy Consumers (Most Recent Year)")
    try:
        top_energy_consumers = data_cleaned[data_cleaned["Year"].dt.year == most_recent_year].groupby("Country")['primary_ej'].sum().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(top_energy_consumers, x='Country', y='primary_ej',
                     title=f"Top 10 Energy Consumers ({most_recent_year})",
                     labels={'primary_ej': 'Energy Consumption (Exajoules)', 'Country': 'Country'},
                     color='primary_ej', color_continuous_scale=px.colors.sequential.Plasma)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error in calculating top energy consumers: {str(e)}")

    # Global CO2 Emissions Trends
    st.subheader("🌡️ Global CO2 Emissions Over Time")
    try:
        global_co2_trends = data_cleaned.groupby("Year")['co2_combust_per_ej'].sum().reset_index()
        fig = px.line(global_co2_trends, x='Year', y='co2_combust_per_ej',
                      title="Global CO2 Emissions Over Time",
                      labels={'co2_combust_per_ej': 'CO2 Emissions (Tonnes per Exajoule)', 'Year': 'Year'},
                      color_discrete_sequence=["#FF6347"])
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error in calculating global CO2 emissions trends: {str(e)}")

    # Global Renewable Energy Share Map
    st.subheader("🗺️ Global Renewable Energy Share (%)")
    try:
        global_renewable_share = data_cleaned.groupby("Country")[['ren_power_ej', 'primary_ej']].sum().reset_index()
        global_renewable_share['renewable_share'] = (global_renewable_share['ren_power_ej'] / global_renewable_share['primary_ej']) * 100
        fig = px.choropleth(global_renewable_share, locations="Country", locationmode="country names",
                            color="renewable_share", title="Global Renewable Energy Share (%)",
                            labels={'renewable_share': 'Renewable Energy Share (%)'}, color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error in calculating global renewable energy share: {str(e)}")

    # Key Global Metrics
    st.subheader("📊 Global Key Metrics")
    try:
        global_key_metrics = {
            "Total Renewable Energy Production (Exajoules)": data_cleaned['ren_power_ej'].sum(),
            "Total Fossil Fuel Consumption (Exajoules)": data_cleaned[['oilcons_ej', 'gascons_ej', 'coalcons_ej']].sum().sum(),
            "Total Global Population": data_cleaned['pop'].max(),
            "Total CO2 Emissions (Tonnes)": data_cleaned['co2_combust_per_ej'].sum()
        }
        global_metrics_df = pd.DataFrame(list(global_key_metrics.items()), columns=["Metric", "Value"])
        st.table(global_metrics_df)
    except Exception as e:
        st.error(f"Error in calculating global key metrics: {str(e)}")

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

# Top 10 Clean Energy Producers
with col1:
    st.subheader("🌱 Top 10 Clean Energy Producers")
    if "renewable_share" not in global_renewable_share.columns:
        st.error("'renewable_share' column is missing in the dataset.")
    else:
        clean_energy_producers = global_renewable_share.sort_values(by="renewable_share", ascending=False).head(10)
        st.dataframe(clean_energy_producers[["Country", "renewable_share"]])

# Top 10 Population Controllers
with col2:
    st.subheader("👥 Top 10 Population Controllers")
    try:
        population_growth_data = data_cleaned.sort_values(by=['Country', 'Year'])
        population_growth_data['pop_growth'] = population_growth_data.groupby("Country")['pop'].pct_change()
        country_population_growth = (
            population_growth_data.groupby("Country")['pop_growth']
            .mean()
            .sort_values(ascending=True)
            .head(10)
            .reset_index()
        )
        country_population_growth.columns = ["Country", "Population Growth Rate (%)"]
        st.dataframe(country_population_growth)
    except Exception as e:
        st.error(f"Error in calculating population growth rates: {str(e)}")

# Worst CO2 Emitters
with col3:
    st.subheader("🔥 Top 10 Worst CO2 Emitters")
    try:
        worst_co2_emitters = (
            data_cleaned.groupby("Country")['co2_combust_per_ej']
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        worst_co2_emitters.columns = ["Country", "Total CO2 Emissions (Tonnes)"]
        st.dataframe(worst_co2_emitters)
    except Exception as e:
        st.error(f"Error in calculating CO2 emissions: {str(e)}")

# Most Energy Efficient Countries
with col4:
    st.subheader("💡 Most Energy Efficient Countries")
    try:
        energy_efficiency = data_cleaned.groupby("Country")[['primary_ej', 'pop']].sum().reset_index()
        energy_efficiency['energy_per_capita'] = energy_efficiency['primary_ej'] / energy_efficiency['pop']
        most_efficient_countries = energy_efficiency.sort_values(by="energy_per_capita", ascending=True).head(10)
        st.dataframe(most_efficient_countries[["Country", "energy_per_capita"]])
    except Exception as e:
        st.error(f"Error in calculating energy efficiency: {str(e)}")