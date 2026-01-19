import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Cars EDA Project", layout="wide")


@st.cache_data
def load_raw():
    return pd.read_csv("carspandas.csv")

@st.cache_data
def load_cleaned():
    return pd.read_csv("Cars_cleaned.csv")

raw = load_raw()
clean = load_cleaned()

page = st.sidebar.radio("Navigation",
["Introduction","Analysis","Conclusions"])


if page == "Introduction":

    st.title("Cars Analytics Dashboard")
    st.markdown("""Why this project ? 
                
                Used automobiles are in high demand in the Indian market right now. 
                The preowned automobile market has grown over the years and is 
                currently larger than the new car market, notwithstanding the
                recent slowdown in sales of new cars. Approximately 
                4 million used cars were purchased and sold in 2018–19, 
                compared to 3.6 million new cars. 
                Sales of new cars are slowing down, which may indicate that 
                demand is moving toward the used car market. 
                In fact, rather than purchasing new vehicles, several automobile 
                sellers swap out their old vehicles for used ones.""")



    c1,c2,c3,c4 = st.columns(4)

    c1.metric("Total Cars", len(clean))
    c2.metric("Average Price", round(clean["Price"].mean(),2))
    c3.metric("Average KM", int(clean["Kilometers_Driven"].mean()))
    c4.metric("Total Companies", clean["Company_Name"].nunique())


    st.subheader("Raw Dataset")
    st.dataframe(raw, use_container_width=True)

    st.subheader("Cleaned Dataset")
    st.dataframe(clean, use_container_width=True)


    st.subheader("Location Map")

    if "Latitude" in clean.columns and "Longitude" in clean.columns:
        st.map(clean[["Latitude","Longitude"]])
    else:
        st.info("Latitude and Longitude not available")




elif page == "Analysis":

    st.title("Exploratory Analysis Studio")

    company = st.sidebar.multiselect(
        "Select Company",
        options=clean["Company_Name"].unique(),
        default=clean["Company_Name"].unique()
    )

    year = st.sidebar.slider(
        "Select Year Range",
        int(clean["Year"].min()),
        int(clean["Year"].max()),
        (int(clean["Year"].min()), int(clean["Year"].max()))
    )

    df = clean[(clean["Company_Name"].isin(company)) &
               (clean["Year"].between(year[0],year[1]))]


    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()


    k1,k2,k3 = st.columns(3)

    k1.metric("Selected Cars", len(df))
    k2.metric("Average Price", round(df["Price"].mean(),2))
    k3.metric("Average Power", round(df["Power_value"].mean(),2))


    st.header("Univariate Analysis")

    col = st.selectbox("Choose Column", df.columns)

    fig, ax = plt.subplots(figsize=(7,4))

    if col in cat_cols:
        sns.countplot(y=df[col], ax=ax)

    else:
        dist = st.radio("View Type",["Histogram","KDE","Boxplot"])

        if dist=="Histogram":
            sns.histplot(df[col], kde=True, ax=ax)

        elif dist=="KDE":
            sns.kdeplot(df[col], fill=True, ax=ax)

        else:
            sns.boxplot(x=df[col], ax=ax)

    st.pyplot(fig)



    st.header("Bivariate Analysis")

    c1,c2 = st.columns(2)

    x = c1.selectbox("X Axis", df.columns)
    y = c2.selectbox("Y Axis", df.columns)

    fig2,ax2 = plt.subplots(figsize=(7,4))

    if x in num_cols and y in num_cols:
        sns.scatterplot(data=df,x=x,y=y,ax=ax2)
        st.write("Correlation:", round(df[x].corr(df[y]),3))

    elif x in num_cols and y in cat_cols:
        sns.boxplot(data=df,x=y,y=x,ax=ax2)

    elif x in cat_cols and y in num_cols:
        sns.boxplot(data=df,x=x,y=y,ax=ax2)

    else:
        sns.countplot(data=df,x=x,hue=y,ax=ax2)

    st.pyplot(fig2)



    st.header("Multivariate Analysis")

    option = st.selectbox("Method",
    ["Heatmap","Pairplot","Grouped Bar"])


    if option=="Heatmap":
        fig3,ax3 = plt.subplots(figsize=(9,5))
        sns.heatmap(df[num_cols].corr(),annot=True,cmap="coolwarm",ax=ax3)
        st.pyplot(fig3)

    elif option=="Pairplot":
        pair = sns.pairplot(df[num_cols])
        st.pyplot(pair)

    else:
        if "Fuel_Type" in df.columns and "Price" in df.columns:

            fig4,ax4 = plt.subplots(figsize=(8,4))

            sns.barplot(
                data=df,
                x="Fuel_Type",
                y="Price",
                hue="Transmission" if "Transmission" in df.columns else None,
                ax=ax4
            )

            st.pyplot(fig4)

        else:
            st.warning("Required columns missing")




else:

    st.title("Automated Insights")

    st.write("Total Records:", len(clean))

    st.write("Highest Price Car:",
             clean.loc[clean["Price"].idxmax(),"Company_Name"])

    st.write("Most Common Fuel:",
             clean["Fuel_Type"].mode()[0])

    st.write("Strongest Correlation with Price:",
             clean.select_dtypes(include=np.number)
             .corr()["Price"].sort_values(ascending=False).index[1])
    st.markdown("---")
st.header("❓ Frequently Asked Questions (FAQ)")

with st.expander("📘 Click to View FAQ", expanded=False):

    question = st.selectbox(
        "Select a question",
        [
            "Does various predicting factors affect the price of the used car?",
            "What all independent variables affect the pricing of used cars?",
            "Does the name of a car have any effect on pricing?",
            "How does type of transmission affect pricing?",
            "Does location in which the car is sold affect the price?",
            "Do Kilometers_Driven and Year of manufacturing have negative correlation with price?",
            "Do Mileage, Engine, and Power affect the pricing of the car?",
            "How do number of seats and fuel type affect pricing?"
        ]
    )

    if question == "Does various predicting factors affect the price of the used car?":
        st.write("""
        **Answer:**  
        Yes. Multiple predicting factors such as year, kilometers driven, brand,
        fuel type, transmission, engine, mileage, and power collectively influence
        used car pricing.
        """)

    elif question == "What all independent variables affect the pricing of used cars?":
        st.write("""
        **Answer:**  
        Independent variables include year, kilometers driven, brand, fuel type,
        transmission, engine capacity, mileage, power, seating capacity, and location.
        """)

    elif question == "Does the name of a car have any effect on pricing?":
        st.write("""
        **Answer:**  
        Yes. Brand and model name significantly impact pricing due to trust,
        maintenance cost, and market reputation.
        """)

    elif question == "How does type of transmission affect pricing?":
        st.write("""
        **Answer:**  
        Automatic transmission cars usually command higher prices due to higher
        demand and driving convenience.
        """)

    elif question == "Does location in which the car is sold affect the price?":
        st.write("""
        **Answer:**  
        Yes. Location affects pricing due to regional demand, income levels, and
        usage patterns.
        """)

    elif question == "Do Kilometers_Driven and Year of manufacturing have negative correlation with price?":
        st.write("""
        **Answer:**  
        Kilometers driven has a negative correlation with price, while year of
        manufacture has a positive correlation.
        """)

    elif question == "Do Mileage, Engine, and Power affect the pricing of the car?":
        st.write("""
        **Answer:**  
        Yes. Engine capacity and power increase price, while mileage influences
        buyer preference and value.
        """)

    elif question == "How do number of seats and fuel type affect pricing?":
        st.write("""
        **Answer:**  
        More seats increase utility and price, while diesel cars currently command
        higher demand due to fuel efficiency.
        """)


st.markdown("""
    ### 🔍 Summary of Findings

    Based on the exploratory data analysis of the used cars dataset, the following
    key insights were observed:

    - **Car price is strongly influenced by model year** — newer cars tend to have higher prices.
    - **Kilometers driven has a negative impact on price**, indicating depreciation with usage.
    - **Fuel type and transmission** significantly affect pricing patterns.
    - Certain **brands consistently command higher resale value** in the market.

    ### 📊 Market Observations
    - The used car market in India is **growing rapidly** and surpassing new car sales.
    - Buyers prefer **lower mileage and newer models**.
    - Petrol and diesel vehicles dominate the resale market.

    ### 🎯 Business Implications
    - Dealers can optimize pricing using historical trends.
    - Buyers can identify value-for-money vehicles.
    - Sellers can time resale for maximum return.

    ### 🚀 Final Note
    This dashboard demonstrates how **data-driven insights** can support
    informed decision-making in the automotive resale market.
    """)

    st.success("✔ Analysis completed successfully.")
    st.info("📘 These insights can be enhanced further with real-time market and customer behavior data.")
    

    st.title("📌 Strategic Recommendations")

    st.markdown("""
    ### 🔍 Key Observations

    While price, year, and kilometers driven play a major role in car valuation,
    there are several **soft and qualitative factors** that significantly influence
    demand and pricing in the used car market:

    1. **Wear and Tear**: The overall condition of a car impacts the amount of work
       required to make it sale-ready, which directly affects profitability.
    2. **Accident History**: Cars involved in accidents tend to lose value due to
       safety concerns and higher refurbishment costs.
    3. **Additional Features**: Comfort and safety features such as **AC, moonroof,
       airbags**, and premium interiors positively influence car prices.
    4. **Vehicle Age**: Very old car models depreciate heavily, reducing both
       demand and resale value.

    ### 🚗 Brand & Market Trends

    5. **Popular Brands**: Brands like **Maruti, Hyundai, and Honda** dominate the
       low-budget used car segment due to affordability and reliability.
    6. **Regional Demand**: Cities such as **Mumbai and Hyderabad** show higher
       activity in the used car market. This trend should be validated using
       data from additional demographic regions.

    ### 📊 Future Data Strategy

    7. **Clustering Approach**: Segmenting data by **location and car type** can help
       in building multiple predictive models tailored to specific markets.
    8. **Transmission Preference**: **Automatic cars** command higher resale prices
       and should be prioritized to improve profit margins.
    9. **Fuel Trends**: With rising petrol prices, **diesel cars** are increasingly
       preferred, offering better resale opportunities.

    ### 💡 Business Recommendations

    10. Introducing **half-day test drive schemes** can help build customer confidence
        and improve conversion rates.
    11. Offering **annual car maintenance packages** with a small upfront fee can
        attract customers and build long-term relationships.

    ### 🚀 Final Note

    Incorporating both **quantitative metrics and qualitative factors** enables
    smarter pricing, better inventory planning, and higher customer satisfaction
    in the used car ecosystem.
    """)

    st.success("✔ actionable insights generated.")


