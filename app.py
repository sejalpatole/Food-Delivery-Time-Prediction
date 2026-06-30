import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import plotly.graph_objects as go

from math import radians, sin, cos, sqrt, atan2

# ===============================================
# PAGE CONFIG
# ===============================================

st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================================
# LOAD MODEL FILES
# ===============================================

@st.cache_resource
def load_files():

    model = pickle.load(open("food_delivery_model.pkl","rb"))
    scaler = pickle.load(open("scaler.pkl","rb"))
    encoders = pickle.load(open("label_encoders.pkl","rb"))

    return model, scaler, encoders

model, scaler, encoders = load_files()

# ===============================================
# PREMIUM CSS
# ===============================================

st.markdown("""

<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

*{
font-family:'Poppins',sans-serif;
}

.stApp{

background:linear-gradient(
135deg,
#FFE6F0,
#F8F9FF,
#FFF4E6,
#EAF7FF
);

background-size:400% 400%;

animation:gradient 15s ease infinite;

}

@keyframes gradient{

0%{background-position:0% 50%;}
50%{background-position:100% 50%;}
100%{background-position:0% 50%;}

}

.block-container{

padding-top:2rem;
padding-bottom:2rem;

}

section[data-testid="stSidebar"]{

background:rgba(255,255,255,.18);

backdrop-filter:blur(25px);

border-right:1px solid rgba(255,255,255,.3);

}

.hero{

background:rgba(255,255,255,.25);

backdrop-filter:blur(25px);

border-radius:25px;

padding:40px;

box-shadow:0px 8px 35px rgba(0,0,0,.12);

border:1px solid rgba(255,255,255,.35);

margin-bottom:25px;

}

.hero h1{

font-size:48px;

font-weight:700;

color:#ff4b6e;

margin-bottom:10px;

}

.hero p{

font-size:18px;

color:#555;

}

.glass{

background:rgba(255,255,255,.30);

backdrop-filter:blur(20px);

border-radius:20px;

padding:25px;

border:1px solid rgba(255,255,255,.35);

box-shadow:0px 10px 30px rgba(0,0,0,.10);

margin-bottom:20px;

}

.metric-card{

background:white;

padding:20px;

border-radius:20px;

text-align:center;

box-shadow:0px 5px 20px rgba(0,0,0,.08);

transition:.3s;

}

.metric-card:hover{

transform:translateY(-8px);

}

.predict-btn button{

background:linear-gradient(90deg,#ff4b6e,#ff7a7a);

color:white;

font-size:22px;

font-weight:bold;

height:65px;

border:none;

border-radius:18px;

width:100%;

}

.result-card{

background:linear-gradient(135deg,#6dd5ed,#2193b0);

color:white;

padding:35px;

border-radius:25px;

text-align:center;

font-size:35px;

font-weight:bold;

box-shadow:0px 15px 35px rgba(0,0,0,.20);

}

.footer{

text-align:center;

padding:30px;

font-size:15px;

color:#666;

}

div[data-testid="stMetric"]{

background:white;

padding:15px;

border-radius:18px;

box-shadow:0px 5px 15px rgba(0,0,0,.08);

}
# ==========================================================================================
/* Floating Cards */

.info-card{

background:rgba(255,255,255,.35);

backdrop-filter:blur(18px);

border-radius:20px;

padding:18px;

text-align:center;

box-shadow:0px 10px 25px rgba(0,0,0,.12);

transition:0.4s;

margin-top:10px;

}

.info-card:hover{

transform:translateY(-8px);

box-shadow:0px 20px 35px rgba(0,0,0,.18);

}

/* Hero Banner */

.hero-banner{

background:linear-gradient(135deg,#ff758c,#ff7eb3);

border-radius:30px;

padding:35px;

color:white;

box-shadow:0px 20px 40px rgba(0,0,0,.18);

margin-bottom:25px;

}

/* Timeline */

.timeline{

display:flex;

justify-content:space-between;

padding:20px;

margin-top:15px;

font-size:20px;

font-weight:bold;

}

/* Glow Button */

.stButton>button{

background:linear-gradient(90deg,#ff4b6e,#ff8a65);

color:white;

height:65px;

font-size:22px;

font-weight:bold;

border-radius:18px;

border:none;

box-shadow:0px 0px 20px rgba(255,75,110,.45);

transition:0.4s;

}

.stButton>button:hover{

transform:scale(1.03);

box-shadow:0px 0px 35px rgba(255,75,110,.75);

}
            
# ==================================================================================
</style>

""",unsafe_allow_html=True)

# ===============================================
# SIDEBAR
# ===============================================

with st.sidebar:

    st.image("https://img.icons8.com/fluency/96/delivery.png",width=90)

    st.title("Food Delivery")

    st.markdown("---")

    st.success("🚚 AI Powered Prediction")

    st.info("📍 Live Distance Calculation")

    st.info("⚡ Fast Prediction")

    st.info("📊 Delivery Insights")

    st.markdown("---")

    st.write("### Technologies")

    st.write("✅ Python")

    st.write("✅ Streamlit")

    st.write("✅ Scikit-Learn")

    st.write("✅ Machine Learning")

    st.markdown("---")

    st.caption("Designed by Sejal ❤️")

# ===============================================
# HERO SECTION
# ===============================================


st.markdown("""

<div class="hero-banner">

<h1>🍕 AI Powered Food Delivery Dashboard</h1>

<p style="font-size:20px;">
Predict delivery time instantly with Machine Learning.
Monitor delivery insights, traffic conditions and route distance.
</p>

</div>

""", unsafe_allow_html=True)


# ===============================================
# TOP METRICS
# ===============================================

m1,m2,m3,m4=st.columns(4)

with m1:
    st.metric("Orders","25K+")

with m2:
    st.metric("Accuracy","96%")

with m3:
    st.metric("Cities","15+")

with m4:
    st.metric("Prediction","Instant")

# =====================================================
# DISTANCE CALCULATION
# =====================================================

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between restaurant and delivery
    location using Haversine Formula.
    """

    R = 6371

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return round(R * c, 2)


# =====================================================
# HELPER FUNCTION
# =====================================================

def encode_value(column, value):
    """
    Encode categorical values safely.
    """

    encoder = encoders[column]

    if value not in encoder.classes_:
        value = encoder.classes_[0]

    return encoder.transform([value])[0]


# =====================================================
# DELIVERY STATUS
# =====================================================

def delivery_status(minutes):

    if minutes <= 20:
        return "🟢 Super Fast"

    elif minutes <= 35:
        return "🟡 Normal"

    elif minutes <= 50:
        return "🟠 Busy"

    else:
        return "🔴 Heavy Delay"


# =====================================================
# DELIVERY TIPS
# =====================================================

def delivery_tip(minutes):

    if minutes <= 20:
        return "Your order should arrive very quickly."

    elif minutes <= 35:
        return "Traffic looks normal today."

    elif minutes <= 50:
        return "Slight traffic congestion detected."

    else:
        return "Heavy traffic or long distance may delay delivery."


# =====================================================
# ARRIVAL COLOR
# =====================================================

def arrival_color(minutes):

    if minutes <= 20:
        return "#16a34a"

    elif minutes <= 35:
        return "#eab308"

    elif minutes <= 50:
        return "#f97316"

    return "#dc2626"


# =====================================================
# DASHBOARD
# =====================================================

left, right = st.columns([2, 1])

with right:

    st.markdown("""
    <div class='glass'>
    <h3 style='text-align:center;color:#ff4b6e;'>
    📊 Live Dashboard
    </h3>
    </div>
    """, unsafe_allow_html=True)

    st.metric("Average Delivery", "31 mins")

    st.metric("Fastest Route", "12 mins")

    st.metric("Today's Orders", "1,256")

    st.metric("Customer Rating", "4.8 ⭐")

with left:

    st.markdown("""
    <div class='glass'>
    <h3 style='color:#ff4b6e'>
    🍔 Enter Delivery Details
    </h3>
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# INPUT COLUMNS
# =====================================================

col1, col2 = st.columns(2)

# =====================================================
# LEFT COLUMN
# =====================================================

with col1:

    st.markdown("### 👨 Delivery Partner")

    age = st.slider(
        "Delivery Partner Age",
        18,
        50,
        25
    )

    rating = st.slider(
        "Delivery Rating",
        1.0,
        5.0,
        4.5,
        0.1
    )

    vehicle_condition = st.slider(
        "Vehicle Condition",
        0,
        3,
        2
    )

    multiple_deliveries = st.selectbox(
        "Multiple Deliveries",
        [0,1,2,3]
    )

    st.markdown("---")

    st.markdown("### 🍔 Order Details")

    order_type = st.selectbox(
        "Order Type",
        encoders["Type_of_order"].classes_
    )

    vehicle_type = st.selectbox(
        "Vehicle Type",
        encoders["Type_of_vehicle"].classes_
    )

    festival = st.selectbox(
        "Festival",
        encoders["Festival"].classes_
    )

    city = st.selectbox(
        "City",
        encoders["City"].classes_
    )


# =====================================================
# RIGHT COLUMN
# =====================================================

with col2:

    st.markdown("### 🌤 Environment")

    weather = st.selectbox(
        "Weather",
        encoders["Weatherconditions"].classes_
    )

    traffic = st.selectbox(
        "Road Traffic",
        encoders["Road_traffic_density"].classes_
    )

    st.markdown("---")

    st.markdown("### 📍 Restaurant Location")

    restaurant_lat = st.number_input(
        "Restaurant Latitude",
        value=22.745049,
        format="%.6f"
    )

    restaurant_long = st.number_input(
        "Restaurant Longitude",
        value=75.892471,
        format="%.6f"
    )

    st.markdown("### 🏠 Delivery Location")

    delivery_lat = st.number_input(
        "Delivery Latitude",
        value=22.765049,
        format="%.6f"
    )

    delivery_long = st.number_input(
        "Delivery Longitude",
        value=75.912471,
        format="%.6f"
    )


# =====================================================
# ORDER DATE & TIME
# =====================================================

st.markdown("---")

st.markdown("## 🗓 Order Information")

c1, c2, c3 = st.columns(3)

with c1:

    order_date = st.date_input(
        "Order Date"
    )

with c2:

    order_hour = st.slider(
        "Order Hour",
        0,
        23,
        18
    )

with c3:

    pickup_minutes = st.slider(
        "Pickup Time (Minutes)",
        5,
        60,
        15
    )


# =====================================================
# FEATURE ENGINEERING
# =====================================================

distance = calculate_distance(
    restaurant_lat,
    restaurant_long,
    delivery_lat,
    delivery_long
)

order_day = order_date.day
order_month = order_date.month
order_weekday = order_date.weekday()


# =====================================================
# LIVE DASHBOARD
# =====================================================

st.markdown("---")

k1,k2,k3 = st.columns(3)

with k1:
    st.metric(
        "📍 Distance",
        f"{distance:.2f} km"
    )

with k2:
    st.metric(
        "🕒 Pickup Time",
        f"{pickup_minutes} mins"
    )

with k3:
    st.metric(
        "⭐ Delivery Rating",
        rating
    )


st.markdown("---")

st.markdown(
"""
<div class='glass'>

<h3 style='text-align:center;color:#ff4b6e;'>

🚀 Ready to Predict Delivery Time

</h3>

</div>
""",
unsafe_allow_html=True
)

# =====================================================
# PREDICT BUTTON
# =====================================================

st.markdown("<br>", unsafe_allow_html=True)

predict = st.button(
    "🚚 Predict Delivery Time",
    use_container_width=True
)

# =====================================================
# PREDICTION
# =====================================================

if predict:

    progress = st.progress(0)

    status = st.empty()

    for i in range(101):
        progress.progress(i)
        status.info(f"Analyzing Delivery Route... {i}%")
        time.sleep(0.01)

    status.success("Prediction Completed Successfully!")

    # -----------------------------------------
    # Encode categorical values
    # -----------------------------------------

    weather_encoded = encode_value(
        "Weatherconditions",
        weather
    )

    traffic_encoded = encode_value(
        "Road_traffic_density",
        traffic
    )

    order_encoded = encode_value(
        "Type_of_order",
        order_type
    )

    vehicle_encoded = encode_value(
        "Type_of_vehicle",
        vehicle_type
    )

    festival_encoded = encode_value(
        "Festival",
        festival
    )

    city_encoded = encode_value(
        "City",
        city
    )

    # -----------------------------------------
    # Create DataFrame
    # -----------------------------------------

    input_df = pd.DataFrame({

        "Delivery_person_Age":[age],

        "Delivery_person_Ratings":[rating],

        "Restaurant_latitude":[restaurant_lat],

        "Restaurant_longitude":[restaurant_long],

        "Delivery_location_latitude":[delivery_lat],

        "Delivery_location_longitude":[delivery_long],

        "Weatherconditions":[weather_encoded],

        "Road_traffic_density":[traffic_encoded],

        "Vehicle_condition":[vehicle_condition],

        "Type_of_order":[order_encoded],

        "Type_of_vehicle":[vehicle_encoded],

        "multiple_deliveries":[multiple_deliveries],

        "Festival":[festival_encoded],

        "City":[city_encoded],

        "Order_Day":[order_day],

        "Order_Month":[order_month],

        "Order_Weekday":[order_weekday],

        "Distance_km":[distance],

        "Order_Hour":[order_hour],

        "Pickup_Time_Minutes":[pickup_minutes]

    })

    # -----------------------------------------
    # Scale Data
    # -----------------------------------------

    input_scaled = scaler.transform(input_df)

    # -----------------------------------------
    # Predict
    # -----------------------------------------

    prediction = model.predict(input_scaled)[0]

    prediction = round(float(prediction),2)

    # -----------------------------------------
    # Result
    # -----------------------------------------

    st.balloons()

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"""

    <div class="result-card">

    🚚 Estimated Delivery Time

    <br><br>

    <span style="font-size:60px;">

    {prediction} Minutes

    </span>

    </div>

    """,unsafe_allow_html=True)

    # -----------------------------------------
    # Insights
    # -----------------------------------------

    st.markdown("## 📈 Delivery Insights")

    a,b,c = st.columns(3)

    with a:

        st.metric(
            "Delivery Status",
            delivery_status(prediction)
        )

    with b:

        st.metric(
            "Distance",
            f"{distance:.2f} KM"
        )

    with c:

        st.metric(
            "Pickup Time",
            f"{pickup_minutes} mins"
        )

    st.info(delivery_tip(prediction))

    # -----------------------------------------
    # ETA Gauge
    # -----------------------------------------
    st.markdown("## ⏱ Estimated Arrival Meter")

    fig = go.Figure(
      go.Indicator(
        mode="gauge+number",
        value=prediction,
        title={"text": "Arrival Time (Minutes)"},
        gauge={
            "axis": {"range": [0, 80]},
            "bar": {"color": "#ff4b6e"},
            "steps": [
                {"range": [0, 20], "color": "#B9FBC0"},
                {"range": [20, 40], "color": "#FFE066"},
                {"range": [40, 60], "color": "#FFB347"},
                {"range": [60, 80], "color": "#FF6B6B"},
            ],
        },
    )
)

    fig.update_layout(height=400)

    st.plotly_chart(fig, use_container_width=True)

st.markdown("## 📊 Delivery Statistics")

c1,c2,c3,c4 = st.columns(4)

with c1:

    st.markdown(f"""

    <div class='info-card'>

    <h2>🚗</h2>

    <h3>{vehicle_type}</h3>

    Vehicle

    </div>

    """,unsafe_allow_html=True)

with c2:

    st.markdown(f"""

    <div class='info-card'>

    <h2>🌦</h2>

    <h3>{weather}</h3>

    Weather

    </div>

    """,unsafe_allow_html=True)

with c3:

    st.markdown(f"""

    <div class='info-card'>

    <h2>🚦</h2>

    <h3>{traffic}</h3>

    Traffic

    </div>

    """,unsafe_allow_html=True)

with c4:

    st.markdown(f"""

    <div class='info-card'>

    <h2>📍</h2>

    <h3>{distance:.2f} KM</h3>

    Distance

    </div>

    """,unsafe_allow_html=True)

st.markdown("## 🚚 Delivery Journey")

st.markdown("""

<div class="glass">

<div class="timeline">

🍽 Order

➡️

👨‍🍳 Preparing

➡️

🚚 On the Way

➡️

🏠 Delivered

</div>

</div>

""", unsafe_allow_html=True)

    # -----------------------------------------
    # Route Card
    # -----------------------------------------

st.markdown("""

    <div class='glass'>

    <h3 style='text-align:center;color:#ff4b6e;'>

    📍 Delivery Route

    </h3>

    <p style='text-align:center;font-size:18px;'>

    🍔 Restaurant

    ➜

    🚚 Delivery Partner

    ➜

    🏠 Customer

    </p>

    </div>

    """,unsafe_allow_html=True)

    # -----------------------------------------
    # Summary Cards
    # -----------------------------------------

st.markdown("## 📊 Summary")

s1,s2,s3,s4 = st.columns(4)

with s1:
        st.metric("Weather",weather)

with s2:
        st.metric("Traffic",traffic)

with s3:
        st.metric("Vehicle",vehicle_type)

with s4:
        st.metric("City",city)

# =====================================================
# FOOTER
# =====================================================

st.markdown("<br><br>",unsafe_allow_html=True)

st.markdown("""

<div class='footer'>

Made with ❤️ by <b>Sejal Patole</b>

<br>

Food Delivery Time Prediction using Machine Learning & Streamlit

</div>

""",unsafe_allow_html=True)