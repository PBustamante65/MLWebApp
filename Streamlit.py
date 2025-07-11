import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="MLWebApp", layout="wide")

pipeline_path = "artifacts/preprocessor/preprocessor.pkl"
model_path = "artifacts/model/svm.pkl"
encoder_path = "artifacts/preprocessor/labelencoder.pkl"

with open(pipeline_path, "rb") as file1:
    print(file1.read(100))

try:
    pipeline = joblib.load(pipeline_path)
    print("Pipeline cargada")
    # st.write("Pipeline cargada")
except Exception as e:
    print(f"Error cargando el pipeline {e}")

with open(model_path, "rb") as file2:
    print(file2.read(100))

try:
    model = joblib.load(model_path)
    print("Modelo cargado")
    # st.write("Modelo cargado")
except Exception as e:
    print(f"Error cargando el modelo {e}")

with open(encoder_path, "rb") as file3:
    print(file3.read(100))

try:
    encoder = joblib.load(encoder_path)
    print("Encoder cargado")
    # st.write("Encoder cargado")
except Exception as e:
    print(f"Error cargando el encoder {e}")


#################################################

st.title("WebApp de Machine Learning")
st.header("Ingreso de los datos")

# #   Column         Non-Null Count  Dtype
# ---  ------         --------------  -----
#  0   battery_power  2000 non-null   int64
#  1   blue           2000 non-null   int64
#  2   clock_speed    2000 non-null   float64
#  3   dual_sim       2000 non-null   int64
#  4   fc             2000 non-null   int64
#  5   four_g         2000 non-null   int64
#  6   int_memory     2000 non-null   int64
#  7   m_dep          2000 non-null   float64
#  8   mobile_wt      2000 non-null   int64
#  9   n_cores        2000 non-null   int64
#  10  pc             2000 non-null   int64
#  11  px_height      2000 non-null   int64
#  12  px_width       2000 non-null   int64
#  13  ram            2000 non-null   int64
#  14  sc_h           2000 non-null   int64
#  15  sc_w           2000 non-null   int64
#  16  talk_time      2000 non-null   int64
#  17  three_g        2000 non-null   int64
#  18  touch_screen   2000 non-null   int64
#  19  wifi           2000 non-null   int64
#  20  price_range    2000 non-null   float64
# dtypes: float64(3), int64(18)
# (base) patrickbustamante@Patricks-MacBook-Air MLTest % python3 ModelTraining.py
#        battery_power       blue  clock_speed     dual_sim           fc       four_g   int_memory        m_dep  ...          ram         sc_h         sc_w    talk_time      three_g  touch_screen         wifi  price_ra
# count    2000.000000  2000.0000  2000.000000  2000.000000  2000.000000  2000.000000  2000.000000  2000.000000  ...  2000.000000  2000.000000  2000.000000  2000.000000  2000.000000   2000.000000  2000.000000  2000.000
# mean     1238.518500     0.4950     1.522250     0.509500     4.309500     0.521500    32.046500     0.501750  ...  2124.213000    12.306500     5.767000    11.011000     0.761500      0.503000     0.507000     1.500
# std       439.418206     0.5001     0.816004     0.500035     4.341444     0.499662    18.145715     0.288416  ...  1084.732044     4.213245     4.356398     5.463955     0.426273      0.500116     0.500076     1.118
# min       501.000000     0.0000     0.500000     0.000000     0.000000     0.000000     2.000000     0.100000  ...   256.000000     5.000000     0.000000     2.000000     0.000000      0.000000     0.000000     0.000
# 25%       851.750000     0.0000     0.700000     0.000000     1.000000     0.000000    16.000000     0.200000  ...  1207.500000     9.000000     2.000000     6.000000     1.000000      0.000000     0.000000     0.750
# 50%      1226.000000     0.0000     1.500000     1.000000     3.000000     1.000000    32.000000     0.500000  ...  2146.500000    12.000000     5.000000    11.000000     1.000000      1.000000     1.000000     1.500
# 75%      1615.250000     1.0000     2.200000     1.000000     7.000000     1.000000    48.000000     0.800000  ...  3064.500000    16.000000     9.000000    16.000000     1.000000      1.000000     1.000000     2.250
# max      1998.000000     1.0000     3.000000     1.000000    19.000000     1.000000    64.000000     1.000000  ...  3998.000000    19.000000    18.000000


col1, col2, col3 = st.columns(3)

with col1:

    battery_power = st.slider(
        "Poder de la bateria (mAh)", min_value=500, max_value=2000, value=800
    )
    clock_speed = st.slider("Velocidad del CPU (Mhz)", min_value=0.5, max_value=3.0)
    fc = st.slider("Camara frontal (MP)", min_value=0, max_value=19, step=1)
    int_memory = st.slider("Memoria interna (GB)", min_value=2, max_value=64, value=32)
    px_height = st.slider(
        "Resolucion de la pantalla (Altura en px)", min_value=100, max_value=2000
    )

with col2:

    m_dep = st.slider("Grosor del telefono (cm)", min_value=0.1, max_value=1.0)
    mobile_wt = st.slider("Peso del dispositivo (g)", min_value=100, max_value=2000)
    n_cores = st.slider("Nucleos del procesador", min_value=1, max_value=10)
    pc = st.slider("Camara trasera (MP)", min_value=0, max_value=19, step=1)
    px_width = st.slider(
        "Resolucion de la pantalla (Ancho en px)", min_value=100, max_value=2000
    )

with col3:

    ram = st.slider("Memoria Ram (MB)", min_value=256, max_value=4000)
    sc_h = st.slider("Altura de la pantalla (cm)", min_value=5, max_value=19)
    sc_w = st.slider("Ancho de la pantalla (cm)", min_value=0, max_value=18)
    talk_time = st.slider(
        "Duracion de la bateria bajo uso constante (Hrs)", min_value=2, max_value=20
    )


st.divider()
col4, col5, col6 = st.columns(3)

with col4:

    blue = st.selectbox("Tiene Bluetooth?", [0, 1])
    three_g = st.selectbox("Tiene 3G?", [0, 1])

with col5:

    dual_sim = st.selectbox("Tiene Dual Sim?", [0, 1])
    touch_screen = st.selectbox("Tiene pantalla tactil?", [0, 1])

with col6:

    four_g = st.selectbox("Tiene 4G?", [0, 1])
    wifi = st.selectbox("Tiene WiFi?", [0, 1])

st.divider()


if st.button("Predict"):

    input_data = pd.DataFrame(
        {
            "battery_power": [battery_power],
            "blue": [blue],
            "clock_speed": [clock_speed],
            "dual_sim": [dual_sim],
            "fc": [fc],
            "four_g": [four_g],
            "int_memory": [int_memory],
            "m_dep": [m_dep],
            "mobile_wt": [mobile_wt],
            "n_cores": [n_cores],
            "pc": [pc],
            "px_height": [px_height],
            "px_width": [px_width],
            "ram": [ram],
            "sc_h": [sc_h],
            "sc_w": [sc_w],
            "talk_time": [talk_time],
            "three_g": [three_g],
            "touch_screen": [touch_screen],
            "wifi": [wifi],
        }
    )

    st.dataframe(input_data)

    pipelined_data = pipeline.transform(input_data)

    prediction = model.predict(pipelined_data)

    # st.write(prediction)
    if prediction[0] == 0:
        st.success("El precio del dispositivo es bajo")
    elif prediction[0] == 1:
        st.success("El precio del dispositivo es medio")
    elif prediction[0] == 2:
        st.success("El precio del dispositivo es alto")
    elif prediction[0] == 3:
        st.success("El precio del dispositivo es muy alto")
