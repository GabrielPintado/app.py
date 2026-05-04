import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import zscore

# 1. Configuración de Estilos (CSS) para marcos y contenedores
st.set_page_config(page_title="DataMaster Pro", layout="wide")
st.markdown("""
    <style>
    .plot-container {
        border: 2px solid #e6e9ef;
        border-radius: 10px;
        padding: 20px;
        background-color: white;
        margin-bottom: 20px;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- NAVEGACIÓN PERSISTENTE ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'page' not in st.session_state:
    st.session_state.page = "Etapa 0: Carga"

def nav_to(page):
    st.session_state.page = page

st.sidebar.title("🎮 Panel de Control")
page = st.sidebar.radio("Ir a la etapa:", 
                        ["Etapa 0: Carga", "Etapa 1: EDA Profundo", "Etapa 2: Limpieza", "Etapa 4: Visualización"], 
                        index=["Etapa 0: Carga", "Etapa 1: EDA Profundo", "Etapa 2: Limpieza", "Etapa 4: Visualización"].index(st.session_state.page))
st.session_state.page = page

# --- ETAPA 0: CARGA ---
if st.session_state.page == "Etapa 0: Carga":
    st.header("📥 Etapa 0: Ingesta de Datos")
    archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])
    if archivo:
        st.session_state.df = pd.read_csv(archivo)
        st.success("✅ ¡Datos cargados exitosamente!")
        with st.container():
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.subheader("Vista Previa del Dataset")
            st.dataframe(st.session_state.df.head(10), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# --- BLOQUEO DE SEGURIDAD ---
if st.session_state.df is None and st.session_state.page != "Etapa 0: Carga":
    st.warning("⚠️ Primero carga un archivo en la Etapa 0.")
    st.stop()

# --- ETAPA 1: EDA ---
elif st.session_state.page == "Etapa 1: EDA Profundo":
    st.header("📊 Etapa 1: Análisis Exploratorio")
    df = st.session_state.df
    
    col_stats, col_info = st.columns([2, 1])
    with col_stats:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("Estadísticas Descriptivas")
        st.dataframe(df.describe().T, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_info:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("Información de Columnas")
        st.write(f"**Total de registros:** {df.shape[0]}")
        st.write(f"**Total de variables:** {df.shape[1]}")
        st.write(df.dtypes)
        st.markdown('</div>', unsafe_allow_html=True)

# --- ETAPA 2: LIMPIEZA ---
elif st.session_state.page == "Etapa 2: Limpieza":
    st.header("🧹 Etapa 2: Limpieza y Estandarización")
    df = st.session_state.df.copy()
    
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.subheader("Estandarización Z-Score")
    cols_num = df.select_dtypes(include=np.number).columns
    cols_to_scale = st.multiselect("Columnas para normalizar (Media 0, Desv. Est. 1):", cols_num)
    if st.button("Ejecutar Estandarización"):
        if cols_to_scale:
            for col in cols_to_scale:
                df[f"{col}_std"] = zscore(df[col], ddof=1)
            st.session_state.df = df
            st.success("Columnas estandarizadas añadidas.")
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- ETAPA 4: VISUALIZACIÓN ---
elif st.session_state.page == "Etapa 4: Visualización":
    st.header("🎨 Etapa 4: Laboratorio de Gráficos")
    df = st.session_state.df
    cols_num = df.select_dtypes(include=np.number).columns.tolist()
    cols_cat = df.select_dtypes(exclude=np.number).columns.tolist()

    tipo_grafico = st.selectbox("Seleccione Visualización:", 
                                ["Violín e Histograma", "Análisis de Boxplot", "Correlación de Pearson", "Barras con Error"])

    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    
    if tipo_grafico == "Violín e Histograma":
        sel_col = st.selectbox("Variable numérica:", cols_num)
        # Métricas de referencia
        m1, m2, m3 = st.columns(3)
        m1.metric("Mínimo", f"{df[sel_col].min():.2f}")
        m2.metric("Promedio", f"{df[sel_col].mean():.2f}")
        m3.metric("Máximo", f"{df[sel_col].max():.2f}")
        
        fig = px.violin(df, y=sel_col, box=True, points="all", title=f"Distribución Detallada: {sel_col}")
        fig.update_layout(showlegend=False, margin=dict(l=40, r=40, t=60, b=40), paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    elif tipo_grafico == "Análisis de Boxplot":
        sel_col = st.selectbox("Variable numérica:", cols_num)
        cat_col = st.selectbox("Comparar por (Opcional):", [None] + cols_cat)
        
        fig = px.box(df, y=sel_col, x=cat_col, notched=True, points="outliers",
                     title=f"Límites y Outliers: {sel_col} por {cat_col if cat_col else 'Total'}")
        fig.update_layout(xaxis_title=cat_col if cat_col else "General", yaxis_title=sel_col)
        fig.update_traces(marker_color='#007bff')
        st.plotly_chart(fig, use_container_width=True)

    elif tipo_grafico == "Correlación de Pearson":
        st.subheader("Matriz de Relaciones Lineales")
        corr = df[cols_num].corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", 
                        color_continuous_scale='RdBu_r', range_color=[-1, 1],
                        title="Marcos de Correlación (Límite -1 a 1)")
        st.plotly_chart(fig, use_container_width=True)

    elif tipo_grafico == "Barras con Error":
        x_col = st.selectbox("Categoría:", cols_cat)
        y_col = st.selectbox("Medida Numérica:", cols_num)
        
        df_stats = df.groupby(x_col)[y_col].agg(['mean', 'std']).reset_index()
        fig = px.bar(df_stats, x=x_col, y='mean', error_y='std',
                     title=f"Promedio de {y_col} con Margen de Error (Std Dev)",
                     labels={'mean': 'Promedio', x_col: x_col.capitalize()})
        fig.update_layout(yaxis_gridcolor='lightgray')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Botón de Descarga
    st.sidebar.markdown("---")
    csv = df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("💾 Descargar Dataset Procesado", data=csv, file_name="data_procesada.csv", mime='text/csv')
