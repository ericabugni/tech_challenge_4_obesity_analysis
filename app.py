import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# ─────────────────────────────────────────
# Configuração da página
# ─────────────────────────────────────────
st.set_page_config(
    page_title="ObesityAI — Diagnóstico Preditivo",
    page_icon="🏥",
    layout="centered"
)

# ─────────────────────────────────────────
# Carregamento dos artefatos
# ─────────────────────────────────────────
@st.cache_resource
def carregar_artefatos():
    with open("model_artifacts/best_model.pkl", "rb") as f:
        pipeline = pickle.load(f)
    with open("model_artifacts/label_encoder.pkl", "rb") as f:
        label_enc = pickle.load(f)
    with open("model_artifacts/model_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return pipeline, label_enc, meta

try:
    pipeline, label_enc, meta = carregar_artefatos()
    artefatos_ok = True
except FileNotFoundError:
    artefatos_ok = False

# ─────────────────────────────────────────
# Mapeamentos de exibição
# ─────────────────────────────────────────
TRADUCAO_CLASSE = {
    "Insufficient_Weight" : "Peso Insuficiente",
    "Normal_Weight"        : "Peso Normal",
    "Overweight_Level_I"   : "Sobrepeso Nível I",
    "Overweight_Level_II"  : "Sobrepeso Nível II",
    "Obesity_Type_I"       : "Obesidade Tipo I",
    "Obesity_Type_II"      : "Obesidade Tipo II",
    "Obesity_Type_III"     : "Obesidade Tipo III",
}
COR_CLASSE = {
    "Insufficient_Weight" : "#1565c0",
    "Normal_Weight"        : "#2e7d32",
    "Overweight_Level_I"   : "#f9a825",
    "Overweight_Level_II"  : "#ef6c00",
    "Obesity_Type_I"       : "#c62828",
    "Obesity_Type_II"      : "#b71c1c",
    "Obesity_Type_III"     : "#4a148c",
}
ICONE_CLASSE = {
    "Insufficient_Weight" : "🔵",
    "Normal_Weight"        : "🟢",
    "Overweight_Level_I"   : "🟡",
    "Overweight_Level_II"  : "🟠",
    "Obesity_Type_I"       : "🔴",
    "Obesity_Type_II"      : "🔴",
    "Obesity_Type_III"     : "🟣",
}

# ─────────────────────────────────────────
# Funções de feature engineering
# (replicam exatamente o notebook do grupo)
# ─────────────────────────────────────────
def calcular_age_group(age: float) -> str:
    if age < 18:  return "adolescente"
    if age < 25:  return "jovem_adulto"
    if age < 35:  return "adulto"
    if age < 50:  return "meia_idade"
    return "idoso"

def calcular_healthy_score(fcvc, ch2o, faf, caec, calc_freq, favc) -> float:
    caec_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    calc_map  = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    caec_num  = caec_map.get(caec, 0)
    calc_num  = calc_map.get(calc_freq, 0)
    favc_num  = 1 if favc == "yes" else 0
    positive  = (ch2o / 3) + (fcvc / 3) + (faf / 3)
    negative  = (caec_num / 3) + (calc_num / 3) + favc_num
    score     = (positive - negative / 2) * 10 / 3
    return float(np.clip(round(score, 2), 0, 10))

def montar_entrada(inputs: dict) -> pd.DataFrame:
    """Recebe os inputs do formulário e devolve o DataFrame pronto para o pipeline."""
    age    = inputs["age"]
    fcvc   = inputs["fcvc"]
    ncp    = inputs["ncp"]
    ch2o   = inputs["ch2o"]
    faf    = inputs["faf"]
    tue    = inputs["tue"]
    caec   = inputs["caec"]
    calc_f = inputs["calc"]
    favc   = inputs["favc"]
    fam    = inputs["family_history"]

    caec_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    calc_map  = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    caec_num  = caec_map.get(caec, 0)
    calc_num  = calc_map.get(calc_f, 0)

    row = {
        # Variáveis originais (sem Height/Weight — removidos para evitar leakage)
        "Gender"         : inputs["gender"],
        "Age"            : age,
        "family_history" : fam,
        "FAVC"           : favc,
        "FCVC"           : fcvc,
        "NCP"            : ncp,
        "CAEC"           : caec,
        "SMOKE"          : inputs["smoke"],
        "CH2O"           : ch2o,
        "SCC"            : inputs["scc"],
        "FAF"            : faf,
        "TUE"            : tue,
        "CALC"           : calc_f,
        "MTRANS"         : inputs["mtrans"],
        # Features de engenharia criadas no notebook
        "CAEC_num"            : caec_num,
        "CALC_num"            : calc_num,
        "age_group"           : calcular_age_group(age),
        "healthy_score"       : calcular_healthy_score(fcvc, ch2o, faf, caec, calc_f, favc),
        "sedentary"           : int(faf == 0 and tue >= 1),
        "fam_x_favc"          : int(fam == "yes") * int(favc == "yes"),
        "meal_activity_ratio" : round(ncp / (faf + 1), 3),
        "high_alcohol"        : int(calc_num >= 2),
    }
    return pd.DataFrame([row])

# ─────────────────────────────────────────
# Cabeçalho
# ─────────────────────────────────────────
st.title("🏥 ObesityAI — Sistema Preditivo")
st.markdown(
    "Preencha os dados do paciente para obter a **previsão do nível de obesidade** "
    "com base em hábitos comportamentais e estilo de vida."
)

if not artefatos_ok:
    st.error(
        "⚠️ Artefatos do modelo não encontrados!\n\n"
        "Certifique-se de que a pasta **`model_artifacts/`** está no mesmo diretório "
        "do `app.py`, contendo:\n"
        "- `best_model.pkl`\n"
        "- `label_encoder.pkl`\n"
        "- `model_meta.json`"
    )
    st.stop()

# ─────────────────────────────────────────
# Sidebar com metadados do modelo
# ─────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Sobre o Modelo")
    st.markdown(f"**Algoritmo:** {meta.get('best_model_name', '—')}")
    st.markdown(f"**Acurácia (teste):** {float(meta.get('test_accuracy', 0)):.1%}")
    st.markdown(f"**AUC-ROC:** {float(meta.get('auc_roc', 0)):.4f}")
    st.markdown(f"**F1-macro:** {float(meta.get('f1_macro', 0)):.4f}")
    st.divider()
    st.caption(
        "⚠️ Height e Weight foram **removidos** do modelo para evitar data leakage "
        "e focar nos fatores comportamentais modificáveis."
    )
    st.divider()
    with st.expander("📊 Todos os modelos testados"):
        for nome, res in meta.get("model_results", {}).items():
            st.markdown(
                f"**{nome}**  \n"
                f"CV: {res['cv_mean']:.4f} ±{res['cv_std']:.4f}  \n"
                f"Teste: {res['test_acc']:.4f} | AUC: {res['auc_roc']:.4f}"
            )

st.divider()

# ─────────────────────────────────────────
# Formulário
# ─────────────────────────────────────────
st.subheader("👤 Dados do Paciente")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gênero", ["Male", "Female"],
        format_func=lambda x: "Masculino" if x == "Male" else "Feminino")
    age    = st.number_input("Idade (anos)", min_value=10, max_value=100, value=25)
    family_history = st.selectbox("Histórico familiar de excesso de peso?",
        ["yes", "no"], format_func=lambda x: "Sim" if x == "yes" else "Não")
    smoke = st.selectbox("Fuma?", ["no", "yes"],
        format_func=lambda x: "Não" if x == "no" else "Sim")

with col2:
    favc = st.selectbox("Come alimentos calóricos com frequência?",
        ["yes", "no"], format_func=lambda x: "Sim" if x == "yes" else "Não")
    scc  = st.selectbox("Monitora calorias ingeridas?",
        ["no", "yes"], format_func=lambda x: "Não" if x == "no" else "Sim")
    mtrans = st.selectbox("Meio de transporte habitual",
        ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"],
        format_func=lambda x: {
            "Public_Transportation": "Transporte Público", "Walking": "Caminhando",
            "Automobile": "Carro", "Motorbike": "Moto", "Bike": "Bicicleta"
        }[x])
    caec = st.selectbox("Come entre as refeições?",
        ["no", "Sometimes", "Frequently", "Always"],
        format_func=lambda x: {
            "no": "Não", "Sometimes": "Às vezes",
            "Frequently": "Frequentemente", "Always": "Sempre"
        }[x])

st.divider()
st.subheader("🥗 Hábitos Alimentares e de Saúde")

col3, col4 = st.columns(2)
with col3:
    fcvc = st.slider("Frequência de consumo de vegetais (0–3)", 0, 3, 2)
    ncp  = st.slider("Refeições principais por dia (1–5)", 1, 5, 3)
    ch2o = st.slider("Litros de água por dia (0–3)", 0, 3, 2)

with col4:
    faf  = st.slider("Dias de atividade física por semana (0–3)", 0, 3, 1)
    tue  = st.slider("Horas em dispositivos tecnológicos por dia (0–3)", 0, 3, 1)
    calc = st.selectbox("Frequência do consumo de álcool",
        ["no", "Sometimes", "Frequently", "Always"],
        format_func=lambda x: {
            "no": "Não bebo", "Sometimes": "Às vezes",
            "Frequently": "Frequentemente", "Always": "Sempre"
        }[x])

st.divider()

# Preview das features calculadas
with st.expander("🔬 Ver features calculadas automaticamente"):
    caec_map_prev = {"no":0,"Sometimes":1,"Frequently":2,"Always":3}
    calc_map_prev = {"no":0,"Sometimes":1,"Frequently":2,"Always":3}
    st.markdown(f"""
| Feature | Valor |
|---|---|
| Faixa etária (`age_group`) | `{calcular_age_group(age)}` |
| Score de hábitos saudáveis (`healthy_score`) | `{calcular_healthy_score(fcvc, ch2o, faf, caec, calc, favc):.2f}` / 10 |
| Sedentário (`sedentary`) | `{int(faf == 0 and tue >= 1)}` |
| Histórico familiar × FAVC (`fam_x_favc`) | `{int(family_history == "yes") * int(favc == "yes")}` |
| Razão refeições/atividade (`meal_activity_ratio`) | `{round(ncp / (faf + 1), 3)}` |
| Alto consumo de álcool (`high_alcohol`) | `{int(calc_map_prev.get(calc, 0) >= 2)}` |
    """)

# ─────────────────────────────────────────
# Botão de previsão
# ─────────────────────────────────────────
if st.button("🔍 Realizar Diagnóstico", type="primary", use_container_width=True):

    inputs = dict(
        gender=gender, age=age, family_history=family_history,
        favc=favc, fcvc=fcvc, ncp=ncp, caec=caec, smoke=smoke,
        ch2o=ch2o, scc=scc, faf=faf, tue=tue, calc=calc, mtrans=mtrans
    )

    X_input    = montar_entrada(inputs)
    pred_enc   = pipeline.predict(X_input)[0]
    pred_class = label_enc.inverse_transform([pred_enc])[0]
    pred_proba = pipeline.predict_proba(X_input)[0]
    confianca  = pred_proba.max() * 100

    cor   = COR_CLASSE.get(pred_class, "#607d8b")
    icone = ICONE_CLASSE.get(pred_class, "⚪")
    label = TRADUCAO_CLASSE.get(pred_class, pred_class)
    hs    = calcular_healthy_score(fcvc, ch2o, faf, caec, calc, favc)

    st.subheader("📋 Resultado do Diagnóstico")
    st.markdown(
        f"""
        <div style="
            background:{cor}18;
            border-left:6px solid {cor};
            padding:20px 24px;
            border-radius:8px;
            margin-bottom:12px;
        ">
            <h2 style="color:{cor}; margin:0;">{icone} {label}</h2>
            <p style="margin:8px 0 0 0; color:#555;">
                Confiança do modelo: <strong>{confianca:.1f}%</strong>
                &nbsp;|&nbsp; Faixa etária: <strong>{calcular_age_group(age)}</strong>
                &nbsp;|&nbsp; Score de hábitos: <strong>{hs:.1f}/10</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("📊 Probabilidade por classe"):
        classes_pt = [TRADUCAO_CLASSE.get(c, c) for c in label_enc.classes_]
        df_proba = pd.DataFrame({
            "Classe": classes_pt,
            "Probabilidade": pred_proba
        }).sort_values("Probabilidade", ascending=False)
        st.dataframe(
            df_proba.style.format({"Probabilidade": "{:.1%}"}),
            use_container_width=True, hide_index=True
        )

    with st.expander("📝 Resumo dos dados inseridos"):
        resumo = {
            "Gênero"                    : "Masculino" if gender == "Male" else "Feminino",
            "Idade"                     : f"{age} anos ({calcular_age_group(age)})",
            "Histórico familiar"        : "Sim" if family_history == "yes" else "Não",
            "Alimentos calóricos freq." : "Sim" if favc == "yes" else "Não",
            "Vegetais (0–3)"            : fcvc,
            "Refeições/dia"             : ncp,
            "Come entre refeições"      : caec,
            "Fuma"                      : "Sim" if smoke == "yes" else "Não",
            "Água/dia (0–3)"            : ch2o,
            "Monitora calorias"         : "Sim" if scc == "yes" else "Não",
            "Atividade física (0–3)"    : faf,
            "Tempo em telas (0–3)"      : tue,
            "Álcool"                    : calc,
            "Transporte"                : mtrans,
        }
        st.table(pd.DataFrame(resumo.items(), columns=["Campo", "Valor"]))

st.divider()
st.caption(
    "⚕️ Este sistema é um auxílio ao diagnóstico e **não substitui** a avaliação clínica. "
    "| POSTECH — Tech Challenge Fase 04 — Data Analytics"
)
