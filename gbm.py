import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import streamlit as st
import joblib

# 读取训练集数据
train_data = pd.read_csv('train_data.csv')

# 分离输入特征和目标变量
X = train_data[['Primary Site', 'Histologic', 'Grade',
                   'T stage', 'N stage', 'Brain metastasis', 'Liver metastasis', 'Bone metastasis']]
y = train_data['lung metastasis']

# 创建并训练GBM模型
gbm_model = GradientBoostingClassifier()
gbm_model.fit(X, y)


# 特征映射
feature_order = [
    'Primary Site', 'Histologic', 'Grade', 'T stage', 'N stage',
    'Brain metastasis', 'Liver metastasis', 'Bone metastasis'
]
class_mapping = {0: "No lung metastasis", 1: "Esophagus cancer lung metastasis"}
primary_site_mapper = {"Upper third of esophagus": 3, "Middle third of esophagus": 1, "Lower third of esophagus": 2}
histologic_mapper = {"Adenocarcinoma": 2, "Squamous–cell carcinoma": 1}
grade_mapper = {"Grade I+II": 3, "Grade III": 1, "Grade Ⅳ": 2}
t_stage_mapper = {"T1": 4, "T2": 1, "T3": 2, "T4": 3}
n_stage_mapper = {"N0": 4, "N1": 1, "N2": 2, "N3": 3}
brain_metastasis_mapper = {"NO": 0, "Yes": 1}
liver_metastasis_mapper = {"NO": 0, "Yes": 1}
bone_metastasis_mapper = {"NO": 0, "Yes": 1}

# 预测函数
def predict_lung_metastasis(primary_site, histologic, grade,
                            t_stage, n_stage, brain_metastasis, liver_metastasis, bone_metastasis):
    input_data = pd.DataFrame({
        'Primary Site': [primary_site_mapper[primary_site]],
        'Histologic': [histologic_mapper[histologic]],
        'Grade': [grade_mapper[grade]],
        'T stage': [t_stage_mapper[t_stage]],
        'N stage': [n_stage_mapper[n_stage]],
        'Brain metastasis': [brain_metastasis_mapper[brain_metastasis]],
        'Liver metastasis': [liver_metastasis_mapper[liver_metastasis]],
        'Bone metastasis': [bone_metastasis_mapper[bone_metastasis]],
    }, columns=feature_order)

    prediction = gbm_model.predict(input_data)[0]
    probability = gbm_model.predict_proba(input_data)[0][1]  # 获取属于类别1的概率
    class_label = class_mapping[prediction]
    return class_label, probability

# 创建Web应用程序
st.title("GBM Model Predicting lung Metastasis of Esophageal Cancer")
st.sidebar.write("Variables")

primary_site = st.sidebar.selectbox("Primary Site", options=list(primary_site_mapper.keys()))
histologic = st.sidebar.selectbox("Histologic", options=list(histologic_mapper.keys()))
grade = st.sidebar.selectbox("Grade", options=list(grade_mapper.keys()))
t_stage = st.sidebar.selectbox("T Stage", options=list(t_stage_mapper.keys()))
n_stage = st.sidebar.selectbox("N Stage", options=list(n_stage_mapper.keys()))
brain_metastasis = st.sidebar.selectbox("Brain Metastasis", options=list(brain_metastasis_mapper.keys()))
liver_metastasis = st.sidebar.selectbox("Liver Metastasis", options=list(liver_metastasis_mapper.keys()))
bone_metastasis = st.sidebar.selectbox("Bone Metastasis", options=list(bone_metastasis_mapper.keys()))

if st.button("Predict"):
    prediction, probability = predict_lung_metastasis(primary_site, histologic, grade,
                                                     t_stage, n_stage, brain_metastasis, liver_metastasis, bone_metastasis )

    st.write("Class Label: ", prediction)  # 结果显示在右侧的列中
    st.write("Probability of developing lung metastasis: ", probability)  # 结果显示在右侧的列中