import time
import streamlit as st
#" page_title：页面标题，str or None。page_icon：页面图标，s.image or Emoji "
#" layout：可以为centered或wide。如果是wide则为宽屏模式。建议在分辨率较低的情况下使用centered，并尽量减少复杂布局。"
#" initial_sidebar_state：在auto模式下，电脑端会自动显示sidebar，而移动端会隐藏sidebar。一般不需要修改。"
#" menu_items：应用右上角的功能框，可以加入你的自定义内容。"
st.set_page_config(
    page_title=" Machine Learning Application　",
    page_icon="🧊2689",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

# 设置网页标题
st.header('2023ML Medicine')

r24c1,r24c2,r24c3 = st.columns(3)
with r24c1:
    sCA199 = st.number_input('CA199', value=-1.12, step=0.01)
with r24c2:
    BMI = st.number_input('BMI',value=-0.99, step=0.01)
with r24c3:
    Tumor_diameter = st.number_input('Tumor_diameter',value=-0.88, step=0.01)

r24c1,r24c2,r24c3= st.columns(3)
with r24c1:
    Biliary_calculus = st.selectbox("Biliary_calculus",("No","Yes"))
with r24c2:
    VascularInvasion = st.selectbox("Vascular.invasion",("No","Yes"))
#with r24c3:
#    my_selectbox = st.selectbox("anatomic.hepatectomy",("No","Yes"))


predict = st.button('Predict')
#print(predict,type(predict))

def modelA(person1):
    import time
    import random
    random.seed(3)  # 改种子进行调整
    import numpy as np
    import pandas as pd
    from pysurvival.models.simulations import SimulationModel
    from pysurvival.models.multi_task import NeuralMultiTaskModel
    from pysurvival.utils.metrics import concordance_index
    from pysurvival.utils.display import integrated_brier_score
    from pysurvival.datasets import Dataset
    import lifelines

    person1 = person1

    ### 1 测试集
    data = pd.read_csv('data1.csv')
    time_column = 'day'
    event_column = 'Status'
    category_columns = [
        'CA199',
        'BMI', 'Biliary_calculus',
        'Vascular.invasion', 'anatomic.hepatectomy', 'Tumor_diameter']
    features = np.setdiff1d(category_columns, [time_column, event_column]).tolist()  # 6个输入属性
    X_train = data[features]  # 6组输入数据 X
    T_train = data[time_column]
    E_train = data[event_column]
    Y_train = pd.concat((pd.DataFrame(T_train), pd.DataFrame(E_train)), axis=1)  # day status 2组输入数据 y
    # print(Y_train,type(Y_train))

    ### 2 训练集
    data = pd.read_csv('recomemd_test1.1.csv')
    data.loc[len(data)] = person1
    # print(data,len(data))

    time_column = 'day'
    event_column = 'Status'
    category_columns = [
        'CA199',
        'BMI', 'Biliary_calculus',
        'Vascular.invasion', 'anatomic.hepatectomy', 'Tumor_diameter']
    # Creating the features
    features = np.setdiff1d(category_columns, [time_column, event_column]).tolist()
    X_test = data[features]
    T_test = data[time_column]
    E_test = data[event_column]
    # print('X_test', X_test)
    # print('T_test', T_test)
    # print('E_test', E_test)
    Y_test = pd.DataFrame(T_test)  # time 序列
    E_test = pd.DataFrame(E_test)  # Status 序列
    # print('Y_test', Y_test)
    # print('E_test', E_test)

    # 3 建立模型
    structure = [{'activation': 'ReLU', 'num_units': 150}, ]  # 结构
    n_mtlr = NeuralMultiTaskModel(structure=structure, bins=150)
    n_mtlr.fit(X_train, T_train, E_train, lr=1e-5, num_epochs=500,
               init_method='orthogonal', optimizer='rmsprop')

    # 4 模型预测
    data1 = X_test['anatomic.hepatectomy']
    treatment = data1.unique()  # 治疗方案[1,0]
    # print(treatment)
    treatment_1 = X_test.copy(deep=True)
    treatment_1['anatomic.hepatectomy'] = treatment[1]  # anatomic.hepatectomy全为0
    treatment_0 = X_test.copy(deep=True)
    treatment_0['anatomic.hepatectomy'] = treatment[0]  # anatomic.hepatectomy全为1
    treatment_0 = treatment_0.values
    treatment_1 = treatment_1.values
    # print('bbb',treatment_0,treatment_0.shape) # (182, 6)
    # print('bbb',treatment_1,treatment_1.shape) # (182, 6)
    h_i = n_mtlr.predict_risk(treatment_0)
    h_j = n_mtlr.predict_risk(treatment_1)
    # print('h_i:',h_i[:8],len(h_i))
    # print('h_j',h_j[:8],len(h_j))

    # 调整的参数
    rec_ij = h_j - h_i - 160  # 0-1
    # print('rec_ij:', rec_ij)
    recommend_treatment = (rec_ij > 0).astype(np.int32)
    ##print(recommend_treatment,len(recommend_treatment))
    #print('person1预测结果：', recommend_treatment[-1])

    return recommend_treatment[-1]


if VascularInvasion=="Yes":
    VI = 1
else:
    VI = 0
if Biliary_calculus=="Yes":
    BC = 1
else:
    BC = 0

# 肿瘤大小，BMI身体质量指数，CA199糖类抗原199等
person1 = [sCA199, BMI, Tumor_diameter, BC, VI, 1, 1, 1]  # 0
#print('个人数据：', person1)



if predict:
    # 运行模型
    print('这个人的数据：',person1)
    res = modelA(person1)
    # 输出结果
    if res==1:
        msg = str(res)+': YES'
    if res==0:
        msg = str(res)+': NO'
    st.info( msg )
    time.sleep(2)
    st.balloons()













