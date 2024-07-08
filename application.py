import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.pipelines.prediction_pipeline import CustomData,PredictPipeline

df = pd.read_csv('notebooks\data\gemstone.csv')
df=df.drop(columns=['id'],axis=1)
numerical_columns = df.columns[df.dtypes!='object']
categorical_columns = df.columns[df.dtypes=='object']


st.set_option('deprecation.showPyplotGlobalUse', False)
title1,title2,title3 = st.columns([1,6,1])
title2.title('Diamond Price Prediction')
st.sidebar.subheader('Navigation')
nav = st.sidebar.radio('',['Home','EDA','Prediction'])

if nav == 'Home':
    img1,img2,img3 = st.columns([2.5,6,1])
    st.markdown("""
             ### Made By - _Priykrit Varma_
             ### priykritv@gmail.com, priykrit21100@iiitnr.edu.in
             ### Contact no. - 9109562757
             """)
    page_bg_img = '''
    <style>
    [data-testid="stApp"] {
    background-image: url("https://th.bing.com/th/id/R.7b1f96df61e34c00bc3039ad30d5f5b9?rik=nMouS49jnp%2blrA&riu=http%3a%2f%2f2.bp.blogspot.com%2f-CVKp91BHEv0%2fTxc1TM3IA_I%2fAAAAAAAAAHE%2fUuncGsBVjrA%2fs1600%2fdreamstime_l_14232138.jpg&ehk=CdAbE%2fNro0wtzQY1BfUeT9vLc%2b3qBiVrYV3m8g3Y2lI%3d&risl=&pid=ImgRaw&r=0");
    background-size: cover;
    
    }
    [data-testid="stHeadingWithActionElements"], 
    [data-testid="stHeadingWithActionElements"] * {
        color: #DCDCDC !important;
        font-weight: bold !important;
    }
    </style>
    '''

    st.markdown(page_bg_img, unsafe_allow_html=True)
elif nav == 'EDA':
    st.header('EDA')
    
    page_bg_img = '''
    <style>
    [data-testid="stApp"] {
    background-image: url("https://cdn.wallpapersafari.com/69/93/aNWVIG.jpg");
    background-size: cover;
    
    }
    [data-testid="stHeadingWithActionElements"], 
    [data-testid="stHeadingWithActionElements"] * {
        color: #DCDCDC !important;
        font-weight: bold !important;
    }
    </style>
    '''

    st.markdown(page_bg_img, unsafe_allow_html=True)
    
    if st.checkbox('Show Data'):
        st.write(df)
    
    if st.checkbox('Show Corelation Matrics'):
        corr = st.multiselect('Select desired Numrical Columns',numerical_columns,['carat', 'depth','table', 'x', 'y', 'z','price'])
        sns.heatmap(df[corr].corr(),annot=True)
        plt.title('Heat map of Selected columns')
        st.pyplot()
    
    if st.checkbox('Numrical Column Hist plot'):
        hist = st.multiselect('Select desired Numrical Columns',numerical_columns,['carat'])
        plt.figure(figsize=(15,20))
        y=1
        for i in hist:
            plt.subplot(4,2,y)
            sns.histplot(data=df,x=i,kde=True)
            y+=1
        st.pyplot()
    
    if st.checkbox('Categorical Column Count plot'):
        cat = st.multiselect('Select desired Categorical Columns',categorical_columns,['cut'])
        plt.figure(figsize=(15,20))
        z=1
        for j in cat: 
            plt.subplot(3,2,z)
            sns.countplot(data=df,x=j)
            z+=1
        st.pyplot()
    
else:
    page_bg_img = '''
    <style>
    [data-testid="stApp"] {
    background-image: url("https://bestanimaljewelry.com/wp-content/uploads/2019/04/BLACK-DIAMOND-1.jpg");
    background-size: cover;
    
    }
    [data-testid="stHeadingWithActionElements"], 
    [data-testid="stHeadingWithActionElements"] * {
        color: #DCDCDC !important;
        font-weight: bold !important;
    }
    </style>
    '''

    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.header('Enter values to predict price of diamond')
    c,d,t =st.columns(3)
    carat=c.text_input('Carat')
    depth=d.text_input('Depth')
    table=t.text_input('Table')
    xc,yc,zc = st.columns(3)
    x=xc.text_input('Enter Value of x')
    y=yc.text_input('Enter Value of y')
    z=zc.text_input('Enter Value of z')
    cc,co,cl = st.columns(3)
    cut=cc.selectbox('Select Value of cut',['Fair', 'Good', 'Very Good','Premium','Ideal'],index=None)
    color=co.selectbox('Select Value of color',['D', 'E', 'F', 'G', 'H', 'I', 'J'],index=None)
    clarity=cl.selectbox('Select Value of clarity',['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'],index=None)
    _,p,_ = st.columns(3)
    if p.button('Predict'):
        try:
            data = CustomData(float(carat),float(depth),float(table),float(x),float(y),float(z),cut,color,clarity)
            final_new_data = data.get_data_as_dataframe()
            predict_pipeline = PredictPipeline()
            pred = predict_pipeline.predict(final_new_data)
            
            result = round(pred[0],2)
            st.success(f'The estimated value of diamond will be around {result}')
        except Exception as e:
            st.error('Some error occured')
