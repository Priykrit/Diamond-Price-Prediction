## Diamond Price Prediction

> Comprehensive Diamond Price Prediction project adhering to **_Industry Standards_**.
>
> ![dp](https://github.com/Priykrit/Diamond-Price-Prediction/assets/98400044/827d85c4-9322-4359-95b9-72fbaf416ffb)

> Developed **_Pipelines_** for training and prediction, including **_automatic model selection_** based on R2-Score. Implemented robust logging and exception-handling mechanisms.
>
> Selected Random Forest Regressor as the best model with an _**R2-Score**_ of _**0.977**_ for predicting the price of diamonds.
>
> Utilized Streamlit to create an interactive interface with dedicated sections for:
>
> > Exploratory Data Analysis (EDA) included visualizations such as:
> > >DataFrame of Data used
> > >
> > > Heatmap of Correlation matrix
> > >
> > >Hist plot of Numerical Columns
> > >
> > >Count plot of Categorical Column.
> > >
>
> > Dedicated page for prediction of the price of diamonds.

## To Run project
> Create venv with **_python==3.8.19_**
> 
> Install all requirements with **_pip install -r requirements.txt_**
>
> Run **_python src/pipelines/training_pipeline.py_** in terminal from root dir to create preprocessor and model files
>
> Run **_streamlit run application.py_** in terminal from root dir to run the web app
