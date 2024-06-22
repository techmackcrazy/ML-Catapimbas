import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
import time
import zipfile

# Título do Aplicativo
st.set_page_config(page_title='Catapimbas: Modelo Preditivo de Vendas', page_icon='🚗')
st.title('🚗 Catapimbas: Modelo Preditivo de Vendas')

with st.expander('Sobre este Aplicativo'):
  st.markdown('**O aplicativo propõe integrar todos os dados de entrada no modelo preditivo que vai entregar, de forma visual e inteligente, diversos insight para tomada estretégica de decisão.**')
  st.info('Este aplicativo auxilia o usuário a construir um modelo preditivo utilizando o conceito de regressão. Basta adicionar o seu arquivo base e ver a magia acontecer diante dos seus olhos!')

  st.markdown('**Como usar este aplicativo?**')
  st.warning('É muito simples. A sua esquerda, no tópico 01, você irá adicionar a sua base de estudo. No tópico 02, é onde serão ajustado os parâmetros do modelo. Como resultado, o modelo será iniciados, apresentando seus resultados e permitindo que você faça os downloads dos modelos gerados.')

  st.markdown('**Vem ver por baixo dos panos:**')
  st.markdown('Bibliotecas utilizadas:')
  st.code('''- Pandas para fazer a análise exploratória;
- Scikit-learn para construir o modelo de machine learning;
- Altair para criação visual das apresentações;
- Streamlit para criação da interface final do usuário.
  ''', language='markdown')


# Criação da barra lateral para colocar os dados de entrada
with st.sidebar:
    # Database
    st.header('1. Database de Entrada')

    st.markdown('**Use sua base de dados**')
    arquivo_upload = st.file_uploader("Faça o upload do seu csv aqui:")
    if arquivo_upload is not None:
        df = pd.read_csv(arquivo_upload, index_col=False)

    st.header('2. Defina seus Parâmetros')
    split_size = st.slider('Proporção da divisão dos dados (% para o treino)', 10, 90, 75, 5)

    st.subheader('2.1. Parâmetros de Aprendizado')
    with st.expander('Abrir parâmetros'):
        n_estimators = st.slider('Número de nós (n_estimators)', 0, 1000, 100, 50)
        max_features = st.select_slider('Atributo de divisão (max_features)', options=['todos', 'sqrt', 'log2'])
        min_samples_split = st.slider('Quantidade mínima de exemplos para dividir um nó interno (min_samples_split)', 2, 10, 2, 1)
        min_samples_leaf = st.slider('Número mínimo de amostras para ser um nó folha (min_samples_leaf)', 1, 10, 2, 1)

    st.subheader('2.2. Parâmetros Gerais')
    with st.expander('Abrir parâmetros', expanded=False):
        random_state = st.slider('Random State:', 0, 1000, 42, 1)
        criterion = st.select_slider('Critério para determinar qualidade dos dados (criterion):', options=['squared_error', 'absolute_error'])
        bootstrap = st.select_slider('Amostras Bootstrap para construção das árvores? (bootstrap)', options=[True, False])
        oob_score = st.select_slider('Usar amostrar prontas para estimar o R2 em dados não vistos (oob_score)?', options=[False, True])

    sleep_time = st.slider('Sleep:', 0, 3, 0)

# Construção do modelo
if arquivo_upload: 
    with st.status("Processando...", expanded=True) as status:
    
        st.write("Carregando dados...")
        time.sleep(sleep_time)

        st.write("Preparando dados...")
        time.sleep(sleep_time)
        X = df.iloc[:,:-1]
        y = df.iloc[:,-1]
            
        st.write("Dividindo dados...")
        time.sleep(sleep_time)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = (100-split_size)/100, random_state = random_state)
    
        st.write("Treinando modelo...")
        time.sleep(sleep_time)

        if max_features == 'todos':
            max_features = None
            max_features_metric = X.shape[1]
        
        if max_features == 'sqrt':
            max_features = 'sqrt'
            max_features_metric = X.shape[1]
        
        if max_features == 'log2':
            max_features = 'log2'
            max_features_metric = X.shape[1]
        
        rf = RandomForestRegressor(
                n_estimators = n_estimators,
                max_features = max_features,
                min_samples_split = min_samples_split,
                min_samples_leaf = min_samples_leaf,
                random_state = random_state,
                criterion = criterion,
                bootstrap = bootstrap,
                oob_score = oob_score)
        rf.fit(X_train, y_train)
        
        st.write("Aplicando modelo para fazer as predições...")
        time.sleep(sleep_time)
        y_train_pred = rf.predict(X_train)
        y_test_pred = rf.predict(X_test)
            
        st.write("Avaliando métricas de desempenho...")
        time.sleep(sleep_time)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        st.write("Apresentando métricas de desempenho...")
        time.sleep(sleep_time)

        criterion_string = ' '.join([x.capitalize() for x in criterion.split('_')])

        rf_results = pd.DataFrame(['Random forest', train_mse, train_r2, test_mse, test_r2]).transpose()
        rf_results.columns = ['Method', f'Training {criterion_string}', 'Training R2', f'Test {criterion_string}', 'Test R2']

        # Convertendo objetos para números

        for col in rf_results.columns:
            rf_results[col] = pd.to_numeric(rf_results[col], errors = 'ignore')
        rf_results = rf_results.round(2)
        
    status.update(label = "Status", state = "complete", expanded = False)

    # Informações dos Dados para o usuário:
    st.header('Dados Importados', divider = 'blue')
    col = st.columns(4)
    col[0].metric(label = "N° de Amostras", value = X.shape[0], delta="")
    col[1].metric(label = "N° de Variáveis", value = X.shape[1], delta="")
    col[2].metric(label = "N° de Amostras de Treino", value=X_train.shape[0], delta="")
    col[3].metric(label = "N° de Amostras de Teste", value=X_test.shape[0], delta="")
    
    with st.expander('Dataset Inicial', expanded = True):
        st.dataframe(df, height = 210, use_container_width = True)
    with st.expander('Divisão do Treinamento', expanded = False):
        train_col = st.columns((3,1))
        with train_col[0]:
            st.markdown('**X**')
            st.dataframe(X_train, height = 210, hide_index = True, use_container_width = True)
        with train_col[1]:
            st.markdown('**Y**')
            st.dataframe(y_train, height = 210, hide_index=True, use_container_width=True)
    with st.expander('Divisão do Teste', expanded = False):
        test_col = st.columns((3,1))
        with test_col[0]:
            st.markdown('**X**')
            st.dataframe(X_test, height = 210, hide_index = True, use_container_width = True)
        with test_col[1]:
            st.markdown('**Y**')
            st.dataframe(y_test, height = 210, hide_index = True, use_container_width = True)

    # Zipar os arquivos do dataset
    df.to_csv('dataset.csv', index = False)
    X_train.to_csv('X_train.csv', index = False)
    y_train.to_csv('Y_train.csv', index = False)
    X_test.to_csv('X_test.csv', index = False)
    y_test.to_csv('Y_test.csv', index = False)
    
    listar_arquivos = ['dataset.csv', 'X_train.csv', 'Y_train.csv', 'X_test.csv', 'Y_test.csv']
    with zipfile.ZipFile('dataset.zip', 'w') as zipF:
        for file in listar_arquivos:
            zipF.write(file, compress_type = zipfile.ZIP_DEFLATED)

    with open('dataset.zip', 'rb') as datazip:
        btn = st.download_button(
                label = 'Download ZIP',
                data = datazip,
                file_name = "dataset.zip",
                mime = "application/octet-stream"
                )
    
    # Apresentação dos parâmetros do modelo
    st.header('Parâmetros do Modelo', divider='blue')
    parameters_col = st.columns(3)
    parameters_col[0].metric(label = "Proporção da divisão dos dados (% para o treino)", value = split_size, delta = "")
    parameters_col[1].metric(label = "Número de nós (n_estimators)", value = n_estimators, delta = "")
    parameters_col[2].metric(label = "Atributo de divisão (max_features)", value = max_features_metric, delta = "")
    
    # Apresentação do plot dos recursos importantes
    importances = rf.feature_importances_
    feature_names = list(X.columns)
    forest_importances = pd.Series(importances, index = feature_names)
    df_importance = forest_importances.reset_index().rename(columns = {'index': 'feature', 0: 'value'})
    
    bars = alt.Chart(df_importance).mark_bar(size=40).encode(
             x='value:Q',
             y=alt.Y('feature:N', sort='-x')
           ).properties(height=250)

    performance_col = st.columns((2, 0.2, 3))
    with performance_col[0]:
        st.header('Performance do Modelo', divider = 'blue')
        st.dataframe(rf_results.T.reset_index().rename(columns = {'index': 'Parâmetro', 0: 'Valor'}))
    with performance_col[2]:
        st.header('Importância do Recurso', divider = 'blue')
        st.altair_chart(bars, theme = 'streamlit', use_container_width = True)

    # Resultados da predição
    st.header('Resultados da Predição', divider = 'blue')
    s_y_train = pd.Series(y_train, name = 'Atual').reset_index(drop = True)
    s_y_train_pred = pd.Series(y_train_pred, name = 'Predição').reset_index(drop = True)
    df_train = pd.DataFrame(data = [s_y_train, s_y_train_pred], index = None).T
    df_train['Classe'] = 'treino'
        
    s_y_test = pd.Series(y_test, name = 'Atual').reset_index(drop = True)
    s_y_test_pred = pd.Series(y_test_pred, name = 'Predição').reset_index(drop = True)
    df_test = pd.DataFrame(data = [s_y_test, s_y_test_pred], index = None).T
    df_test['Classe'] = 'teste'
    
    df_prediction = pd.concat([df_train, df_test], axis = 0)
    
    prediction_col = st.columns((2, 0.2, 3))
    
    # Apresentação do Dataframe
    with prediction_col[0]:
        st.dataframe(df_prediction, height = 320, use_container_width=True)

    # Apresentação do Scatter Plot com o valores atuais X Preditos
    with prediction_col[2]:
        scatter = alt.Chart(df_prediction).mark_circle(size = 60).encode(
                        x = 'Atual',
                        y = 'Predição',
                        color = 'Classe'
                  )
        st.altair_chart(scatter, theme = 'streamlit', use_container_width = True)

    
# Perguntar ao usuário quando não foi feito o upload de nenhum arquivo CSV
else:
    st.warning('⬅️ Vamos começar? Faça o upload de um arquivo CSV!')
