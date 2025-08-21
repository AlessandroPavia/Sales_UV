import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.express as px
from scipy import stats

def detect_outliers(df, col_name, method="iqr", threshold=1.5):
    """
    Rileva outlier in una colonna usando IQR o Z-score.
    
    Parametri:
    - df: pd.DataFrame
    - col_name: nome della colonna su cui fare outlier detection
    - method: 'iqr' o 'zscore'
    - threshold: soglia per il metodo scelto
    
    Ritorna:
    - outliers: pd.Series con valori considerati outlier
    - df_clean: DataFrame senza gli outlier
    - report: DataFrame con info dettagliate su ciascun valore
    """
    col_data = df[col_name]
    report = pd.DataFrame({
        "Date": df['Order Date_def'],
        "Value": col_data,
        "Method": method,
        "Threshold": threshold
    })

    if method.lower() == "iqr":
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (col_data < lower_bound) | (col_data > upper_bound)

    elif method.lower() == "zscore":
        z_scores = np.abs(stats.zscore(col_data, nan_policy='omit'))
        mask = z_scores > threshold

    else:
        raise ValueError("Metodo non valido: scegliere 'iqr' o 'zscore'")

    report["Outlier"] = mask

    outliers = col_data[mask]

    return outliers, report

def find_hierarchies(df, candidate_cols=None, exclude_cols=None, min_children_per_parent=2):
    # 1) scegli le colonne candidate (categoriche/testuali)
    if candidate_cols is None:
        candidate_cols = [
            c for c in df.columns
            if (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]))
        ]
    if exclude_cols:
        candidate_cols = [c for c in candidate_cols if c not in set(exclude_cols)]

    results = []
    for child in candidate_cols:
        for parent in candidate_cols:
            if child == parent:
                continue

            # 2) parent non costante
            if df[parent].nunique(dropna=True) < 2:
                continue

            # 3) child più granulare del parent
            if df[child].nunique(dropna=True) <= df[parent].nunique(dropna=True):
                continue

            # 4) per ogni child, il parent deve essere deterministico (1 solo valore)
            sizes = df.groupby(child, dropna=True)[parent].nunique(dropna=True)
            if not (sizes == 1).all():
                continue

            # 5) opzionale: ogni parent abbia almeno N child in media
            avg_children = df.groupby(parent, dropna=True)[child].nunique(dropna=True).mean()
            if avg_children < min_children_per_parent:
                continue

            results.append({
                "child": child,
                "parent": parent,
                "nunique_child": int(df[child].nunique(dropna=True)),
                "nunique_parent": int(df[parent].nunique(dropna=True))
            })
    df_results = pd.DataFrame(results)

    if df_results.empty:
        return pd.DataFrame(columns=[
            "child","parent","nunique_child","nunique_parent"])

    else:
        return df_results.sort_values(
            by=["nunique_child"],
            ascending=[False]
        ).reset_index(drop=True)



ordine_mesi=['January','February','March','April','May','June','July','August','September','October','November','December']
ordine_giorni=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

st.set_page_config(page_title="Dataset Explorer", layout="wide")

# Drag & Drop del file
uploaded_file = st.file_uploader("📂 Carica un file CSV", type=["csv"])

if uploaded_file is not None:
    # Leggi direttamente il CSV caricato
    df = pd.read_csv(uploaded_file, parse_dates=["Order Date_def", "Ship Date_def"])

    # Trasforma tutte le colonne di tipo object in string[python]
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].astype('string')

    #Ordino i mesi e i giorni della settimana
    df['Order Month'] = pd.Categorical(
            df['Order Month'],
            categories=ordine_mesi,
            ordered=True
            )
    df['Order Day of the Week']=pd.Categorical(df['Order Day of the Week'], categories=ordine_giorni, ordered=True)
    
    #applico la funzione che ricerca le gerarchie
    df_gerarchie=find_hierarchies(df)


    scelta = st.selectbox(
    "Cosa vuoi vedere?",
    ["Dataset completo", "Dataset details", "Dataset visualization"])

    if scelta == "Dataset completo":
        st.title("Dataset completo")
        st.dataframe(df)

    elif scelta == "Dataset details":
        st.title("Dataset Details")


        # Tabella statistiche
        table_dict = {}
        for col in df.columns:
            col_data = df[col]
            stats = {}
            stats['Storage Type'] = str(col_data.dtype)
            stats['Count'] = col_data.count()
            stats['Missing'] = col_data.isna().sum()
        
            if pd.api.types.is_numeric_dtype(col_data):
                stats['Mean'] = round(col_data.mean(), 4)
                stats['Min'] = col_data.min()
                stats['Max'] = col_data.max()
                stats['StDev'] = round(col_data.std(), 4)
                stats['Unique'] = col_data.nunique()
                stats['Top Frequency'] = '--'
            else:
                stats['Mean'] = '--'
                stats['Min'] = '--'
                stats['Max'] = '--'
                stats['StDev'] = '--'
                stats['Unique'] = col_data.nunique()
                stats['Top Frequency'] = col_data.value_counts().iloc[0]
        
            table_dict[col] = stats

        summary_df = pd.DataFrame(table_dict).T
        summary_df = summary_df.T  

        st.dataframe(summary_df)


        # Selectbox per scegliere la colonna
        col_selected = st.selectbox("Seleziona una colonna", df.columns)

        col_data = df[col_selected]

        # Grafico solo se unique < 30
        if col_data.nunique() < 30 and pd.api.types.is_numeric_dtype(col_data):
            min_val = col_data.min()
            max_val = col_data.max()
            # Istogramma
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            sns.histplot(col_data.dropna(), bins=50, kde=False, ax=ax1, color="skyblue", alpha=0.7)
            ax1.set_title(f"Istogramma: {col_selected}")
            plt.xlim(min_val,max_val)
            st.pyplot(fig1)

            # Boxplot
            fig2, ax2 = plt.subplots(figsize=(8, 2))  # più basso, i boxplot non hanno bisogno di altezza
            sns.boxplot(x=col_data.dropna(), ax=ax2, showfliers=True, color="orange")
            ax2.set_title(f"Boxplot: {col_selected}")
            plt.xlim(min_val,max_val)
            st.pyplot(fig2)

        elif pd.api.types.is_numeric_dtype(col_data):
            st.info(f"La colonna '{col_selected}' ha troppi valori unici ({col_data.nunique()}). Nessun grafico generato.")
        
        elif col_data.nunique()<30 and not pd.api.types.is_numeric_dtype(col_data):
            fig,ax=plt.subplots(figsize=(8,5))
            sns.histplot(col_data.dropna().astype(str), ax=ax, color='blue')
            plt.xticks(rotation=45)
            st.pyplot(fig)
         

        elif col_selected=='State_def':
            for regione in df['Region'].dropna().unique():
                fig, ax = plt.subplots(figsize=(8, 4))
                subset = df[df["Region"] == regione]["State_def"]
                sns.countplot(x=subset, ax=ax, color="skyblue")
                ax.set_title(f"Regione: {regione}")
                ax.set_xlabel("Stato")
                ax.set_ylabel("Conteggio")
                plt.xticks(rotation=50)
                st.pyplot(fig)


        
        else:
            st.info(f"La colonna '{col_selected}' contiene troppi valori unici. Nessun grafico generato.")

       

    
    elif scelta == "Dataset visualization":
        st.title("Dataset Visualization")
        st.header('Correlation graph:')
         # Calcolo matrice di correlazione
        corr = df.corr(numeric_only=True)

        # Slider per soglia
        threshold = st.slider("Soglia di correlazione", 0.0, 1.0, 0.7, 0.05)
        

        # Creazione grafo
        G = nx.Graph()
        for col in corr.columns:
            G.add_node(col)

        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                value = corr.iloc[i, j]
                if abs(value) >= threshold:
                    color = "green" if value > 0 else "red"
                    G.add_edge(corr.columns[i], corr.columns[j], weight=abs(value), color=color)

        # Layout fisso in cerchio
        pos = nx.circular_layout(G)

        # Colori e spessori archi
        edges = G.edges() #recupero gli archi, ciascuno è una coppia di nodi tipo: [(col1,col3), (col 2, col5)]
        colors = [G[u][v]["color"] for u, v in edges] 
        

        # Disegno grafo
        fig, ax = plt.subplots(figsize=(8, 6))
        nx.draw(
            G, pos, with_labels=True,
            edge_color=colors,
            node_size=1500, node_color="lightblue",
            font_size=10, ax=ax
        )

        st.pyplot(fig)


        st.subheader("Esplora le correlazioni tra colonne")

        col1 = st.selectbox("Seleziona la prima colonna", corr.columns)
        # Trovo le colonne correlate sopra soglia
        correlated_cols = corr.index[abs(corr[col1]) >= threshold].drop(col1)

        if len(correlated_cols) > 0:
            col2 = st.selectbox("Seleziona una colonna correlata", correlated_cols)

            # Scatterplot delle due colonne
            st.subheader(f"Scatterplot: {col1} vs {col2}")
            fig2, ax2 = plt.subplots(figsize=(7, 5))
            ax2.scatter(df[col1], df[col2], alpha=0.6)
            ax2.set_xlabel(col1)
            ax2.set_ylabel(col2)
            ax2.set_title(f"Correlazione: {corr.loc[col1, col2]:.2f}")
            st.pyplot(fig2)
        else:
            st.info(f"Nessuna colonna correlata con '{col1}' sopra la soglia {threshold}.")

        st.subheader('Ulteriori grafici:')
        col3=st.selectbox('Seleziona la colonna di cui vuoi vedere altri grafici:', df.columns)
        datetime_cols = df.select_dtypes(include="datetime").columns.tolist()
        category_cols= df.select_dtypes(include='category').columns.tolist()
        col_temporale=st.selectbox('Per i grafici temporali, scegliere quale colonna usare:', datetime_cols)
        
        if pd.api.types.is_numeric_dtype(df[col3]) and df[col3].nunique() > 100: #quindi nel nostro caso ['Sales_def','Profit_def']:
            df_grouped_col3=df.groupby(col_temporale)[col3].sum().reset_index()
            fig3,ax3=plt.subplots(figsize=(8,5))
            sns.lineplot(data=df_grouped_col3, x=col_temporale,y=col3, ax=ax3)
            ax3.set_xlabel("Tempo")
            ax3.set_ylabel(col3)
            ax3.set_title(f"Andamento di {col3} nel tempo")

            st.pyplot(fig3)
            scelta_outliers=st.selectbox('Vuoi vedere gli outlier evidenziati per il grafico precedente?', ['Si', 'No'])
            
            if scelta_outliers=='No':
                pass
                
            
            else:
                outliers_sales, report = detect_outliers(df_grouped_col3, col_name=col3, method="iqr", threshold=1.5)
                outliers = report[report['Outlier'] == True]

                # Lineplot con outlier evidenziati
                fig_out, ax_out = plt.subplots(figsize=(8,5))
                sns.lineplot(data=df_grouped_col3, x=col_temporale, y=col3, ax=ax_out)

                # Usa le date degli outlier per l'asse x
                ax_out.scatter(outliers['Date'], outliers['Value'], color='red', marker='x', s=100, label='Outlier')

                ax_out.set_xlabel("Tempo")
                ax_out.set_ylabel(col3)
                ax_out.set_title(f"Andamento di {col3} con outlier evidenziati")
                ax_out.legend()
                st.pyplot(fig_out)
                

            col_category=st.selectbox('Decidere la colonna per cui raggruppare i dati:',category_cols)
            df_grouped_col3_stati=df.groupby([col_category,'State_def','Category_def'])[col3].sum().reset_index()
            top5stati=df.groupby('State_def')[col3].sum().nlargest(5).index
            for stato in top5stati:
                df_stato = df_grouped_col3_stati[df_grouped_col3_stati['State_def'] == stato].sort_values(col_category)
    
                fig4,ax4 = plt.subplots(figsize=(10,6))
                sns.lineplot(data=df_stato, x=col_category, y=col3, hue='Category_def', marker='o', ax=ax4)
    
                ax4.set_title(f"Andamento di {col3} nel tempo per lo stato {stato} per categoria di prodotto ")
                ax4.set_xlabel("Tempo")
                ax4.set_ylabel(col3)
                ax4.grid(True)
                plt.xticks(rotation=45)
    
                st.pyplot(fig4)
            



            state_to_code = {
            'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR',
            'CALIFORNIA': 'CA', 'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE',
            'FLORIDA': 'FL', 'GEORGIA': 'GA', 'HAWAII': 'HI', 'IDAHO': 'ID',
            'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA', 'KANSAS': 'KS',
            'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD',
            'MASSACHUSETTS': 'MA', 'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS',
            'MISSOURI': 'MO', 'MONTANA': 'MT', 'NEBRASKA': 'NE', 'NEVADA': 'NV',
            'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ', 'NEW MEXICO': 'NM',
            'NEW YORK': 'NY', 'NORTH CAROLINA': 'NC', 'NORTH DAKOTA': 'ND',
            'OHIO': 'OH', 'OKLAHOMA': 'OK', 'OREGON': 'OR', 'PENNSYLVANIA': 'PA',
            'RHODE ISLAND': 'RI', 'SOUTH CAROLINA': 'SC', 'SOUTH DAKOTA': 'SD',
            'TENNESSEE': 'TN', 'TEXAS': 'TX', 'UTAH': 'UT', 'VERMONT': 'VT',
            'VIRGINIA': 'VA', 'WASHINGTON': 'WA', 'WEST VIRGINIA': 'WV',
            'WISCONSIN': 'WI', 'WYOMING': 'WY'
            }

            df_state_sales = df.groupby('State_def')[col3].sum().reset_index()
            df_state_sales['state_code'] = df_state_sales['State_def'].map(state_to_code) #funziona solo con gli stati scritti come sigle, quindi è necessario fare questa mappatura
                                                                                #l'etichetta(hover_name) invece si può fare con il nome intero

            df_state_sales[col3 + "_formatted"] = df_state_sales[col3].apply(
            lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") + " €"
            )

            fig = px.choropleth(
                df_state_sales,
                locations='state_code',
                locationmode='USA-states',
                color=col3,  # numerico, usato per la scala colori
                scope='usa',
                color_continuous_scale='Turbo',
                hover_name='State_def',
                hover_data={
                'state_code': False,
                col3: False,  # nasconde la colonna numerica
                col3 + "_formatted": True  # mostra quella formattata
                },
            )
            fig.update_layout(
            title=dict(
            text=f"Distribuzione di {col3} per Stato",
            x=0.3,
            xanchor="left",
            yanchor="top",
            font=dict(size=15, family='helvetica')  # puoi provare anche 14 o 18
            )
            )
            st.plotly_chart(fig, use_container_width=True)



            #piechart
            colonne_piechart = [col for col in df.columns if df[col].nunique() < 13]  # meno di 13 valori unici
            colonna_piechart=st.selectbox('Seleziona la colonna con cui costruire il grafico a torta',colonne_piechart)
            vendite_per_colonna_scelta=df.groupby(colonna_piechart)[col3].sum().reset_index()
            plt.figure(figsize=(10,7))
            plt.pie(
            vendite_per_colonna_scelta[col3], #valori
            labels=vendite_per_colonna_scelta[colonna_piechart], #etichette
            autopct='%1.1f%%', #percentuale con un numero decimale
            startangle=90, #angolo di partenza
            colors=plt.cm.Paired.colors 
            )
            plt.title(f'Distribuzione di {col3} in base a {colonna_piechart}')
            plt.axis('equal')  # Per avere la torta rotonda
            st.pyplot(plt)

        
        '''
        for col3 in df.columns:  
            if pd.api.types.is_string_dtype(df[col3]):
                mask = (
                ((df_gerarchie["parent"] == col3) & 
                (df_gerarchie["nunique_parent"] <= 50) & 
                (df_gerarchie["nunique_child"] <= 50)) |
                ((df_gerarchie["child"] == col3) & 
                (df_gerarchie["nunique_parent"] <= 50) & 
                (df_gerarchie["nunique_child"] <= 50))
                )
                if mask.any():   # cioè esiste almeno una riga che rispetta tutte le condizioni
                # fai quello che ti serve

                    

            #heatmap

            # Ciclo sulle regioni
                    for regione in df['Region'].unique():
                        # filtro solo quella regione
                        df_regione = df[df['Region'] == regione]

                        # creo matrice stati × mesi
                        ordini_state = df_regione.groupby(['State_def', 'Order Month'])['Order ID'].nunique().unstack(fill_value=0)

                        if ordini_state.empty:   # se la regione non ha dati, skippo
                            continue

                        # ordino gli stati in base al totale ordini (dall’alto al basso)
                        ordini_state_sorted = ordini_state.loc[ordini_state.sum(axis=1).sort_values(ascending=False).index]


                        fig5, ax5 = plt.subplots(figsize=(10, 6))
                        sns.heatmap(ordini_state_sorted, cmap="YlOrRd", linewidths=0.3)
                        plt.title(f"Numero di ordini per stato ({regione})")
                        plt.xlabel("Mese")
                        plt.ylabel("Stato")
                        plt.xticks(rotation=45)
                        st.pyplot(fig5)'''

else:
    st.info("⬆️ Carica un file CSV per iniziare.")
