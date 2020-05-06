import dash_core_components as dcc
import dash_table
import dash_html_components as html
import dash
from dash.dependencies import Input, Output
import flask
import pandas as pd
import plotly
import random
import plotly.graph_objs as go
from collections import deque
import dash
import time
from plotly.tools import mpl_to_plotly
import dash_bootstrap_components as dbc

#############################################################

import plotly as px
import plotly.graph_objects as go
import plotly.express as pl
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

theme =  {
    'dark': True,
    'detail': '#007439',
    'primary': '#00EA64',
    'secondary': '#6E6E6E',
}

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])


dfOg = pd.read_excel('dataset.xlsx')
dfTime = dfOg
dfTime['Stay'] = (dfTime['travel_end_date'] - dfTime['travel_start_date']) / pd.offsets.Day(1)

dfTime = dfTime.set_index('travel_start_date')


dfExp = dfOg[
    ['max_amount', 'distance', 'expense_num', 'total_expense', 'air_fare', 'hotel', 'mileage', 'travel_start_date',
     'travel_end_date',
     'car_rental', 'per_diem', 'per_diem_based_on_rate']]

dfExp['travel_end_date'] = pd.to_datetime(dfExp['travel_end_date'])
dfExp['travel_start_date'] = pd.to_datetime(dfExp['travel_start_date'])
dfExp['Stay'] = (dfExp['travel_end_date'] - dfExp['travel_start_date']) / pd.offsets.Day(1)
yT = dfExp[['total_expense']]

dfExp = dfExp.drop(
    ['car_rental', 'air_fare', 'hotel', 'mileage', 'expense_num', 'total_expense', 'travel_start_date',
     'travel_end_date', 'per_diem_based_on_rate'], axis=1)
dfExp.head()

scaler = preprocessing.MinMaxScaler()

dfExp = scaler.fit_transform(dfExp)

X_train, X_test, y_train, y_test = train_test_split(dfExp, yT, test_size=0.33, random_state=42)

clf = Ridge(alpha=1.0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
clf.score(X_test, y_test, sample_weight=None)
list_of_lists = y_pred
flattened = [val for sublist in list_of_lists for val in sublist]

dfOut = dfOg
dfOut = dfOut.drop(['travel_start_date','per_diem_based_on_rate','approval_date','taxi','travel_end_date','air_fare','mileage_based_on_rate', 'mileage','hotel', 'car_rental', 'per_diem'],axis=1)
dfOut['TotalExpensePred'] = pd.DataFrame(flattened)

dfOut.to_excel('outputRidge.xlsx')

dfLogistic = pd.read_excel('outputRidge.xlsx')


app.layout = html.Div(children=[
                dbc.Navbar([html.A(dbc.Row([dbc.Col(dbc.NavbarBrand("Financial Statement Fraud Detection", className="ml-6")),],
                                           align="center",no_gutters=True,),href="https://plot.ly",),
                                            dbc.NavbarToggler(id="navbar-toggler")],color="dark",dark=True,),
                dbc.Row(dbc.Col(html.Div("Time Series Anomaly : Support Vector Machine"), md=7)),
                dbc.Row(dbc.Col(html.H5("Select Employee(s) Name"), width={"size": 10})),
                dbc.Nav([html.H5(children='',className='nine columns')],justified=True),
                dcc.Dropdown(id='Input',
                        options=[{'label': 'Jane Duzetsky', 'value': 'Jane Duzetsky'},
                            {'label': 'Carl Svenson', 'value': 'Carl Svenson'},
                            {'label': 'Carl John Karino', 'value': 'John Karino'},
                            {'label': 'Ian Botham', 'value': 'Ian Botham'},
                            {'label': 'Tom Dolan', 'value': 'Tom Dolan'},
                            {'label': 'Jennifer Johnson', 'value': 'Jennifer Johnson'}
                        ],multi=True,value=['Jane Duzetsky']),
                html.Div(id='Output-graph'),

                dbc.Row(dbc.Col(html.H6("Output files saved"), width={"size": 10})),

                dbc.Row(dbc.Col(html.Div("Expense Forecasting : Ridge Regression"), md=7)),

                dbc.Nav([html.H6(children='Total Expense Table',className='nine columns')],justified=True),

                dash_table.DataTable(id='table',style_as_list_view=True,style_header={'backgroundColor': 'rgb(30, 30, 30)','color': 'white'},
                                    style_data={'height': 'auto'},
                                    style_table={'maxHeight': '500px','maxWidth': '1750px','overflowY': 'scroll'},
                                    columns=[{"name": i, "id": i} for i in dfLogistic.columns],data=dfLogistic.to_dict('records'),
                                    style_data_conditional=[ {'if': {'column_id': str('TotalExpensePred')},
                                    'color': 'red'} for x in dfLogistic.columns.tolist()]),

                dbc.Row(dbc.Col(html.Div("Forecasting Comparisons"), md=7)),

                dcc.Dropdown(
                        id='Input1',
                        options=[
                            {'label': 'Jane Duzetsky', 'value': 'Jane Duzetsky'},
                            {'label': 'Tom Dolan', 'value': 'Tom Dolan'},
                            {'label': 'Carl John Karino', 'value': 'John Karino'}
                        ],value=['Jane Duzetsky'],multi=True
                    ),
                html.Div(id='Output-graph1'),
                ])

@app.callback(dash.dependencies.Output('Output-graph','children'),
              [dash.dependencies.Input(component_id='Input',component_property='value')])

def graph_update(input_data):

    graphs = []
    for names in input_data:
        dfTime1 = dfTime.where(dfTime["sales_person_name"] == str(names))
        dfplot = dfTime1[
            ['Stay', 'expense_num', 'total_expense', 'max_amount', 'air_fare', 'hotel', 'distance', 'car_rental',
             'per_diem_based_on_rate']]
        dfAno = dfTime1[['Stay', 'expense_num', 'total_expense', 'air_fare', 'hotel', 'car_rental', 'per_diem',
                        'per_diem_based_on_rate']]

        dfAno = dfAno.fillna(0)
        scaler = preprocessing.MinMaxScaler()
        dfAno = scaler.fit_transform(dfAno)
        model = OneClassSVM(nu=0.01, kernel="rbf", gamma=0.01)
        model.fit(dfAno)

        dfplot['anomalySVM'] = np.array(model.predict(dfAno))
        a = dfplot.loc[dfplot['anomalySVM'] == -1, ['travel_start_date', 'total_expense']]
        dfOg['SVMout'] = np.array(model.predict(dfAno))

        # fig = go.Figure()
        #
        # fig.add_trace(go.Scatter(x=dfTime.index, y=dfTime['total_expense'], name="Expenses", line_color='deepskyblue'))
        # fig.add_trace(go.Scatter(x=a.index, y=a['total_expense'], mode='markers'))
        # fig.update_layout(title_text='Expense : ' + input_data, xaxis_rangeslider_visible=True)

        dfOg.to_excel('outputSVM.xlsx')
        data = [go.Scatter(x=dfTime1.index, y=dfTime1['total_expense'], name="Expenses", line_color='deepskyblue'),
                        go.Scatter(x=a.index, y=a['total_expense'], mode='markers')]
        graphs.append(html.Div(dcc.Graph(id =names,
        figure={'data':data,'layout': go.Layout(title_text='Expense : ' + str(names), xaxis_rangeslider_visible=True)})))
    return graphs

@app.callback(Output(component_id='Output-graph1',component_property='children'),
              [dash.dependencies.Input(component_id='Input1',component_property='value')])

def graph_update1(input1):
    graph = []

    for name in input1:
        dfLogistic1 = dfLogistic.where(dfLogistic["sales_person_name"] == str(name))

        dfOg.head()
        data = [go.Scatter(x=dfLogistic1.index, y=dfLogistic1['TotalExpensePred'], name="Expenses", line_color='deepskyblue'),
                go.Scatter(x=dfLogistic1.index, y=dfLogistic1['total_expense'])]
        graph.append(dcc.Graph(id =name,figure={'data':data,'layout': go.Layout(title_text='Prediction Comparison', xaxis_rangeslider_visible=True)}))

    return graph

if __name__ == '__main__':
    app.run_server(debug=True)
