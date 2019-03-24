
# coding: utf-8

# # DATA 608: Module 4 Assignment
# By: Sang Yoon (Andy) Hwang
#     
# Date: 2019-03-22

# In[ ]:


import pandas as pd
import numpy as np


# In[4]:


url = 'https://data.cityofnewyork.us/resource/nwxe-4ae8.json'
trees = pd.read_json(url)


# In[6]:


soql_url = ('https://data.cityofnewyork.us/resource/nwxe-4ae8.json?' +        '$select=count(tree_id),boroname,spc_common, health,status' +         '&$where=health!=\'NaN\'' +        '&$group=boroname,health,status,spc_common').replace(' ', '%20')
soql_trees = pd.read_json(soql_url)

soql_trees.head()

# For Q.1
# x --> boro (good, fair, poor --  3 bars for each boro)
# y --> proportion of health
# Select health not null, no stump or dead trees
# input --> spc_common


# In[7]:


soql_url_2 = ('https://data.cityofnewyork.us/resource/nwxe-4ae8.json?' +        '$select=count(tree_id), boroname, health, steward' +         '&$where=health!=\'NaN\'' +        '&$group=health,steward, boroname').replace(' ', '%20')
soql_trees_2 = pd.read_json(soql_url_2)

soql_trees_2.head()
# Q.2-1
# x --> health (None, 1or2, 3or4, 4orMore -- 4 bars for each health)
# y --> proportion of steward
# Select health not null, no stump or dead trees
# input --> boro


# In[8]:


soql_url_3 = ('https://data.cityofnewyork.us/resource/nwxe-4ae8.json?' +        '$select=count(tree_id), spc_common, health, steward' +         '&$where=health!=\'NaN\'' +        '&$group=health,steward, spc_common').replace(' ', '%20')
soql_trees_3 = pd.read_json(soql_url_3)

soql_trees_3.head()
# Q.2-2
# x --> health (None, 1or2, 3or4, 4orMore -- 4 bars for each health)
# y --> proportion of steward
# Select health not null, no stump or dead trees
# input --> spc_common


# In[9]:


soql_trees_sum = soql_trees.groupby(['boroname', 'spc_common']).agg({'count_tree_id': [np.sum]})
soql_merged = pd.merge(soql_trees, soql_trees_sum, on=['boroname','spc_common'])
soql_merged.head()

# Got how many trees each boro has for each spc_common for each health. Divide this number by total number of trees for each spc_common by boro


# In[15]:


soql_trees_sum_2 = soql_trees_2.groupby(['boroname', 'health']).agg({'count_tree_id': [np.sum]})
soql_merged_2 = pd.merge(soql_trees_2, soql_trees_sum_2, on=['boroname','health'])
soql_merged_2.head()

# Got how many trees each boro has for each steward for each health. Divide this number by total number of trees for each health by boro.


# In[16]:


soql_trees_sum_3 = soql_trees_3.groupby(['spc_common', 'health']).agg({'count_tree_id': [np.sum]})
soql_merged_3 = pd.merge(soql_trees_3, soql_trees_sum_3, on=['spc_common','health'])
soql_merged_3.head()

# Got how many trees each spc_common has for each steward for each health. Divide this number by total number of trees for each health by spc_common.



# In[17]:


# Calculate total count for merged table.
# Get data for Q.1
soql_merged.columns = ['boroname', 'count_tree_id', 'health', 'spc_common', 'status', 'sum_cnt_tree_id']
soql_merged['prop_health'] = soql_merged['count_tree_id'] / soql_merged['sum_cnt_tree_id']
soql_merged.sort_values(by=['boroname','spc_common'])

q1_graph = soql_merged[['boroname','health','spc_common','prop_health']]
q1_graph.head()


# In[18]:


# Calculate total count for merged table.
# Get data for Q.2-1 (by boroname)
soql_merged_2.columns = ['boroname', 'count_tree_id', 'health', 'steward', 'sum_cnt_tree_id']
soql_merged_2['prop_steward'] = soql_merged_2['count_tree_id'] / soql_merged_2['sum_cnt_tree_id']
soql_merged_2.sort_values(by=['boroname','steward'])

q2_graph = soql_merged_2
q2_graph.head()


# In[19]:


# Calculate total count for merged table.
# Get data for Q.2-2 (by spc_common)
soql_merged_3.columns = ['count_tree_id', 'health', 'spc_common', 'steward', 'sum_cnt_tree_id']
soql_merged_3['prop_steward'] = soql_merged_3['count_tree_id'] / soql_merged_3['sum_cnt_tree_id']
soql_merged_3.sort_values(by=['spc_common','steward'])

q3_graph = soql_merged_3
q3_graph.head()


# In[ ]:


## Q1. done -- Graph
import dash
import dash_core_components as dcc
import dash_html_components as html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

df = q1_graph

available_indicators = df['spc_common'].unique()

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)



app.layout = html.Div([
    html.H1('Q.1 Prop. of health by boroname for each SPC'),
    html.Div('''
        spc_common
    '''),
    dcc.Dropdown(
        id='my-dropdown',
        options=[{'label': i, 'value': i} for i in available_indicators],
        value='Atlas cedar'
    ),
    dcc.Graph(
        id='example-graph'    
    )
    
])

@app.callback(
    dash.dependencies.Output('example-graph', 'figure'),
    [dash.dependencies.Input('my-dropdown', 'value')])

def update_output(selected_dropdown_value):
    dff = df[df['spc_common'] == selected_dropdown_value]
    figure = {
            'data': [
                {'x': dff.boroname[dff['health'] == 'Good'], 'y': dff.prop_health[dff['health'] == 'Good'], 'type': 'bar', 'name': 'Good'},
                {'x': dff.boroname[dff['health'] == 'Fair'], 'y': dff.prop_health[dff['health'] == 'Fair'], 'type': 'bar', 'name': 'Fair'},
                {'x': dff.boroname[dff['health'] == 'Poor'], 'y': dff.prop_health[dff['health'] == 'Poor'], 'type': 'bar', 'name': 'Poor'}
            ],
            'layout': {
                'title': 'Prop. of health by boroname'
            }
        }
    return figure 


if __name__ == '__main__':
    app.run_server()


# In[ ]:


## Q2-1. done -- Graph
import dash
import dash_core_components as dcc
import dash_html_components as html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

df = q2_graph

available_indicators = df['boroname'].unique()

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)



app.layout = html.Div([
    html.H1('Q.2 Prop. of Steward by health for each boro'),
    html.Div('''
        boroname
    '''),
    dcc.Dropdown(
        id='my-dropdown',
        options=[{'label': i, 'value': i} for i in available_indicators],
        value='Queens'
    ),
    dcc.Graph(
        id='example-graph'    
    )
    
])

@app.callback(
    dash.dependencies.Output('example-graph', 'figure'),
    [dash.dependencies.Input('my-dropdown', 'value')])

def update_output(selected_dropdown_value):
    dff = df[df['boroname'] == selected_dropdown_value]
    figure = {
            'data': [
                {'x': dff.health[dff['steward'] == 'None'], 'y': dff.prop_steward[dff['steward'] == 'None'], 'type': 'bar', 'name': 'None'},
                {'x': dff.health[dff['steward'] == '1or2'], 'y': dff.prop_steward[dff['steward'] == '1or2'], 'type': 'bar', 'name': '1or2'},
                {'x': dff.health[dff['steward'] == '3or4'], 'y': dff.prop_steward[dff['steward'] == '3or4'], 'type': 'bar', 'name': '3or4'},
                {'x': dff.health[dff['steward'] == '4orMore'], 'y': dff.prop_steward[dff['steward'] == '4orMore'], 'type': 'bar', 'name': '4orMore'}
            ],
            'layout': {
                'title': 'Prop. of Steward by health'
            }
        }
    return figure 


if __name__ == '__main__':
    app.run_server()


# In[ ]:


## Q2-2. done -- Graph
import dash
import dash_core_components as dcc
import dash_html_components as html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

df = q3_graph

available_indicators = df['spc_common'].unique()

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)



app.layout = html.Div([
    html.H1('Q.2 Prop. of Steward by health for each spc_common'),
    html.Div('''
        spc_common
    '''),
    dcc.Dropdown(
        id='my-dropdown',
        options=[{'label': i, 'value': i} for i in available_indicators],
        value='Atlas cedar'
    ),
    dcc.Graph(
        id='example-graph'    
    )
    
])

@app.callback(
    dash.dependencies.Output('example-graph', 'figure'),
    [dash.dependencies.Input('my-dropdown', 'value')])

def update_output(selected_dropdown_value):
    dff = df[df['spc_common'] == selected_dropdown_value]
    figure = {
            'data': [
                {'x': dff.health[dff['steward'] == 'None'], 'y': dff.prop_steward[dff['steward'] == 'None'], 'type': 'bar', 'name': 'None'},
                {'x': dff.health[dff['steward'] == '1or2'], 'y': dff.prop_steward[dff['steward'] == '1or2'], 'type': 'bar', 'name': '1or2'},
                {'x': dff.health[dff['steward'] == '3or4'], 'y': dff.prop_steward[dff['steward'] == '3or4'], 'type': 'bar', 'name': '3or4'},
                {'x': dff.health[dff['steward'] == '4orMore'], 'y': dff.prop_steward[dff['steward'] == '4orMore'], 'type': 'bar', 'name': '4orMore'}
            ],
            'layout': {
                'title': 'Prop. of Steward by health'
            }
        }
    return figure 


if __name__ == '__main__':
    app.run_server()

