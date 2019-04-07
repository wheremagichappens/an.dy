
# coding: utf-8

# data608 - Module 5
# 
# By: Sang Yoon (Andy) Hwang
# 
# Date: 2019-04-07

# In[ ]:


from flask import Flask, jsonify, send_from_directory, render_template
import pandas as pd


app = Flask(__name__)


# This is an API meant to serve some trees data
@app.route('/trees/<string:boroname>/<string:health>')
def return_hpi_data(boroname, health):

    # Read in raw data
    soql_url = ('https://data.cityofnewyork.us/resource/nwxe-4ae8.json?' +        '$select=count(tree_id),boroname,spc_common, health,status' +         '&$where=health!=\'NaN\'' +        '&$group=boroname,health,status,spc_common').replace(' ', '%20')
    
    raw_data = pd.read_json(soql_url)

    # Filter based on boroname and health
    filtered_data = raw_data.loc[(raw_data['boroname'] == boroname) & (raw_data['health'] == health)]

    # Build our json, then return it with jsonify
    filtered_data_json = {
        'spc_common': filtered_data['spc_common'].tolist(),
        'count_tree_id': filtered_data['count_tree_id'].tolist(),
        'status': filtered_data['status'].tolist()
    }

    return jsonify(filtered_data_json)


# This routing allows us to view index.html
@app.route('/')
def index():
    return render_template('index.html')


# This routing allows us to load local Javascript
@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('js', path)


if __name__ == '__main__':
    app.run()

