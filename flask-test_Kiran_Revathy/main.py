from flask import Flask, render_template, redirect, url_for, request, send_file
import pandas as pd
import numpy as np
from pandas import DataFrame, read_csv
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import jinja2
plt.style.use('ggplot')
from io import BytesIO
import seaborn as sns

app = Flask(__name__)

#df = pd.read_csv("movie_metadataproject.csv")
#df["budget"] = df["budget"].fillna(0)
#df["gross"] = df["gross"].fillna(0)
#df['Profit'] = df['gross'] - df['budget']

@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template("home.html")
 
@app.route("/input", methods = ['POST','GET'])
def input():
    if request.method == 'POST':
        moviename = request.form["moviename"]
        directorname = request.form["dname"]
        actor1 = request.form["a1name"]
        actor2 = request.form["a2name"]
        actor3 = request.form["a3name"]
        genres = request.form.getlist("genre")
        language = request.form.get("lang")
        genred = concatenate_list(genres)
        iparray = [language,directorname,actor1,actor2,actor3,moviename,genred,0,0,0]
        df = pd.read_csv("movie_metadataproject.csv")
        #print(df.shape)
        df = pd.read_csv("movie_metadataproject.csv")
        df["budget"] = df["budget"].fillna(0)
        df["gross"] = df["gross"].fillna(0)
        df['Profit'] = df['gross'] - df['budget']
        df = df.drop(['aspect_ratio','movie_imdb_link','plot_keywords'],axis =1)
        df = df.dropna()
        #print(df.shape)
        df= df[df['language'] != "Telugu"]
        df= df[df['language'] != "Arabic"]
        df= df[df['language'] != "Aramaic"]
        df= df[df['language'] != "Bosnian"] 
        df= df[df['language'] != "Czech"]
        df= df[df['language'] != "Dzongkha"]
        df= df[df['language'] != "Filipino"]
        df= df[df['language'] != "Hungarian"]
        df= df[df['language'] != "Icelandic"]
        df= df[df['language'] != "Kazakh"]
        df= df[df['language'] != "Maya"]
        df= df[df['language'] != "Mongolian"]
        df= df[df['language'] != "None"]
        df= df[df['language'] != "Romanian"]
        df= df[df['language'] != "Russian"]
        df= df[df['language'] != "Swedish"]
        df= df[df['language'] != "Vietnamese"]
        df= df[df['language'] != "Zulu"]

        df_usefuldata = df[['language','director_name','actor_1_name','actor_2_name','actor_3_name','movie_title','genres','gross','budget','Profit']]
        df_usefuldata = df_usefuldata.dropna()
        df_appendedlang = df_usefuldata.append(pd.Series([iparray[0],iparray[1],iparray[2],iparray[3],iparray[4],iparray[5],iparray[6],iparray[7],iparray[8],iparray[9]], index=df_usefuldata.columns), ignore_index=True)
        #print(df_appendedlang.shape)
        df_appendedlang1 = df_appendedlang[df_appendedlang['language'] != 'None']
        df_appendedlang1 = df_appendedlang1.dropna()
        #print(df_appendedlang1.shape)
        column_values1 = df_appendedlang1["language"].unique().tolist()
        #print(column_values1)
        column_values2 = df_appendedlang1["director_name"].unique().tolist()
        df_appendedlang2 = df_appendedlang1
        #df_appendedlang3 = df_appendedlang1
        for value in column_values1:
            df_appendedlang2 = pd.concat([df_appendedlang2,pd.get_dummies(value)], axis=1)
        
        for value in column_values1:
            df_appendedlang2[value] = 0

        for value in column_values1:
            df_appendedlang2.loc[df_appendedlang2['language'] == value,value] = 1
        
        drop_cols = ['language','genres','movie_title','director_name','actor_1_name','actor_2_name','actor_3_name','gross','budget','Profit']
        for dropCol in drop_cols:
            df_appendedlang2 = df_appendedlang2.drop(dropCol,axis=1)

        df_appendedlang2 = df_appendedlang2.dropna()
        #print(df_appendedlang2)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters = 23)
        kmeans = kmeans.fit(df_appendedlang2)
        df_appendedlang2['cluster'] = kmeans.labels_

        df_appendedlang3 = pd.concat([df_appendedlang1,df_appendedlang2], axis=1, ignore_index=False)
        df_appendedlang3 = df_appendedlang3.dropna()

        df_appendedlang3 = df_appendedlang3.loc[df_appendedlang3['language'] == iparray[0]]
        print(df_appendedlang3.shape)
        
        df_appendedlang3 = df_appendedlang3.drop('cluster',axis=1)

        for value in column_values1:
            df_appendedlang3 = df_appendedlang3.drop(value,axis=1)

       # print(df_appendedlang3.shape)

        df_appendedlang4 = df_appendedlang3

        for value in column_values2:
            df_appendedlang4 = pd.concat([df_appendedlang4,pd.get_dummies(value)], axis=1)
            
        for value in column_values2:
            df_appendedlang4[value] = 0

        for value in column_values2:
            df_appendedlang4.loc[df_appendedlang4['director_name'] == value,value] = 1

        for dropCol in drop_cols:
            df_appendedlang4 = df_appendedlang4.drop(dropCol,axis=1)

       # print(df_appendedlang4.shape)

        from sklearn.cluster import KMeans
        n = 1
        if iparray[0] == "English":
            n = 200
            
        elif iparray[0] == "Spanish":
            n = 10

        elif iparray[0] == "French":
            n = 13

        elif iparray[0] == "Mandarin":
            n = 5

        elif iparray[0] == "German":
            n = 5

        elif iparray[0] == "Japanese":
            n = 5

        elif iparray[0] == "Hindi":
            n = 3

        elif iparray[0] == "Cantonese":
            n = 3
        
        elif iparray[0] == "Italian":
            n = 3

        elif iparray[0] == "Korean":
            n = 2

        elif iparray[0] == "Portuguese":
            n = 2
        
        elif iparray[0] == "Norwegian":
            n = 2

        kmeans = KMeans(n_clusters = n)
        kmeans = kmeans.fit(df_appendedlang4)
        df_appendedlang4['cluster'] = kmeans.labels_

        df_appendedlang5 = pd.concat([df_appendedlang3,df_appendedlang4], axis=1, ignore_index=False)
        df_appendedlang5 = df_appendedlang5.dropna()
        
        cnum = df_appendedlang5.loc[df_appendedlang5['director_name'] == iparray[1],'cluster']
        print(cnum.size)
        if cnum.size == 1:
            cnumb = cnum.item()
        else:
            cnumb = cnum[0]
        
        df_appendedlang5 = df_appendedlang5.loc[df_appendedlang5['cluster'] == cnumb]
        df_appendedlang5 = df_appendedlang5.drop('cluster',axis=1)

        for value in column_values2:
            df_appendedlang5 = df_appendedlang5.drop(value,axis=1)

        df_appendedlang6 = df_appendedlang5

        column_values6 = df_appendedlang6["actor_1_name"].unique().tolist()
        #column_values6
        column_values7 = df_appendedlang6["actor_2_name"].unique().tolist()
        column_values8 = df_appendedlang6["actor_3_name"].unique().tolist()
        column678 = column_values6+column_values7+column_values8
        unique_values678 =  pd.unique(column678)

       # for v in unique_values678:
       #     print(v)

        for value in unique_values678:
            df_appendedlang6 = pd.concat([df_appendedlang6,pd.get_dummies(value)], axis=1)
            df_appendedlang6[value] = 0
            df_appendedlang6.loc[df_appendedlang6['actor_1_name'] == value,value] = 1
            df_appendedlang6.loc[df_appendedlang6['actor_2_name'] == value,value] = 1
            df_appendedlang6.loc[df_appendedlang6['actor_3_name'] == value,value] = 1


        drop_cols = ['language','director_name','genres','movie_title','gross','budget','Profit','actor_1_name','actor_2_name','actor_3_name']

        for value in drop_cols:
            df_appendedlang6 = df_appendedlang6.drop(value,axis=1)

       # from sklearn.cluster import KMeans
       # kmeans = KMeans(n_clusters = 3)
       # kmeans = kmeans.fit(df_appendedlang6)
       # df_appendedlang6['cluster'] = kmeans.labels_

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters = 2)
        kmeans.fit(df_appendedlang6)    
        
        #y.labels_
        y_pred = kmeans.predict(df_appendedlang6)

        from sklearn.metrics.cluster import adjusted_rand_score

        ARI = adjusted_rand_score(kmeans.labels_,y_pred)
        print(ARI)

        from sklearn.metrics import accuracy_score
        Accuracy = accuracy_score(kmeans.labels_,y_pred,normalize=True,sample_weight=None)
        print(Accuracy)

        df_appendedlang6['cluster'] = kmeans.labels_
        df_final = pd.concat([df_appendedlang5,df_appendedlang6], axis=1, ignore_index=False)
        df_final = df_final.dropna()
        print(df_final)
        print(df_final.shape)
        print(df_final.columns.unique)
        min_budget = df_final['budget'].where(df_final['budget'].gt(0)).min(0)
        #print(df_final['budget'].unique().tolist())
        print(min_budget)
        max_budget = df_final['budget'].max()
        print(max_budget)
        mean_budget = df_final['budget'].mean()
        print(mean_budget)

        min_gross = df_final['gross'].min()
        #print(df_final['gross'].unique().tolist())
        print(min_gross)
        max_gross = df_final['gross'].max()
        print(max_gross)
        mean_gross = df_final['gross'].mean()
        print(mean_gross)

        #print(df_final['Profit'].unique().tolist())
        min_profit = df_final['Profit'].min()
        print(min_profit)
        max_profit = df_final['Profit'].max()
        print(max_profit)
        mean_profit = df_final['Profit'].mean()
        print(mean_profit)


      #  iparray1 = [language,directorname,actor1,actor2,actor3,moviename,genred,mean_gross,mean_budget,mean_profit]
       # df_usefuldata = df[['language','director_name','actor_1_name','actor_2_name','actor_3_name','movie_title','genres','gross','budget','Profit']]
       # df_revenue_data = df_final.append(pd.Series([iparray1[0],iparray1[1],iparray1[2],iparray1[3],iparray1[4],iparray1[5],iparray1[6],iparray1[7],iparray1[8],iparray1[9]], index=df_final.columns), ignore_index=True)
      #  df_revenue_data = df_revenue_data.dropna()
      #  print(df_revenue_data.shape)

        return redirect(url_for('result', min_budget= min_budget, max_budget= max_budget, min_profit= min_profit,max_profit=max_profit,max_gross= max_gross, min_gross=min_gross))

    
    return render_template('input.html')

def concatenate_list(list):
    result = ''
    for elements in list:
        if result == '':
            result = elements
        else:
            result = result + "|" + elements

    return result
#-----------------------------------------DATA VISUALIZATION--------------------------------------

def split(x):
    df = pd.read_csv("movie_metadataproject.csv")
    df["budget"] = df["budget"].fillna(0)
    df["gross"] = df["gross"].fillna(0)
    df['Profit'] = df['gross'] - df['budget']
    a = df[x].str.cat(sep = '|')
    splitdata = pd.Series(a.split('|'))
    info = splitdata.value_counts(ascending=False)
    return info
total_genre_movies = split('genres')


@app.route('/genrea/')
def visualization1():
    
    fig, ax = plt.subplots()
    total_genre_movies = split('genres')
    total_genre_movies.plot(kind= 'barh',figsize = (13,6),fontsize=12,colormap='tab20c')

    #setup the title and the labels of the plot.
    plt.title("Number of movies in each Genre",fontsize=15)
    plt.xlabel('Number Of Movies',fontsize=13)
    plt.ylabel("Genres",fontsize= 13)
    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')



@app.route('/score/')
def visualizationf1():
    df = pd.read_csv("movie_metadataproject.csv")
    df["budget"] = df["budget"].fillna(0)
    df["gross"] = df["gross"].fillna(0)
    df['Profit'] = df['gross'] - df['budget']
    fig,ax = plt.subplots()
    info = pd.DataFrame(df['imdb_score'].sort_values(ascending = False))
    info['movie_title'] = df['movie_title']
    data = list(map(str,(info['movie_title'])))

    x = list(data[:10])
    y = list(info['imdb_score'][:10])

    ax = sns.pointplot(x=y,y=x)
    #sns.set(rc={'figure.figsize':(10,20)})
    ax.set_title("Top 10 movies",fontsize = 10)
    ax.set_xlabel("IMDB Score",fontsize = 10)
    #setup the stylesheet
    sns.set_style("whitegrid")
    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')
    

@app.route('/money/')
def visualizationf2():
    df = pd.read_csv("movie_metadataproject.csv")
    df["budget"] = df["budget"].fillna(0)
    df["gross"] = df["gross"].fillna(0)
    df['Profit'] = df['gross'] - df['budget']
    fig, ax = plt.subplots()
    info = pd.DataFrame(df['Profit'].sort_values(ascending = False))
    info['movie_title'] = df['movie_title']
    data = list(map(str,(info['movie_title'])))
    x = list(data[:10])
    y = list(info['Profit'][:10])

    #make a plot usinf pointplot for top 10 profitable movies.
    ax = sns.pointplot(x=y,y=x)

    #setup the figure size
    sns.set(rc={'figure.figsize':(10,5)})
    #setup the title and labels of the plot.
    ax.set_title("Top 10 Profitable Movies",fontsize = 15)
    ax.set_xlabel("Profit",fontsize = 13)
    sns.set_style("darkgrid")
    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')

@app.route('/budget/')
def visualizationf3():
    df = pd.read_csv("movie_metadataproject.csv")
    df["budget"] = df["budget"].fillna(0)
    df["gross"] = df["gross"].fillna(0)
    df['Profit'] = df['gross'] - df['budget']
    fig, ax = plt.subplots()
    info = pd.DataFrame(df['budget'].sort_values(ascending = False))
    info['movie_title'] = df['movie_title']
    data = list(map(str,(info['movie_title'])))

    #extract the top 10 budget movies data from the list and dataframe.
    x = list(data[:10])
    y = list(info['budget'][:10])

    #plot the figure and setup the title and labels.
    ax = sns.pointplot(x=y,y=x)
    sns.set(rc={'figure.figsize':(10,5)})
    ax.set_title("Top 10 High Budget Movies",fontsize = 15)
    ax.set_xlabel("Budget",fontsize = 13)
    sns.set_style("darkgrid")
    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')

@app.route('/revenue/')
def visualizationf4():
    df = pd.read_csv("movie_metadataproject.csv")
    df["budget"] = df["budget"].fillna(0)
    df["gross"] = df["gross"].fillna(0)
    df['Profit'] = df['gross'] - df['budget']
    fig, ax = plt.subplots()
    info = pd.DataFrame(df['gross'].sort_values(ascending = False))
    info['movie_title'] = df['movie_title']
    data = list(map(str,(info['movie_title'])))

    #extract the top 10 movies with high revenue data from the list and dataframe.
    x = list(data[:10])
    y = list(info['gross'][:10])

    #make the point plot and setup the title and labels.
    ax = sns.pointplot(x=y,y=x)
    sns.set(rc={'figure.figsize':(10,5)})
    ax.set_title("Top 10 High Revenue Movies",fontsize = 15)
    ax.set_xlabel("Revenue",fontsize = 13)
    sns.set_style("darkgrid")
    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')


    
        

#-------------------------------------------DATA VISUALIZATION--------------------------------
#@app.route("/<min_budget>&emsp;&emsp;<max_budget>&emsp;&emsp;<min_profit>&emsp;&emsp;<max_profit>")
#def result(min_budget,max_budget, min_profit, max_profit):
#   return f"<p1>Minimum Budget</p1><h1>{min_budget}</h1>&emsp;<p2>Maximum budget</p2><h2>{max_budget}</h2>&emsp;<p3>Minimum profit</p3><h3>{min_profit}</h3>&emsp;<p4>Maximum profit</p4><h4>{max_profit}</h4>"


@app.route("/<min_budget>&emsp;&emsp;<max_budget>&emsp;&emsp;<min_profit>&emsp;&emsp;<max_profit>&emsp;&emsp;<min_gross>&emsp;&emsp;<max_gross>")
def result(min_budget,max_budget, min_profit, max_profit,min_gross,max_gross):
   return f"<table><tr><th>Minimum budget</th>&emsp;&emsp;<th>Maximum budget</th>&emsp;&emsp;<th>Minimum profit</th>&emsp;&emsp;<th>Maximum profit</th>&emsp;&emsp;<th>Minimum gross</th>&emsp;&emsp;<th>Maximum gross</th></tr><tr><td>{min_budget}</td>&emsp;&emsp;<td>{max_budget}</td>&emsp;&emsp;<td>{min_profit}</td>&emsp;&emsp;<td>{max_profit}&emsp;&emsp;<td>{min_gross}</td>&emsp;&emsp;<td>{max_gross}</td></tr></table>"

#@app.route('/result')
#def result(min_budget,max_budget, min_profit, max_profit):
#    return render_template('result.html', min_budget=min_budget, max_budget=max_budget, min_profit=min_profit, max_profit=max_profit)

@app.route("/news")
def news():
    if request.method == 'POST':
        
        return redirect(url_for('home'))

    
    return render_template('news.html')

@app.route("/genre")
def genre():
    if request.method == 'POST':
        
        return redirect(url_for('news'))

    
    return render_template('genre.html')

@app.route("/score")
def score():
    if request.method == 'POST':
        
        return redirect(url_for('news'))

    
    return render_template('score.html')

@app.route("/otherstats")
def otherstats():
    if request.method == 'POST':
        
        return redirect(url_for('news'))

    
    return render_template('otherstats.html')

@app.route("/budget")
def budget():
    if request.method == 'POST':
        
        return redirect(url_for('news'))

    
    return render_template('budget.html')

@app.route("/revenue")
def revenue():
    if request.method == 'POST':
        
        return redirect(url_for('news'))

    
    return render_template('revenue.html')
    
if __name__ == "__main__":
    app.run(debug=True)