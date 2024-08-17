import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import requests
from io import BytesIO
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


#st.set_option('deprecation.showPyplotGlobalUse', False) #do not show warning when plotting on streamlit

# Title and description of the Streamlit app
st.title("GAME RECOMMENDATION SYSTEM")
st.subheader("ABOUT PROJECT")
st.write("Welcome to Game Recommendation System! This project utilizes the Steam Game Dataset sourced from Kaggle to provide personalized game suggestions. Three methods for training the model has been implemented in this project:")
st.write("1. Leveraging scikit-learn's Nearest Neighbour library for efficient recommendation based on game features.")
st.write("2. Developing a custom Nearest Neighbour algorithm from scratch, ensuring a deeper understanding of recommendation processes.")
st.write("3. Implementing a user-personalized model that tracks individual search histories, enhancing recommendation accuracy over time.")

# Reading datasets
data=pd.read_csv("games.csv")
meta_data=pd.read_json("games_metadata.json",lines=True)

st.subheader("EXPLORATORY DATA ANALYSIS")
# Merging datasets
merge_data=data.merge(meta_data,on="app_id")


# Filtering out games with less reviews
filter_=merge_data[merge_data['user_reviews']<500]
merge_data.drop(filter_.index,inplace=True)

# Cleaning data that do not have any description
filter_data=merge_data[merge_data["description"]==""]
merge_data=merge_data.drop(filter_data.index)
merge_data.reset_index(drop=True,inplace=True)

# Function to convert three OS columns into a single column

def OS_column(row):
  array = np.array(["win", "mac", "linux"])
  filter=row['win':'linux'].values.astype(bool)
  return array[filter].tolist()

# converting three columns into a single column of OS
merge_data["OS"]=merge_data.apply(lambda row:OS_column(row),axis=1)
# Splitting description into a list of words
merge_data['description']=merge_data['description'].apply(lambda x:x.split())

# Removing spaces between words in tags
merge_data["tags"]=merge_data["tags"].apply(lambda x:[i.replace(" ","") for i in x])

# Renaming column tags to keywords
merge_data.rename(columns={"tags":"keywords"},inplace=True)

# Combining description, keywords, and OS columns into tags column

merge_data['tags']=merge_data['description']+merge_data['keywords']+merge_data['OS']

merge_data['tags']=merge_data["tags"].apply(lambda x:[i.lower() for i in x])
merge_data.reset_index(drop=True,inplace=True)
new_df=merge_data[["app_id","title","tags"]].copy()
new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))

#functions for exploratory data analysis

# Function to visualize game distribution

def game_distribution():

      fig, ax=plt.subplots(1,2,figsize=(20,10))
      merge_data.groupby('rating').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'),ax=ax[0])
      ax[0].set_ylabel("Rating",fontsize=15)
      ax[1].pie(merge_data['rating'].value_counts().tolist(),autopct='%1.1f%%',
          labels=merge_data['rating'].value_counts().keys(),
          wedgeprops={'edgecolor': 'black', 'linewidth': 0.5, 'width': 1})

      fig.suptitle("Game Distribution Overview", fontsize=20, y=1.02)


      st.pyplot()


# Function to visualize game reviews
def game_reviews():

      plt.figure(figsize=(15,10))
      sns.barplot(x='title',y='user_reviews',
                  data=merge_data[["title",'rating','user_reviews','positive_ratio']].
                  sort_values(by='user_reviews',ascending=False).head(35),
                  hue='rating',)
      plt.tick_params(axis='x',rotation=90,width=5)
      plt.xlabel("Game Titles",fontsize=10)
      plt.ylabel("No of Reviews",fontsize=10)
      plt.title("Top 30 User Reviewed Games",fontsize=16)
      st.pyplot()


# Function to visualize game prices

def game_price():
        plt.figure(figsize=(15,10))
        sns.barplot(x='title',y='price_final',data=merge_data[["title",'rating','price_final']].query('price_final>60').
                    sort_values(by='price_final',ascending=False),hue='rating')
        plt.tick_params(axis='x',rotation=90,width=5)
        plt.xlabel("Titles",fontsize=12)
        plt.ylabel("Price($)",fontsize=12)
        plt.title(" Most Expensive Games and Softwares",fontsize=16)
        st.pyplot()

# Function to visualize low-rated games

def low_rated():

        plt.figure(figsize=(20,10))
        sns.barplot(x='title',y='positive_ratio',
                    data=merge_data[['title','positive_ratio','rating']].
                    sort_values(by='positive_ratio',ascending=True).head(10),hue='rating',palette="magma")
        plt.tick_params(axis='x',rotation=90,width=15)
        plt.xlabel("Titles",fontsize=10)
        plt.ylabel("Positivity Ratio",fontsize=10)
        plt.title("Top 10 Low Rated Games",fontsize=16)
        st.pyplot()

# Create a dropdown to select the plot
selected_plot = st.selectbox('Select Plot', ['Distribution of Games', 'Game Reviews', 'Most Expensive Games', 'Low Rated Games'])
plot_button = st.button('Plot')

if plot_button:
        if selected_plot == 'Distribution of Games':
              game_distribution()
        elif selected_plot == 'Game Reviews':
              game_reviews()
        elif selected_plot == 'Most Expensive Games':
              game_price()
        elif selected_plot == 'Low Rated Games':
              low_rated()
st.subheader("Top Rated Games List")
table_button=st.button("Show")

if table_button:

  top_games=merge_data[['title','positive_ratio','user_reviews']].query('positive_ratio>80 and user_reviews>80000 ').sort_values(by='positive_ratio',ascending=False).copy()
  top_games.reset_index(drop=True,inplace=True)
  top_games.rename(columns={"title":"Game Title","positive_ratio":"Positive Ratio","user_reviews":"No of Reviews"},inplace=True)
  st.table(top_games)

#initializing stemming which is the process of reducing words to their word stem, base, or root form.

ps=PorterStemmer()


def stemming(text):
    words = nltk.word_tokenize(text)  # make a list words from the given text.
    stemmed_words = [ps.stem(word) for word in words] #apply stemming
    return " ".join(stemmed_words) #rejoing words after stemming


# Caching the expensive computation using Streamlit cache

@st.cache_data(ttl=3600)
def expensive_computation():

      new_df['tags']=new_df['tags'].apply(stemming)

      cv=CountVectorizer(max_features=3000,stop_words="english")

      vector=cv.fit_transform(new_df["tags"]).toarray()

      similarity=cosine_similarity(vector)

      return vector,similarity


# Fetching vector and similarity using Streamlit cache

vector,similarity=expensive_computation()



# Recommender functions


# Recommender based on cosine similarity
def recommender(game):
  game_list=sorted(list(enumerate(similarity[new_df[new_df['title']==game]
                                              .index[0]])),reverse=True,
                                               key=lambda x:x[1] )[1:5]
  recommendation=[(new_df.iloc[i[0]]["title"],new_df.iloc[i[0]]["app_id"]) for i in game_list]
  return recommendation


# Recommender using Nearest Neighbors

model_nn = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute')

model_nn.fit(vector)

def nn_recommender(game_):
  distances,indices=model_nn.kneighbors(vector[new_df.query('title==@game_').index[0]].reshape(1,-1))
  recommender_=[(new_df.iloc[i]["title"],new_df.iloc[i]["app_id"]) for i in indices[:,1:].reshape(4,)]
  return recommender_


# Recommender based on user preferences

def Preffered_recommendation(game_list):

  user_vector=np.zeros(6000)
  game_vector=[]
  index_vector=[]
  result_tuple=[]
  x=list(enumerate(vector))

  for i in range(len(game_list)):
    index_value=new_df[new_df['title']==game_list[i]].index[0]
    user_vector+=x[index_value][1]
    y=sorted(list(enumerate(similarity[index_value])),reverse=True,key=lambda x:x[1] )[1:11]
    for values in y:
        game_vector.append(x[values[0]][1])
        index_vector.append(values[0])



  final_user_vector=user_vector/len(game_list)
  answer=np.dot(np.array(final_user_vector),np.array(game_vector).T)

  result_tuple = sorted(list(zip(index_vector, answer)),reverse=True,key=lambda x:x[1])[:6]

  recommendation=[(new_df.iloc[j[0]]["title"],new_df.iloc[j[0]]["app_id"]) for j in result_tuple]
  return recommendation



# Function to get game header image from Steam API

def get_game_header_image(app_id):
    url = f"https://store.steampowered.com/api/appdetails?appids={app_id}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if str(app_id) in data:
            game_data = data[str(app_id)]
            if "data" in game_data:
                game_info = game_data["data"]
                if "header_image" in game_info:
                    return game_info["header_image"]

    return "NA"  # Return "NA" if header image URL not found


# Session state to store searched games

if 'games_list' not in st.session_state:
    st.session_state.games_list = []

def display_game(recommendation_fetched,num_columns):
    i=0
    columns = st.columns(num_columns)
    for game in recommendation_fetched:
                image_url = get_game_header_image(game[1])
                with columns[i]:
                    if image_url != "NA":
                      response = requests.get(image_url)
                      if response.status_code == 200:
                          img = Image.open(BytesIO(response.content))
                          st.image(img, caption=game[0], use_column_width=True)
                      else:
                          st.write("Unable to fetch image for", game[0])
                    else:
                        st.write(game[0])
                i+=1


st.subheader("Recommender")


game_name = st.selectbox("Select a game", new_df['title'])



if st.button("Search"):
    if game_name:
        st.session_state.games_list.append(game_name)

        st.markdown("**Searched Games:**")
        for game1 in st.session_state.games_list:
            st.write(game1)


        st.markdown("**Recommended Games**")

        recommendations_1 = recommender(game_name)
        recommendations_2 = nn_recommender(game_name)
        recommendations_3 = Preffered_recommendation(st.session_state.games_list)

        # Display images and titles for Nearest Neighbors recommendations
        st.subheader("Unsupervised Nearest Neighbors Algorithm")

        display_game(recommendations_2,4)



        # Display images and titles for Similarity Based recommendations

        st.subheader("Similarity Based Recommendation")
        display_game(recommendations_1,4)




        # Display images and titles for User Personalized Algorithm recommendations
        st.subheader("User Personalized Algorithm")

        display_game(recommendations_3,6)











