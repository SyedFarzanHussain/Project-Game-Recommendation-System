# Game Recommendation System

Welcome to the Game Recommendation System! This project provides personalized game suggestions using the Steam Game Dataset, which is sourced from Kaggle. The system is designed to recommend games to users based on their preferences and search history. The project includes three distinct approaches for generating recommendations:

## Project Overview

The Game Recommendation System leverages various methodologies to deliver personalized game recommendations:

1. **Scikit-Learn Nearest Neighbour Algorithm**:
   - Utilizes scikit-learn’s Nearest Neighbour library to efficiently recommend games based on features.
   - This approach computes the similarity between games using feature vectors, allowing for quick and accurate recommendations.

2. **Custom Nearest Neighbour Algorithm**:
   - Implements a custom Nearest Neighbour algorithm from scratch.
   - This approach helps in understanding the underlying mechanics of recommendation systems and provides insights into how game similarity is calculated without relying on external libraries.

3. **User-Personalized Model**:
   - Develops a personalized recommendation model that adapts to individual users' search histories.
   - Enhances recommendation accuracy by considering user preferences and historical interactions, making the suggestions more relevant over time.

## Dataset

The recommendations are based on the [Steam Game Dataset](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam) available on Kaggle. This dataset includes various game attributes such as genre, tags, and user ratings.

