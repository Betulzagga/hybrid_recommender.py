
"""
hybrid_recommender.py

Hybrid Recommender System using User-Based and Item-Based Collaborative Filtering

Author: Betul Zagga
Email: betulzagga@gmail.com

This module builds a hybrid movie recommendation system using MovieLens dataset.
It combines user-based and item-based collaborative filtering to provide top movie suggestions
for a target user based on their ratings and similar users' behavior.
"""

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)


def load_movie_data(movie_path, rating_path):
    movie = pd.read_csv(movie_path)
    rating = pd.read_csv(rating_path)
    df = movie.merge(rating, how="left", on="movieId")
    return df


def create_user_movie_df(df, min_rating_count=1000):
    comment_counts = df["title"].value_counts()
    common_movies = df[~df["title"].isin(comment_counts[comment_counts <= min_rating_count].index)]
    user_movie_df = common_movies.pivot_table(index="userId", columns="title", values="rating")
    return user_movie_df


def get_similar_users(user_movie_df, target_user_id, min_similarity_ratio=0.6):
    random_user_df = user_movie_df.loc[[target_user_id]]
    movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
    movies_watched_df = user_movie_df[movies_watched]

    user_movie_count = movies_watched_df.T.notnull().sum().reset_index()
    user_movie_count.columns = ["userId", "movie_count"]
    min_movies = int(len(movies_watched) * min_similarity_ratio)

    users_same_movies = user_movie_count[user_movie_count["movie_count"] > min_movies]["userId"]
    return users_same_movies.tolist(), movies_watched_df.loc[users_same_movies]


def get_top_users_corr(target_user_id, movies_watched_df_filtered, threshold=0.65):
    corr_df = movies_watched_df_filtered.T.corr().unstack().sort_values()
    corr_df = pd.DataFrame(corr_df, columns=["corr"]).reset_index()
    corr_df.columns = ["user_id_1", "user_id_2", "corr"]

    top_users = corr_df[(corr_df["user_id_1"] == target_user_id) & (corr_df["corr"] >= threshold)]
    top_users = top_users.sort_values(by='corr', ascending=False)[["user_id_2", "corr"]]
    top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
    return top_users


def user_based_recommendation(target_user_id, user_movie_df, rating_df, movie_df):
    users_same_movies, filtered_df = get_similar_users(user_movie_df, target_user_id)
    top_users = get_top_users_corr(target_user_id, filtered_df)

    top_users_ratings = top_users.merge(rating_df, on="userId")
    top_users_ratings = top_users_ratings[top_users_ratings["userId"] != target_user_id]
    top_users_ratings["weighted_rating"] = top_users_ratings["corr"] * top_users_ratings["rating"]

    recommendation_df = top_users_ratings.groupby("movieId").agg({"weighted_rating": "mean"}).reset_index()
    recommendation_df = recommendation_df[recommendation_df["weighted_rating"] > 3.5]
    movies_to_recommend = recommendation_df.sort_values("weighted_rating", ascending=False)

    top_titles = movie_df.merge(movies_to_recommend, on="movieId")[["title"]].head(5)["title"].tolist()
    return top_titles


def item_based_recommendation(user_id, rating_df, movie_df, user_movie_df):
    last_favorite_movie_id = rating_df[(rating_df["userId"] == user_id) & (rating_df["rating"] == 5.0)]                                   .sort_values(by="timestamp", ascending=False).iloc[0]["movieId"]

    last_favorite_title = movie_df[movie_df["movieId"] == last_favorite_movie_id]["title"].values[0]
    target_movie_ratings = user_movie_df[last_favorite_title]

    correlation_series = user_movie_df.corrwith(target_movie_ratings).sort_values(ascending=False)
    top_movies = correlation_series[1:6].index.tolist()
    return top_movies


if __name__ == "__main__":
    movie_path = "datasets/movie_lens_dataset/movie.csv"
    rating_path = "datasets/movie_lens_dataset/rating.csv"

    df = load_movie_data(movie_path, rating_path)
    user_movie_df = create_user_movie_df(df)
    rating_df = pd.read_csv(rating_path)
    movie_df = pd.read_csv(movie_path)

    user_id = 108170

    print("🎯 User-Based Recommendations:")
    for title in user_based_recommendation(user_id, user_movie_df, rating_df, movie_df):
        print(" -", title)

    print("\n🎯 Item-Based Recommendations:")
    for title in item_based_recommendation(user_id, rating_df, movie_df, user_movie_df):
        print(" -", title)
