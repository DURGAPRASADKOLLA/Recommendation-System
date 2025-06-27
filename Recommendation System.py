# Step 1: Import Libraries

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import random

# Step 2: Load Movie Data
movies_df = pd.read_csv("Indian_movies.csv")
movie_list = movies_df["Movie Names"].dropna().unique().tolist()

# Step 3: Generate Synthetic User Ratings
random.seed(42)
user_ids = [f"user_{i}" for i in range(1, 21)]
ratings = []
for user in user_ids:
    rated_movies = random.sample(movie_list, k=random.randint(10, 20))
    for movie in rated_movies:
        rating = round(random.uniform(6, 10), 1)
        ratings.append([user, movie, rating])
ratings_df = pd.DataFrame(ratings, columns=["user_id", "movie_name", "rating"])
ratings_df.head()

# Step 4: Create User-Item Matrix
user_item_matrix = ratings_df.pivot_table(index='user_id', columns='movie_name', values='rating').fillna(0)
user_item_matrix.head()

# Step 5: Apply SVD (Matrix Factorization)
from numpy.linalg import svd
# Convert to NumPy array
R = user_item_matrix.values
user_means = np.mean(R, axis=1).reshape(-1, 1)
R_demeaned = R - user_means
# SVD
U, sigma, Vt = svd(R_demeaned, full_matrices=False)
sigma_diag_matrix = np.diag(sigma)
# Reconstruct ratings with top k latent factors
k = 15  # You can tune this
R_pred = np.dot(np.dot(U[:, :k], sigma_diag_matrix[:k, :k]), Vt[:k, :]) + user_means

# Step 6: Convert Predictions to DataFrame
pred_df = pd.DataFrame(R_pred, index=user_item_matrix.index, columns=user_item_matrix.columns)
pred_df.head()

# Step 7: Evaluate with RMSE (only where actual ratings exist)
from sklearn.metrics import mean_squared_error
from math import sqrt
actual = []
predicted = []
for i in range(user_item_matrix.shape[0]):
    for j in range(user_item_matrix.shape[1]):
        true_rating = user_item_matrix.iloc[i, j]
        if true_rating > 0:  # Only consider non-zero ratings
            pred_rating = pred_df.iloc[i, j]
            actual.append(true_rating)
            predicted.append(pred_rating)
rmse = sqrt(mean_squared_error(actual, predicted))
print(f"RMSE on known ratings: {rmse:.4f}")

# Step 8: Recommend Top 5 Movies for Each User
def get_top_recommendations(user_id, preds_df, original_df, n=5):
    user_row = preds_df.loc[user_id]
    already_rated = original_df.loc[user_id]
    recommendations = user_row[already_rated == 0].sort_values(ascending=False).head(n)
    return recommendations
# Example: Recommend for user_5
top_movies = get_top_recommendations('user_5', pred_df, user_item_matrix)
print("Top recommendations for user_5:")
print(top_movies)