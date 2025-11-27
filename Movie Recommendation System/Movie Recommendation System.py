import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv("C:/Users/krsar/Downloads/Internship - YBI Foundation/Movie Recommendation System/archive/Top_10000_Movies.csv")
df = pd.DataFrame(data)

df['content'] = df['overview'].astype(str) + ' ' + df['genre'].astype(str)

df = df.reset_index(drop=True)
indices = pd.Series(df.index, index=df['original_title']).drop_duplicates()

print("Data Preparation Complete.")

tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(df['content'])

print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

print(f"Cosine Similarity Matrix Shape: {cosine_sim.shape}")

def get_recommendations(title, cosine_sim=cosine_sim, df=df, indices=indices):
    try:
        idx = indices[title]
    except KeyError:
        return f"Error: Movie '{title}' not found in the dataset."

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:7]

    movie_indices = [i[0] for i in sim_scores]

    return df['original_title'].iloc[movie_indices]

print("\n--- Recommendations ---")

print("Recommendations for 'The Dark Knight':")
print(get_recommendations('The Dark Knight'))

print("\nRecommendations for 'Toy Story':")
print(get_recommendations('Toy Story'))

