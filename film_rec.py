import pandas as pd
import numpy as np
from fuzzywuzzy import process
import requests
from PIL import Image
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

# API anahtarını doğrudan koda ekliyoruz
api_key = "0050cc5e662a24abfd8fa871587310f1"  # Kendi API anahtarınızı buraya yazın

# Veri setlerini yükleme
credits = pd.read_csv("tmdb_5000_credits.csv")
movies = pd.read_csv("tmdb_5000_movies.csv")

# Veri setlerini birleştirme
credits_column_renamed = credits.rename(index=str, columns={"movie_id": "id"})
movies_merge = movies.merge(credits_column_renamed, on='id')

# Gereksiz sütunları temizleme
movies_cleaned = movies_merge.drop(columns=['homepage', 'title_x', 'title_y', 'status', 'production_countries'])

# NaN (boş) değerlerini boş string ile doldur
movies_cleaned['overview'] = movies_cleaned['overview'].fillna('')

# FuzzyWuzzy kullanarak en yakın eşleşmeyi bulma fonksiyonu
def fuzzy_match(title, choices, limit=3):
    results = process.extract(title, choices, limit=limit)
    return [result[0] for result in results]

# Film posterini almak için fonksiyon
def get_poster(movie_title):
    base_url = "https://api.themoviedb.org/3"
    search_url = f"{base_url}/search/movie?api_key={api_key}&query={movie_title}"
    response = requests.get(search_url).json()
    try:
        poster_path = response['results'][0]['poster_path']
        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
        img = Image.open(requests.get(poster_url, stream=True).raw)
        return img
    except IndexError:
        return None  # Poster bulunamazsa None döndürür

# Film bilgilerini almak için fonksiyon
def get_movie_details(movie_title):
    base_url = "https://api.themoviedb.org/3"
    search_url = f"{base_url}/search/movie?api_key={api_key}&query={movie_title}"
    response = requests.get(search_url).json()
    try:
        movie_id = response['results'][0]['id']
        details_url = f"{base_url}/movie/{movie_id}?api_key={api_key}"
        details_response = requests.get(details_url).json()
        
        # Film bilgilerini çıkartıyoruz
        title = details_response.get('original_title', 'N/A')
        overview = details_response.get('overview', 'No overview available.')
        release_date = details_response.get('release_date', 'N/A')
        budget = details_response.get('budget', 'N/A')
        revenue = details_response.get('revenue', 'N/A')
        genres = ", ".join([genre['name'] for genre in details_response.get('genres', [])])
        
        # Credits API'den cast bilgisini alıyoruz
        cast_url = f"{base_url}/movie/{movie_id}/credits?api_key={api_key}"
        cast_response = requests.get(cast_url).json()
        cast = ", ".join([actor['name'] for actor in cast_response.get('cast', [])[:5]])  # İlk 5 oyuncuyu al
        
        return {
            "title": title,
            "overview": overview,
            "release_date": release_date,
            "budget": budget,
            "revenue": revenue,
            "genres": genres,
            "cast": cast
        }
    except IndexError:
        return None

# Film önerilerini almak için fonksiyon
def give_recommendations(title, sig, movies_cleaned, indices):
    idx = indices.get(title)
    if idx is None:
        return pd.Series([])  # Eğer film bulunmazsa boş döndür
    
    sig_scores = list(enumerate(sig[idx]))  # Burada sig[idx] kullanarak score'ları elde ediyoruz
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    sig_scores = sig_scores[1:11]  # İlk öneriyi hariç tutarak en iyi 10 öneriyi al
    movie_indices = [i[0] for i in sig_scores]
    return movies_cleaned['original_title'].iloc[movie_indices]

# Streamlit arayüzü için ayarlar
st.title("Movie Recommendation System")
st.sidebar.header("Search for a Movie")
movie_title_input = st.sidebar.text_input("Enter a Movie Title:")

# TF-IDF ve Sigmoid Kernel hesaplaması
tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 3), stop_words='english')
tfv_matrix = tfv.fit_transform(movies_cleaned['overview'])

sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

# İndekslerin ters eşleşmesi
indices = pd.Series(movies_cleaned.index, index=movies_cleaned['original_title']).drop_duplicates()

# Kullanıcıdan film ismi alıp öneri yapma
if movie_title_input:
    # Fuzzy match ile en yakın eşleşmeleri alıyoruz
    matched_movies = fuzzy_match(movie_title_input, movies_cleaned['original_title'], limit=3)

    if matched_movies:
        # Film başlığına ait posterin gösterilmesi
        poster = get_poster(matched_movies[0]) if matched_movies else None
        if poster:
            st.sidebar.image(poster, caption=matched_movies[0], width=150)  # Poster küçük boyutta, sidebar altında

        # Film bilgilerini gösterme butonu
        if st.button("Show Movie Details"):
            movie_details = get_movie_details(matched_movies[0])
            if movie_details:
                st.write(f"### {movie_details['title']}")
                st.write(f"**Release Date:** {movie_details['release_date']}")
                st.write(f"**Genres:** {movie_details['genres']}")
                st.write(f"**Budget:** ${movie_details['budget']}")
                st.write(f"**Revenue:** ${movie_details['revenue']}")
                st.write(f"**Overview:** {movie_details['overview']}")
                st.write(f"**Top Cast:** {movie_details['cast']}")
            else:
                st.write("Sorry, we couldn't fetch details for this movie.")
        
        # İlk eşleşen başlıkla öneri yapılır
        recommended_movies = give_recommendations(matched_movies[0], sig, movies_cleaned, indices)

        if not recommended_movies.empty:
            st.write(f"Recommended Movies for '{matched_movies[0]}':")
            for movie in recommended_movies:
                st.write(movie)
                poster = get_poster(movie)
                if poster:
                    st.image(poster, caption=movie, use_container_width=True)
                else:
                    st.write("No poster found.")
        else:
            st.write("Sorry, we couldn't find any recommendations.")
    else:
        st.write("Sorry, we couldn't find any matching movies.")
