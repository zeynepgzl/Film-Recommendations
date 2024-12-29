# Movie Recommendation System

## Overview
This project is a **Movie Recommendation System** built using **Streamlit** and the **TMDB (The Movie Database)** API. The system leverages **natural language processing (NLP)** techniques and a **content-based filtering** approach to recommend movies based on their overview descriptions. It also provides additional movie details such as budget, revenue, genres, and top cast members.

---

## Features
- Search for movies by title.
- Get detailed information about movies, including:
  - Release date
  - Genres
  - Budget
  - Revenue
  - Overview
  - Top 5 cast members
- View movie posters retrieved via the TMDB API.
- Receive content-based movie recommendations based on an input movie's overview.
- Fuzzy matching to handle partial or misspelled movie titles.

---

## Dataset
The project uses the following datasets:
1. **tmdb_5000_movies.csv**: Contains information about 5,000 movies, including titles, overviews, budgets, and revenues.
2. **tmdb_5000_credits.csv**: Contains cast and crew details for the same set of movies.

---

## Requirements
### Python Libraries
Make sure you have the following Python libraries installed:
- `pandas`
- `numpy`
- `fuzzywuzzy`
- `requests`
- `Pillow`
- `streamlit`
- `scikit-learn`

To install them, you can use the following command:
```bash
pip install pandas numpy fuzzywuzzy requests Pillow streamlit scikit-learn
```

---

## How to Run
### 1. Clone the Repository
Clone this repository to your local machine:
```bash
git clone <repository_url>
cd <repository_folder>
```

### 2. Add Your TMDB API Key
Replace the `your_api_key` placeholder in the code with your actual TMDB API key. You can obtain your API key by creating an account at [TMDB](https://www.themoviedb.org/) and generating an API key under the developer settings.

### 3. Prepare the Dataset
Ensure that the following CSV files are present in the same directory as the script:
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

### 4. Run the Application
Run the Streamlit app using the following command:
```bash
streamlit run app.py
```

### 5. Access the Application
After running the above command, the application will be accessible in your browser at `http://localhost:8501`.

---

## How It Works
### Content-Based Filtering
- **TF-IDF Vectorization**: The `overview` field of each movie is vectorized using TF-IDF to capture the importance of terms.
- **Sigmoid Kernel Similarity**: A sigmoid kernel is used to compute the similarity scores between movies based on their vectorized overviews.
- **Recommendation**: Movies with the highest similarity scores are recommended to the user.

### Fuzzy Matching
The `fuzzywuzzy` library is used to match user input with movie titles in the dataset, improving usability for partial or misspelled inputs.

### TMDB API Integration
- **Movie Posters**: Retrieved dynamically from the TMDB API using the movie's title.
- **Additional Movie Details**: Detailed information about movies (e.g., cast, genres) is fetched using TMDB's endpoints.

---

## Example Usage
1. Enter a movie title (e.g., `Inception`) in the sidebar.
2. View details about the movie, including:
   - Poster
   - Release date, genres, budget, and more
3. Get personalized movie recommendations based on the selected movie.
4. Browse through recommended movies with their posters displayed.

---

## Screenshots
### Main Interface
![Movie Poster](https://github.com/lenovo/proje/film_recommendaions/main_interface.png)


### Movie Recommendations
![Recommendations](screenshots/recommendations.png)

---

## Improvements & Future Work
- Add collaborative filtering for more robust recommendations.
- Include user ratings in the recommendation engine.
- Implement caching for API requests to improve performance.
- Expand the UI with more interactive elements.

---

## Acknowledgments
- The Movie Database (TMDB) for providing the datasets and API.
- `scikit-learn` and `fuzzywuzzy` for their excellent libraries.
- Streamlit for making interactive applications easy to build.

