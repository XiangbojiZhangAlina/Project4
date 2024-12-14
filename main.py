import pandas as pd
import numpy as np
import streamlit as st


@st.cache_resource
def load_data():
    """Load pickle files from local directory"""
    try:
        S = pd.read_pickle('processed_similarity_matrix.pkl')
        popularity_ranking = pd.read_pickle('movie_popularity_ranking.pkl')
        return S, popularity_ranking
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None


@st.cache_data
def load_movie_data():
    base_url = "https://liangfgithub.github.io/MovieData/"
    ratings = pd.read_csv(base_url + 'ratings.dat?raw=true',
                          sep='::',
                          engine='python',
                          header=None,
                          names=['UserID', 'MovieID', 'Rating', 'Timestamp'])

    movies = pd.read_csv(base_url + 'movies.dat?raw=true',
                         sep='::',
                         engine='python',
                         encoding='ISO-8859-1',
                         header=None,
                         names=['MovieID', 'Title', 'Genres'])

    rating_matrix = ratings.pivot(index='UserID',
                                  columns='MovieID',
                                  values='Rating')

    # Add image URLs
    movies['image_url'] = movies['MovieID'].apply(
        lambda x: f"https://liangfgithub.github.io/MovieImages/{x}.jpg?raw=true"
    )

    return ratings, movies, rating_matrix


def myIBCF(newuser):
    """Generate movie recommendations for a new user based on their ratings"""
    # Get the saved similarity matrix from loaded data
    S, popularity_ranking = load_data()

    # Convert newuser to pandas Series for easier handling
    user_ratings = pd.Series(newuser)
    rated_movies = user_ratings.dropna()

    # Get predictions for all unrated movies
    predictions = {}

    for movie_id in S.index:
        # Skip if movie was already rated by user
        if pd.isna(user_ratings.get(movie_id, np.nan)):
            # Get similarities between current movie and rated movies
            similarities = S.loc[movie_id, rated_movies.index].dropna()

            if len(similarities) > 0:
                # Get corresponding ratings
                ratings = rated_movies[similarities.index]

                # Calculate weighted average
                numerator = (similarities * ratings).sum()
                denominator = similarities.sum()

                if denominator != 0:
                    predictions[movie_id] = numerator / denominator

    # Convert predictions to DataFrame
    pred_df = pd.Series(predictions)

    # If we have 10 or more predictions, return top 10
    if len(pred_df) >= 10:
        top_movies = pred_df.nlargest(10)
    else:
        # Get MovieIDs in order of popularity
        popular_movie_ids = popularity_ranking['MovieID'].values

        # Filter out movies that user has rated or we have predictions for
        available_movies = []
        for movie_id in popular_movie_ids:
            if (movie_id not in rated_movies.index) and (movie_id not in pred_df.index):
                available_movies.append(movie_id)
            if len(available_movies) >= (10 - len(pred_df)):
                break

        # Create series for popular movies with 0 as predicted rating
        popular_movies = pd.Series(0, index=available_movies)

        # Combine predicted and popular movies
        top_movies = pd.concat([pred_df, popular_movies])

    # Format output with 'm' prefix
    return ['m' + str(int(movie_id)) for movie_id in top_movies.index[:10]]

def display_movie_card(title, genres, image_url, rating_widget=None):
    # Change column ratio from [1, 3] to [2, 5] to give more space for the image
    col1, col2 = st.columns([2, 2])
    with col1:
        try:
            # Increase image width from 100 to 120 pixels
            st.image(image_url, width=120)
        except:
            st.image("https://via.placeholder.com/120x180?text=No+Image", width=120)
    with col2:
        st.write(f"**{title}**")
        st.write(f"*{genres}*")
        if rating_widget is not None:
            rating_widget()


# Add a function to handle clearing ratings
def clear_ratings():
    if 'user_ratings' in st.session_state:
        st.session_state.user_ratings = pd.Series(np.nan, index=st.session_state.user_ratings.index)
    # Clear all rating selectbox values
    for key in list(st.session_state.keys()):
        if key.startswith('rating_'):
            del st.session_state[key]
    st.rerun()  # Use st.rerun() instead of experimental_rerun()

# Streamlit app
st.title('Movie Recommendation System')

# Load data
ratings, movies, rating_matrix = load_movie_data()
S, popularity_ranking = load_data()

# Load data and setup
if S is not None and popularity_ranking is not None:
    if 'user_ratings' not in st.session_state:
        st.session_state.user_ratings = pd.Series(np.nan, index=S.index)

    # Filter movies first
    movies_in_S = movies[movies['MovieID'].isin(S.index)]

    # Add radio button for view mode
    view_mode = st.radio(
        "Select view mode:",
        ["Top 50 Popular Movies", "Search All Movies"]
    )

    # Search box (only shown for "Search All Movies" mode)
    if view_mode == "Search All Movies":
        search_term = st.text_input('Search for movies:', '')

    # Filter movies based on view mode
    if view_mode == "Top 50 Popular Movies":
        # Get top 50 movies from popularity ranking
        top_50_ids = popularity_ranking['MovieID'].head(50)
        filtered_movies = movies_in_S[movies_in_S['MovieID'].isin(top_50_ids)]
    else:  # Search All Movies mode
        if 'search_term' in locals() and search_term:
            filtered_movies = movies_in_S[movies_in_S['Title'].str.contains(search_term, case=False)]
        else:
            filtered_movies = movies_in_S.head(50)  # Show first 50 if no search term
            st.info('Enter a search term to find specific movies')

    # Add buttons at the top
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Get Recommendations', type='primary'):
            if st.session_state.user_ratings.isna().all():
                st.warning('Please rate at least one movie first!')
            else:
                recommendations = myIBCF(st.session_state.user_ratings)

                st.header('Your Recommendations:')
                for i, movie_id in enumerate(recommendations, 1):
                    movie_id_num = int(movie_id[1:])
                    movie_row = movies[movies['MovieID'] == movie_id_num].iloc[0]

                    st.write(f"Rank {i}")
                    display_movie_card(movie_row['Title'],
                                       movie_row['Genres'],
                                       movie_row['image_url'])
                    st.markdown("---")

    with col2:
        if st.button('Clear All Ratings'):
            clear_ratings()

    st.header('Rate Movies')

    # Display filtered movies
    for idx, row in filtered_movies.iterrows():
        movie_id = row['MovieID']
        rating_key = f"rating_{movie_id}"

        # Get current rating if exists
        current_rating = st.session_state.user_ratings.get(movie_id, np.nan)
        current_rating = int(current_rating) if not np.isnan(current_rating) else None

        # Create rating widget
        def make_rating_widget(movie_id=movie_id):
            rating = st.selectbox('Rating:',
                                  options=[None, 1, 2, 3, 4, 5],
                                  key=f"rating_{movie_id}",
                                  index=0 if current_rating is None else current_rating)
            if rating is not None:
                st.session_state.user_ratings[movie_id] = rating

        # Display movie card
        display_movie_card(row['Title'],
                           row['Genres'],
                           row['image_url'],
                           lambda movie_id=movie_id: make_rating_widget(movie_id))
        st.markdown("---")

    # Add statistics to sidebar
    st.sidebar.header('Statistics')
    total_ratings = (~st.session_state.user_ratings.isna()).sum()
    st.sidebar.write(f"Movies you've rated: {total_ratings}")
    if total_ratings > 0:
        avg_rating = st.session_state.user_ratings.mean()
        st.sidebar.write(f"Your average rating: {avg_rating:.1f}")

else:
    st.error(
        'Unable to load required data files. Please ensure the pickle files are in the same directory as the application.')

# Add explanation in sidebar
st.sidebar.header('About')
st.sidebar.write(
    'This movie recommendation system uses Item-Based Collaborative Filtering to suggest movies based on your ratings.')
st.sidebar.write('To use:')
st.sidebar.write('1. Rate some movies (1-5 stars)')
st.sidebar.write('2. Click "Get Recommendations"')
st.sidebar.write('3. Explore your personalized suggestions!')
