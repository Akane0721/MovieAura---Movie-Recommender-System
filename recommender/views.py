from django.http import JsonResponse
import pandas as pd
from django.shortcuts import render, redirect
from django.contrib import messages
import logging
#from . import submodule_01_Image_Based
from . import submodule_02_Content_Based
#from . import submodule_03_Graph_Based
#from . import submodule_04_Transformer

def get_recommendations(movie_sequence):
    print("Receive: ", movie_sequence)
    path = 'recommender/Datasets/Movies_Merged.csv'
    movies_data = pd.read_csv(path)
    movie_titles = movies_data["Title"]
    try:
        scores = submodule_02_Content_Based.get_movie_scores(movie_sequence, path=path)
    except Exception as e:
        raise e
    
    movie_recommendations = list(zip(movie_titles, scores))
    movie_recommendations = [movie for movie in movie_recommendations if movie[0] not in movie_sequence]
    top_movies = sorted(movie_recommendations, key=lambda x: x[1], reverse=True)[:5]
    
    return top_movies

logger = logging.getLogger(__name__)

def index(request):
    if request.method == 'POST':
        movie_sequence = request.POST.getlist('movies[]')[0].split(',')
        if isinstance(movie_sequence, list) and len(movie_sequence) > 0:
            formatted_movies = ', '.join(movie_sequence[:-1]) + ('' if len(movie_sequence) == 1 else f" and {movie_sequence[-1]}")
        else:
            formatted_movies = 'Unknown Sequence'

        request.session['sequence'] = formatted_movies
        logger.debug(f"Received movie sequence: {movie_sequence}")
        
        if not movie_sequence:
            messages.error(request, "Please add at least one movie to the queue.")
            logger.warning("User tried to submit empty movie queue.")
            return redirect('index')
        
        try:
            recommendations = get_recommendations(movie_sequence)
            logger.debug(f"Generated recommendations: {recommendations}")
        except Exception as e:
            logger.error(f"Error generated: {e}")
            messages.error(request, f"An error occurred while processing your request:\n{e}")
            return redirect('index')
        
        if not recommendations:
            messages.error(request, "No recommendations found for the provided movies.")
            logger.info("No recommendations found for the given movies.")
            return redirect('index')
        
        request.session['recommendations'] = recommendations
        return redirect('results')
    else:
        return render(request, 'recommender/index.html')

def results(request):
    recommendations = request.session.get('recommendations', [])
    sequence = request.session.get('sequence',[])
    print("result: ", recommendations)
    total = sum([r[1] for r in recommendations])
    recommendations = [
        {'title': rec[0], 'probability': rec[1] / total * 100} for rec in recommendations
    ]
    return render(request, 'recommender/results.html', {'recommendations': recommendations, 'sequence': sequence})


