# Movie-Recommendation-System

Movie Recommendation System based on K-nearest neighbours accelerated using GPU  
Dataset : https://grouplens.org/datasets/movielens/100k/  
This data set consists of:  
	* 100,000 ratings (1-5) from 943 users on 1682 movies.   
	* Each user has rated at least 20 movies.   

## KNN Algorithm  
Step-1: Select the number K of the neighbors  
Step-2: Calculate the **Euclidean distance** of K number of neighbors  
Step-3: Take the K nearest neighbors as per the calculated Euclidean distance.  
Step-4: Among these k neighbors, count the number of the data points in each category.  
Step-5: Assign the new data points to that category for which the number of the neighbor is maximum.  
Step-6: Our model is ready  

## Main Approach    

1. Populate movie_data[n_movies * movie_size] with each movie its corresponding genres.    
2. Populate movie_ratings & user_ratings.  
3. Create n * n distance matrix using  grid stride loop method where each thread calculates eucledian distance between 2 movies.  
4. Convert distance to similiarity coefficients.  
5. Populate user_ratings_arr which contains (user_id ,rating ,movie) grouped by user_id.   
6. Populate user_ranges containing ranges of indices (start, end) of movies rated by each user.    
7. Create knn_predictions[pred_users * pred_movies]  
8. For thread with user u and movie m , create topk : all movies associated with u sorted(selection sort) in decreasing order of their eucledian distance  from m.   
9. Computed weighted average of all topk movies to predict rating of give movie,user pair.  

For a more detailed explanation refer : https://docs.google.com/document/d/1jQ3OVbBUyj-LM5-O5-lp4Nt8Fe_2u9iyhORpKxSghyc/edit?usp=sharing
