#include <math.h>
#include <time.h>
#include <bits/stdc++.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>



// Structure to handle sorting in place
struct sort_struct
{
    int movie;
    float weight;
    float rating;
};
bool euclidean_compare(sort_struct lhs, sort_struct rhs) {return lhs.weight < rhs.weight; }


// Computes the average rating for a given movie movie
float compute_mean(std::unordered_map<int, int> &single_movie_ratings)
{
    float m = 0.0;
    int num = 0;
    for (auto it : single_movie_ratings)
    {
        m += it.second;
        num += 1;
    }
    m = m / (float) num;
    return m;
}

// Computes the standard deviation of a given movie
// std(X) = sqrt(E[(X - ux)^2])
float compute_std_dev(std::unordered_map<int, int> &single_movie_ratings, float &m)
{
    float std_dev = 0.0;
    int num = 0;
    for (auto it : single_movie_ratings)
    {
        std_dev += pow(it.second - m, 2);
        num += 1;
    }
    std_dev = sqrt(std_dev / (float) num);
    return std_dev;
}


// Distance metric that takes the euclidean distance between
// each pair of movies based on the genres that each movie falls under
void euclidean_distances(unsigned int **movie_data, const int &n_movies,
    const int &movie_size, float **movie_distances)
{
    *movie_distances = new float[n_movies * n_movies];
    float dist;
    for (int i = 0; i < n_movies; i++)
    {
       for (int j = i; j < n_movies; j++)
       {
           if (i == j) {
               (*movie_distances)[i * n_movies + j] = -1.0;
           } else {
               // Get the squared difference
               dist = 0;
               for (int k = 0; k < movie_size; k++)
               {
                   dist += pow(float((*movie_data)[i * movie_size + k])
                       - float((*movie_data)[j * movie_size + k]), 2);
               }
               (*movie_distances)[i * n_movies + j] = dist;
               (*movie_distances)[j * n_movies + i] = dist;
           }
       }
    }
}

// For a given movie, user pair, returns the KNN prediction
float predict(int &user, int &movie, int &n_movies, int &k_val,
    float ** movie_distances,
    std::unordered_map<int, int> &u_ratings)
{

    // Find the k nearest neighboring movie ratings
    // that the same user has rated
    // and take an average to return the prediction
    sort_struct *mu_row = new sort_struct[u_ratings.size()];
    sort_struct aStruct;

    int i = 0;
    // Look at all other movies that current user has rated
    // And create a sortable structure of movie, weight (metric), and rating
    for (auto movie_rating : u_ratings)
    {
        aStruct.movie = movie_rating.first;
        aStruct.weight = (*movie_distances)[movie * n_movies + aStruct.movie];
        aStruct.weight = 1 / (aStruct.weight + 1);
        aStruct.rating = movie_rating.second;
        mu_row[i] = aStruct;
        i++;
    }


    std::sort(mu_row, mu_row + u_ratings.size(), euclidean_compare);

    // Return a weighted average of the k-nearest neighbors
    float weight_sum = 0.0;
    float result = 0.0;
    for (int j = 0; j < k_val; j++)
    {
        result += mu_row[j].weight * mu_row[j].rating;
        weight_sum += mu_row[j].weight;
    }
    if (weight_sum == 0.0)
        return 0.0;
    result = result / weight_sum;
    //std::cout << "User " << user << " Movie " << movie << ": " << result << std::endl;
    return result;
}


// KNN with an input distance metric
void knn(std::vector<std::unordered_map<int, int>> &user_ratings,
    float **movie_distances, int &k_val, int &n_movies, int &n_users,
    float **knn_predictions)
{
    // Take a sample of about 100k, movie-user pairs as a test set
    int pred_movies = 1500;
    int pred_users = 900;

    //*knn_predictions = new float[n_movies * k_val];
    *knn_predictions = new float[pred_users * pred_movies];
    int percent = 0;
    for (int i = 0; i < pred_users; i++)
    {
        std::unordered_map<int, int> u_ratings = user_ratings[i];

        // Make sure user has enough ratings to be at least k
        if (u_ratings.size() >= k_val) {
            for (int j = 0; j < pred_movies; j++)
            {
                (*knn_predictions)[i * pred_movies + j] = predict(i, j,
                    n_movies, k_val, movie_distances, u_ratings);
            }
        }
        else {
            for (int j = 0; j < pred_movies; j++)
            {
                (*knn_predictions)[i * pred_movies + j] = 0;
            }
        }

        if (i % int(pred_users/100) == 0)
        {
            percent += 1;
            std::cout << "KNN Prediction is " << percent << "% complete." << std::endl;
        }
    }
}
