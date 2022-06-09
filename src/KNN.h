void knn(std::vector<std::unordered_map<int, int>> &user_ratings,
    float **movie_distances, int &k_val, int &n_movies, int &n_users,
    float **knn_predictions);

void euclidean_distances(unsigned int **movie_data, const int &n_movies,
    const int &movie_size, float **movie_distances);