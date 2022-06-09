void LoadRatings(std::string ratings_fname, const int &n_movies, const int &n_users,
    std::vector<std::unordered_map<int, int>> &movie_ratings,
    std::vector<std::unordered_map<int, int>> &user_ratings);

void LoadGenres(std::string movie_fname, int &n_movies,
    int &movie_size, unsigned int **movie_data, std::vector<std::string> genres);