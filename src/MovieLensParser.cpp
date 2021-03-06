#include <math.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <unordered_map>

bool is_number(const std::string& s)
{
    std::string::const_iterator it = s.begin();
    while (it != s.end() && std::isdigit(*it)) ++it;
    return !s.empty() && it == s.end();
}

std::vector<std::string> split(const std::string& str, const std::string& delim)
{
    std::vector<std::string> tokens;
    size_t prev = 0, pos = 0;
    do
    {
        pos = str.find(delim, prev);
        if (pos == std::string::npos) pos = str.length();
        std::string token = str.substr(prev, pos-prev);
        if (!token.empty()) tokens.push_back(token);
        prev = pos + delim.length();
    }
    while (pos < str.length() && prev < str.length());
    return tokens;
}

void LoadRatings(std::string ratings_fname, const int &n_movies, const int &n_users,
    std::vector<std::unordered_map<int, int>> &movie_ratings,
    std::vector<std::unordered_map<int, int>> &user_ratings)
{
    // movie_ratings: for each movie, the users and ratings that it has

    movie_ratings.reserve(n_movies);
    std::unordered_map<int,int> blank;
    for (int i = 0; i < n_movies; i++)
    {
        movie_ratings[i] = blank;
    }

    // user_ratings: for each user, the movie ids and ratings

    user_ratings.reserve(n_users);
    for (int i = 0; i < n_users; i++)
    {
        user_ratings[i] = blank;
    }

    std::ifstream inputFile(ratings_fname);
    if (inputFile.is_open())
    {
        std::vector<std::string> tokens;
        std::string line;

        int movie_index;
        int user_no;
        int rating;
        while (std::getline(inputFile, line))
        {
            //std::cout << line << std::endl;

            tokens = split(line, "\t");
            
            if (tokens.size() == 4)
            {
                
                user_no = std::stoi(tokens[0]) - 1;
                movie_index = std::stoi(tokens[1]) - 1;
                rating = std::stoi(tokens[2]);

                //std::cout << movie_index << "," << user_no << "," << rating << std::endl;
                if (movie_index < n_movies) {
                    movie_ratings[movie_index][user_no] = rating;
                    user_ratings[user_no][movie_index] = rating;
                }
            }
        }
    }

}


void LoadGenres100k(std::string movie_fname, int &n_movies,
    int &movie_size, unsigned int **movie_data)
{
    // movie_fname      :
    // n_movies (1682)  :
    // movie_size (20)  :
    // movie_data       :

    //storing 2d movie data in a 1d array row-wise
    *movie_data = new unsigned int[n_movies * movie_size];
    std::ifstream inputFile(movie_fname);

    if (inputFile.is_open())
    {
        std::string line;
        int genre_count;
        int movie_count = 0;
        std::string i;

        // Read the movie entry into memory
        while (movie_count < n_movies && std::getline(inputFile, line))
        {
            std::istringstream iss(line);
            genre_count = 0;
            while (std::getline(iss, i, '|'))
            {
                if (is_number(i))
                {
                    // Update the entry for the given category
                    if (genre_count > 0)
                        (*movie_data)[(movie_count * movie_size) + genre_count - 1] = abs(std::atoi(i.c_str()));
                    genre_count++;
                }
            }
            movie_count++;
        }
        inputFile.close();
    }
}


void LoadGenres(std::string dir_name, int &n_movies,
    int &movie_size, unsigned int **movie_data, std::vector<std::string> genres)
{
    LoadGenres100k(dir_name + "u.item", n_movies, movie_size, movie_data);
}

// Function to load data into movie and user dataset pointers
// Make sure these things are int pointers

// Movie: movie_id, and then 19 ints of either 0 or 1
// corresponding to the genres that movies are a part of
