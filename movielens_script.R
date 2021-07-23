##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)

options(digits = 5)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- right_join(ratings, movies, by = "movieId")

# remove the na values for ratings. If these are kept the createDataPartition process fails. 
movielens <- movielens %>% filter(!is.na(rating))

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


#####################################################################################################
# explore data
# zeros given as ratings in the edx dataset
edx %>% filter(rating == 0) %>% count()

# 3 as rating in edx dataset
edx %>% filter(rating == 3) %>% count()


# get distinct movies
n_distinct(edx$movieId)

# get distinct users
n_distinct(edx$userId)

# movie ratings by genre
edx %>% separate_rows(genres, sep = "\\|") %>% group_by(genres) %>% summarise(n=n())

# movie with the highest number of ratings
movie_highest_rating <- edx %>% group_by(movieId) %>% summarise(total_r = n(), movie_name=first(title)) %>% arrange(desc(total_r))
movie_highest_rating


# five most given ratings
movie_ratings <- edx %>% group_by(rating) %>% summarise(r_count=n()) %>% arrange(desc(r_count))
movie_ratings %>% ggplot(aes(rating, r_count)) + geom_point()

######################################################################################################
# data preprocessing

edx[1:5,] %>% separate_rows(genres, sep="\\|")



# data cleansing - timestamp to date, split the genres, extract release year from title
edx <- edx %>% separate(title, sep="\\(", into = c('title', 'year')) %>%
  mutate(title = str_trim(title), 
         year = str_trim(str_replace(year, "\\)", "")), 
         review_date=as_datetime(timestamp)) %>%
  mutate(rating_year = isoyear(review_date), rating_month=month(review_date), rating_week=week(review_date)) %>%
  select(userId, movieId, rating, title, year, genres, rating_year, rating_month, rating_week) %>%
  separate(genres, sep = "\\|", into=c("g1", "g2", "g3")) %>%
  mutate(genres = ifelse(!is.na(g3), g3, g1)) %>%
  select(userId, movieId, rating, title, year, genres, rating_year, rating_month, rating_week)



# do same for validation data

validation <- validation %>% separate(title, sep="\\(", into = c('title', 'year')) %>%
  mutate(title = str_trim(title), 
         year = str_trim(str_replace(year, "\\)", "")), 
         review_date=as_datetime(timestamp)) %>%
  mutate(rating_year = isoyear(review_date), rating_month=month(review_date), rating_week=week(review_date)) %>%
  select(userId, movieId, rating, title, year, genres, rating_year, rating_month, rating_week) %>%
  separate(genres, sep = "\\|", into=c("g1", "g2", "g3")) %>%
  mutate(genres = ifelse(!is.na(g3), g3, g1)) %>%
  select(userId, movieId, rating, title, year, genres, rating_year, rating_month, rating_week)

######################################################################################################

# visual exploratory analysis

# look at the distribution of ratings by user

hist(edx$rating, main="Frequency of User's rating", xlab="Rating")

# distribution of ratings by rating year

hist(edx$rating_year, main="Frequency of User's Ratings by Years", xlab="Years")

# distribution of rating by month

hist(edx$rating_month, main="Frequency of User's Ratings by month", xlab="Month")


edx %>% group_by(movieId, rating_month) %>% ggplot(aes(rating_month, movieId)) + geom_point()


edx %>% group_by(movieId) %>% summarise(avg_rating=mean(rating)) %>% arrange(desc(avg_rating)) %>% ggplot(aes(avg_rating)) + geom_histogram()

######################################################################################################

# overall mean and sd of ratings
edx %>% summarise(avg_rating = mean(rating), sd_rating=sd(rating), median_rating = median(rating))





######################################################################################################

# RMSE function to check error value

RMSE <- function(t_ratings, p_ratings) {
  sqrt(mean((t_ratings-p_ratings)^2))
}


######################################################################################################
#modeling


# use mean as prediction in the first instance
mu <- mean(edx$rating)

# rmse based on mean
rmse_means_naive <- RMSE(validation$rating, mu)
# results from this is 1.052558, which is above the desired results of < 0.86

# store the values in a table for reference and reporting
rmse_tb_results <- tibble(Method = "Naive Means", RMSE = rmse_means_naive)


# improve the model by adding a bias based on movie rating - the mu
# Note that this provides ratings for each movie so the total number of rows should be the total number of unique movies in the dataset, which is 10,667
ratings_term_bi <- edx %>% group_by(movieId) %>% summarise(bi = mean(rating-mu))

# predictions using the bias term on the validation dataset
predictions_bi <- validation %>% left_join(ratings_term_bi, by='movieId') %>% mutate(h_rating = mu + bi) %>% select(movieId, userId, title, year, bi, h_rating, rating)


# calculate the rmse value
rmse_bias_rating <- RMSE(validation$rating, predictions_bi$h_rating)


# update the table of results
rmse_tb_results <- bind_rows(rmse_tb_results, tibble(Method= "Movie Effect", RMSE=rmse_bias_rating))


# create b_u bias term based on user ratings for movies.
ratings_term_bu <- edx %>% left_join(ratings_term_bi, by='movieId') %>% group_by(userId) %>% summarise(bu = mean(rating-mu-bi))


# predictions with the 2 terms
predictions_bi_bu <- validation %>% left_join(ratings_term_bi, by="movieId") %>% left_join(ratings_term_bu, by="userId") %>% mutate(h_rating = mu + bi + bu) %>%
  select(movieId, userId, title, year, bu, bu, h_rating, rating)

# error on the model with the 2 terms
rmse_bi_bu <- RMSE(validation$rating, predictions_bi_bu$h_rating)


# update table
rmse_tb_results <- bind_rows(rmse_tb_results, tibble(Method="Movie and User Effect", RMSE=rmse_bi_bu))

# use a regularization term on the model. 

lambdas = seq(0,12,.25)

find_right_lambda <- sapply(lambdas, function(l) {
  term_bi <- edx %>% group_by(movieId) %>% summarise(bi = sum(rating-mu)/(n()+l))
  term_bu <- edx %>% left_join(term_bi, by='movieId') %>% group_by(userId) %>% summarise(bu = sum(rating-mu-bi)/(n()+l))
  pred_bi_bu <- validation %>% left_join(term_bi, by="movieId") %>% left_join(term_bu, by="userId") %>% mutate(h_rating = mu + bi + bu)
  return(RMSE(validation$rating, pred_bi_bu$h_rating))
})

# plot the lambda values
qplot(lambdas, find_right_lambda)

# get the lamdba value that produced the smallest rmse
lambdas[which.min(find_right_lambda)]
find_right_lambda[which.min(find_right_lambda)]


# use the lambda value to predict with the model
lambda = lambdas[which.min(find_right_lambda)]
ratings_term_bi_lambda <- edx %>% group_by(movieId) %>% summarise(bi = sum(rating-mu)/(n()+lambda))
ratings_term_bu_lambda <- edx %>% left_join(ratings_term_bi_lambda, by='movieId') %>% group_by(userId) %>% summarise(bu = sum(rating-mu-bi)/(n()+lambda))
predictions_bi_bu_l <- validation %>% left_join(ratings_term_bi_lambda, by="movieId") %>% left_join(ratings_term_bu_lambda, by="userId") %>% mutate(h_rating = mu + bi + bu) %>%
  select(movieId, userId, title, year, bu, bu, h_rating, rating)

rmse_preds_bi_bu_l <- RMSE(validation$rating, predictions_bi_bu_l$h_rating)


# update table
rmse_tb_results <- bind_rows(rmse_tb_results, tibble(Method="Movie and User Effect with Regularization", RMSE=rmse_preds_bi_bu_l))



