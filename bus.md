ST558 - Project 2
================
Sandra Aziz, Simon Weisenhorn

## Introduction

This project focuses around an online news popularity dataset provided
by the UC Irvine Machine Learning Repository. The dataset consists of 61
total variables where 58 are considered predictive, 2 are
non-predictive, and 1 is the targeted response. Since 2 of the variables
are non-predictive and not of interest to our summaries, they will be
removed when bringing in the data. The remaining predictor variables
consist of information regarding aspects of online news articles. These
aspects include things ranging from the rate of positive words, the
level of subjectivity, the amount of images, and the number of keywords
in the metadata to name just a few of the many variables that we are
interested in exploring their relationship with the target response
variable, which is the number of shares each news article received.
Additionally, there are a six total data channels that break up the
dataset based on which data channel the news article falls under. Each
report will be based on a singular data channel to amass six total
reports.

The purpose of this analysis is to first gain a better understanding of
how the relationships between the number of shares change as different
predictor variables change. This exploration will occur in the
summarizations section below and will be investigated on the training
dataset after splitting the full dataset into a 70/30 split of training
data and testing data. Following this investigation, we will enter the
methods section, which aims to model the response through a few
different kinds of models. Each model will model the shares variable as
the response and utilize all of the variables labeled “predictive” as
the predictors. The models will also preprocess the training data and
use cross-validation to select the best version of itself. After all of
the models have been created, they will be compared and selected based
on the smallest RMSE value. The purpose of creating multiple models is
to ensure we are picking a model that predicts the testing data the best
so that our predictions are more accurate.

## Loading Libraries

Before we load our data or do any analysis, we will first load the
libraries needed throughout the project:

``` r
library(tidyverse) #for cleaning the data
library(ggplot2) # for creating ggplots
library(stringr) # for str_to_title()
library(caret) # for model training and evaluation
library(gbm) #Needed this library to run the training function for boosted tree
```

## Reading in Dataset

Now, we will read in our data set.

``` r
# reading in the data
data <- read_csv("data/OnlineNewsPopularity.csv", show_col_types=FALSE)
head(data) #viewing the first 6 observations of the data
```

    ## # A tibble: 6 × 61
    ##   url       timedelta n_tokens_title n_tokens_content n_unique_tokens n_non_stop_words n_non_stop_unique_to…¹
    ##   <chr>         <dbl>          <dbl>            <dbl>           <dbl>            <dbl>                  <dbl>
    ## 1 http://m…       731             12              219           0.664             1.00                  0.815
    ## 2 http://m…       731              9              255           0.605             1.00                  0.792
    ## 3 http://m…       731              9              211           0.575             1.00                  0.664
    ## 4 http://m…       731              9              531           0.504             1.00                  0.666
    ## 5 http://m…       731             13             1072           0.416             1.00                  0.541
    ## 6 http://m…       731             10              370           0.560             1.00                  0.698
    ## # ℹ abbreviated name: ¹​n_non_stop_unique_tokens
    ## # ℹ 54 more variables: num_hrefs <dbl>, num_self_hrefs <dbl>, num_imgs <dbl>, num_videos <dbl>,
    ## #   average_token_length <dbl>, num_keywords <dbl>, data_channel_is_lifestyle <dbl>,
    ## #   data_channel_is_entertainment <dbl>, data_channel_is_bus <dbl>, data_channel_is_socmed <dbl>,
    ## #   data_channel_is_tech <dbl>, data_channel_is_world <dbl>, kw_min_min <dbl>, kw_max_min <dbl>,
    ## #   kw_avg_min <dbl>, kw_min_max <dbl>, kw_max_max <dbl>, kw_avg_max <dbl>, kw_min_avg <dbl>,
    ## #   kw_max_avg <dbl>, kw_avg_avg <dbl>, self_reference_min_shares <dbl>, self_reference_max_shares <dbl>, …

Next, we will subset our data set into one with the desired channel of
interest.

``` r
#This function was written to assist in data filtering

data_subsetter <- function(channelOfInterest) {
  
  if(!(channelOfInterest %in% c("lifestyle", "entertainment", "bus", "socmed", 
                                "tech", "world"))){
    stop("Error: The channel must be lifestyle, entertainment, bus, socmed, 
         tech, or world")
  }
  #This conditional statement ensures that the user passed one of the possible
  #six data channels, otherwise the function is stopped and an error message is
  #presented
  
  subset <- data %>% filter(eval(parse(text=paste0("data_channel_is_", 
                                                channelOfInterest))) == 1)
  #Subsetting the data to filter the rows based on the given data channel
    
  thinner <- subset[, c(14:19)] #Selecting the 6 data channel columns
  thinner <- thinner[, colSums(thinner != 0) == 0] 
  #Extracting the columns that were not selected by the user input
  drops <- names(thinner) #Selecting the names of irrelevant data channels
  drops <- c(drops, "url") 
  #adding the url column to be removed because it is non-predictive
  data2 <- subset[ , !(names(subset) %in% drops)]
  #removing irrelevant data channels from dataset
  
  channelOfInterest <<- str_to_title(channelOfInterest) 
  #Making global variable out of user supplied channel for later uses
  
  return(data2) 
  #Returning dataset with filtered rows based on the selected channel and 
  #removing the other channels that the user did not supply
}
```

``` r
channel_data <- data_subsetter(params[[1]][1]) #subsetting for given channel
head(channel_data) #inspecting subsetted data
```

    ## # A tibble: 6 × 55
    ##   timedelta n_tokens_title n_tokens_content n_unique_tokens n_non_stop_words n_non_stop_unique_to…¹ num_hrefs
    ##       <dbl>          <dbl>            <dbl>           <dbl>            <dbl>                  <dbl>     <dbl>
    ## 1       731              9              255           0.605             1.00                  0.792         3
    ## 2       731              9              211           0.575             1.00                  0.664         3
    ## 3       731              8              397           0.625             1.00                  0.806        11
    ## 4       731             13              244           0.560             1.00                  0.680         3
    ## 5       731             11              723           0.491             1.00                  0.642        18
    ## 6       731              8              708           0.482             1.00                  0.688         8
    ## # ℹ abbreviated name: ¹​n_non_stop_unique_tokens
    ## # ℹ 48 more variables: num_self_hrefs <dbl>, num_imgs <dbl>, num_videos <dbl>, average_token_length <dbl>,
    ## #   num_keywords <dbl>, data_channel_is_bus <dbl>, kw_min_min <dbl>, kw_max_min <dbl>, kw_avg_min <dbl>,
    ## #   kw_min_max <dbl>, kw_max_max <dbl>, kw_avg_max <dbl>, kw_min_avg <dbl>, kw_max_avg <dbl>,
    ## #   kw_avg_avg <dbl>, self_reference_min_shares <dbl>, self_reference_max_shares <dbl>,
    ## #   self_reference_avg_sharess <dbl>, weekday_is_monday <dbl>, weekday_is_tuesday <dbl>,
    ## #   weekday_is_wednesday <dbl>, weekday_is_thursday <dbl>, weekday_is_friday <dbl>, …

Now that we’ve subset our data set. We will now split it into train and
test sets for summary and analysis purposes.

``` r
set.seed(558) #setting seed for reproducibility 

trainIndex <- createDataPartition(channel_data$shares, p = 0.7, 
                                  list = FALSE)
#Creating indexes for training data 

channelTrain <- channel_data[trainIndex, ] #Creating training dataset
channelTest <- channel_data[-trainIndex, ] #Creating testing dataset

channelTrain_summary <- channelTrain 
#Creating copy of training data so that summaries can be performed on it below
#This is so that summary variables can be added to the training data without 
#having to add them to the test set
```

## Summarizations

After splitting our data into train and test sets, we will now create
some summary statistics and analysis plots on our train set.

### The following summarizations were created by Sandra Aziz

``` r
# summary stats for shares
summary_stats <- summary(channelTrain_summary$shares)
summary_stats
```

    ##     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
    ##      1.0    952.2   1400.0   3265.6   2500.0 690400.0

The statistics above explore the spread of the `shares` variable. If the
mean is less than the median, the data is skewed to the left. If the
median is less than the mean, then the data is skewed to the right.

``` r
# contingency table for number of videos
videos_table <- table(channelTrain_summary$num_videos)
videos_table
```

    ## 
    ##    0    1    2    3    4    5    6    7    8   10   11   12   13   16   17   18   19   20   21   23   26 
    ## 3443  670  117   32   19    7    7    3    1   17   28    2    2    2    1    1    2    9    9    2    1 
    ##   31   65   73   74   75 
    ##    1    1    2    2    1

The contingency table above shows the frequency of the number of videos
in each observation of our data.

``` r
ggplot(channelTrain_summary, aes(x = n_tokens_content, y = shares)) +
geom_point() +
labs(x = "Number of words in the content", y = "Number of Shares") +
ggtitle("Shares vs. Number of words in the content")
```

![](bus_files/figure-gfm/summaries%203-1.png)<!-- -->

``` r
# Creating ggplot of number of words in the content vs shares

ggplot(channelTrain_summary, aes(x = rate_positive_words, y = shares)) +
geom_point() +
labs(x = "Positive Word Rate", y = "Number of Shares") +
ggtitle("Shares vs. Positive Word Rate")
```

![](bus_files/figure-gfm/summaries%203-2.png)<!-- -->

``` r
# Creating ggplot of positive word rate vs shares

ggplot(channelTrain_summary, aes(x = rate_negative_words, y = shares)) +
geom_point() +
labs(x = "Negative Word Rate", y = "Number of Shares") +
ggtitle("Shares vs. Negative Word Rate")
```

![](bus_files/figure-gfm/summaries%203-3.png)<!-- -->

``` r
# Creating ggplot of negative word rate vs shares
```

Above are three scatter plots that explore the relationship between the
number of shares and the number of words in the content, the positive
word rate, and negative word rate, respectively.

### The following summarizations were created by Simon Weisenhorn

``` r
sunday_data <- channelTrain_summary %>% filter(weekday_is_sunday == 1) %>% 
  transmute(day="Sunday", shares=shares)
#Creating new dataset of shares for when the day is Sunday

monday_data <- channelTrain_summary %>% filter(weekday_is_monday == 1) %>% 
  transmute(day="Monday", shares=shares)
#Creating new dataset of shares for when the day is Monday

tuesday_data <- channelTrain_summary %>% filter(weekday_is_tuesday == 1) %>% 
  transmute(day="Tuesday", shares=shares)
#Creating new dataset of shares for when the day is Tuesday

wednesday_data <- channelTrain_summary %>% filter(weekday_is_wednesday == 1) %>% 
  transmute(day="Wednesday", shares=shares)
#Creating new dataset of shares for when the day is Wednesday

thursday_data <- channelTrain_summary %>% filter(weekday_is_thursday == 1) %>% 
  transmute(day="Thursday", shares=shares)
#Creating new dataset of shares for when the day is Thursday

friday_data <- channelTrain_summary %>% filter(weekday_is_friday == 1) %>% 
  transmute(day="Friday", shares=shares)
#Creating new dataset of shares for when the day is Friday

saturday_data <- channelTrain_summary %>% filter(weekday_is_saturday == 1) %>% 
  transmute(day="Saturday", shares=shares)
#Creating new dataset of shares for when the day is Saturday

weekly_share_data <- rbind(sunday_data, monday_data, tuesday_data, 
                           wednesday_data, thursday_data, friday_data, 
                           saturday_data)
#Stacking the rows of the seven previous datasets together with rbind

weekly_share_data %>% group_by(day) %>% summarise(Min = min(shares),
                                              firstQuartile = quantile(shares, 
                                                                       0.25),
                                              Avg=mean(shares),
                                              Med=median(shares),
                                              thirdQuartile = quantile(shares, 
                                                                       0.75),
                                              max = max(shares),
                                              stdDev = sd(shares)) %>%
  arrange(match(day, c("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", 
                       "Friday", "Saturday")))
```

    ## # A tibble: 7 × 8
    ##   day         Min firstQuartile   Avg   Med thirdQuartile    max stdDev
    ##   <chr>     <dbl>         <dbl> <dbl> <dbl>         <dbl>  <dbl>  <dbl>
    ## 1 Sunday      692         1500  3141.  2100          3500  25300  3209.
    ## 2 Monday        1          946. 4494.  1400          2600 690400 33775.
    ## 3 Tuesday      44          920. 3128.  1300          2300 310800 12241.
    ## 4 Wednesday    63          891  2802.  1300          2200 139500  8234.
    ## 5 Thursday     81          895  3016.  1300          2200 306100 15118.
    ## 6 Friday       22          991  2450.  1500          2400 102200  5948.
    ## 7 Saturday    318         1800  4663.  2600          3900 144400 11585.

``` r
#Creating numerical summary of the amount of shares by day of the week
```

We can inspect the numerical summaries of shares split by the day the
article was posted. These summaries provide an indication of how the
amount of shares shift as the day of the week changes.

``` r
weekly_share_summaries <- weekly_share_data %>% group_by(day) %>% 
  summarise(Average = mean(shares), Median = median(shares))
#Creating new variables for the average and median amount of shares by day

weekly_share_summaries$day <- factor(weekly_share_summaries$day, 
                                     levels=c("Sunday", "Monday", "Tuesday", 
                                              "Wednesday", "Thursday", "Friday", 
                                              "Saturday"))
#Reordering the days of the week so that they appear in chronological order in 
#the subsequent graph

weekly_share_summaries <- weekly_share_summaries %>% 
  pivot_longer(cols = c(Average, Median),names_to="statistic", 
               values_to = "value")
#Converting data from wide to long format so that the fill type can be used 
#in the graph below

plot4 <- ggplot(weekly_share_summaries, aes(x=day, y=value, fill=statistic)) 
#Creating the base of the plot

plot4 + geom_bar(stat = "identity", position = "dodge") + 
  #utilizing geom_bar to create a side by side bar graph
  labs(x = "Day of Week", y = "Amount of Shares",
       title = paste0("Bar Chart for Center of Shares by Publish Day for 
    the ",channelOfInterest, " Channel")) +
  #Adding descriptive labels
  theme(plot.title = element_text(hjust = 0.5)) +
  #Centering the title
  scale_fill_discrete(name = "Measure of Center")
```

![](bus_files/figure-gfm/Bar%20Chart%20by%20Day-1.png)<!-- -->

``` r
  #Renaming the legend for clarity
```

We can inspect the measures of the center for the amount of shares by
each day an article was posted. If the mean (in red) is taller than the
median (in blue) then the amount of shares is skewed right for that
particular day. On the other hand, if the median (in blue) is taller
than the mean (in red) then the amount of shares is skewed left for that
particular day. Finally, if the mean and median are similar heights then
the amount of shares are symmetrically distributed.

``` r
weekend_data <- channelTrain_summary %>% filter(is_weekend == 1) %>% 
  transmute(day="Weekend", shares=shares)
#Creating new dataset of shares for when the article was posted on a weekend

weekday_data <- channelTrain_summary %>% filter(is_weekend == 0) %>% 
  transmute(day="Weekday", shares=shares)
#Creating new dataset of shares for when the article was posted on a weekday

new_weekly_share_data <- rbind(weekend_data, weekday_data)
#Stacking the rows of the seven previous datasets together with rbind

maxxlim <- (quantile(new_weekly_share_data$shares, 0.75) - 
              quantile(new_weekly_share_data$shares, 0.25)) * 1.5
#creating the maximum x value by removing any share values that may be
#considered an outlier so that the histogram is easier to view

plot5 <- ggplot(new_weekly_share_data, aes(x=shares, fill=day)) 
#Creating the base of the plot

plot5 + geom_histogram(data=subset(new_weekly_share_data, day == 'Weekday'), 
                       bins=20, alpha = 0.75) +
  #Adding the first histogram layer for the amount of shares if the article 
  #was posted on a weekday
  geom_histogram(data=subset(new_weekly_share_data, day == 'Weekend'), bins=20, 
                 alpha = 0.75) +
  #Adding the second histogram layer for the amount of shares if the article 
  #was posted on a weekend
  labs(x = "Amount of Shares", y = "Frequency", 
       title = paste0("Histogram of Amount of Shares for the ",channelOfInterest,
                      " Channel")) +
  #Adding descriptive labels
  xlim(0,maxxlim) +
  #limiting the x axis to avoid outliers so that the histogram is viewable
  theme(plot.title = element_text(hjust = 0.5)) +
  #Centering the title
  scale_fill_discrete(name = "Part of Week")
```

![](bus_files/figure-gfm/Histogram%20of%20Shares%20by%20Part%20of%20Week-1.png)<!-- -->

``` r
  #Renaming the legend for clarity
```

We can inspect the distributions of the amount of shares based on
whether the article was posted on a weekday or a weekend. If the
distributions are shaped similarly, then we would have no indication
that the distribution of the amount of shares changes for whether the
article was posted on a weekday or a weekend. However, if one of the
distributions are shaped differently than the other, such as skewed a
certain direction or multi-modal, then we would have reason to believe
that the distribution of the amount of shares changes for whether the
article was posted on a weekday or a weekend.

``` r
med_shares <- median(channelTrain_summary$shares)
#Storing the median amount of shares for a cutoff value

channelTrain_summary <- channelTrain_summary %>% 
  mutate(share_indicator = case_when((shares >= med_shares) ~ "High Shares",
                           (shares < med_shares) ~ "Low Shares"),
#Creating a new variable for indicating when shares are higher than 
#the median amount of all shares for this channel
         subjective_indicator = case_when(global_subjectivity >= 0.5 ~ 
                                            "More Subjective",
                           (global_subjectivity < 0.5) ~ "More Objective"))
#Creating a new variable for indicating when articles are more than half
#subjective or less than half subjective

table(channelTrain_summary$share_indicator, 
      channelTrain_summary$subjective_indicator)
```

    ##              
    ##               More Objective More Subjective
    ##   High Shares           1779             549
    ##   Low Shares            1706             348

``` r
#Creating contingency table for sharing indicator with subjective indicator
```

We can inspect the distributions of the sharing indicator with the
subjective indicator. An article with a high sharing rate is one that
has the same amount of shares or more than the median amount of shares
for all articles in the dataset, whereas an article with a low sharing
rate is one that has less than the median amount of shares for all
articles in the dataset. Additionally, an article is considered more
objective if the text subjectivity is less than .5, whereas an article
is considered more subjective if the text subjectivity is greater than
or equal to .5. Thus, these groups provide insight into how sharing is
affected when the text subjectivity is changed on a binary scale.

``` r
channelTrain_summary <- channelTrain_summary %>% 
  mutate(image_indicator = case_when(num_imgs == 0 ~ "No Pictures",
                           num_imgs == 1 ~ "One Picture",
                           num_imgs > 1 ~ "Two or More Pictures",))
#Creating a new variable for indicating when the number of images in the article
#is zero, 1, or more than 1.

table(channelTrain_summary$share_indicator, 
      channelTrain_summary$image_indicator)
```

    ##              
    ##               No Pictures One Picture Two or More Pictures
    ##   High Shares         276        1512                  540
    ##   Low Shares          242        1535                  277

``` r
#Creating contingency table for sharing indicator with image indicator
```

We can inspect the distributions of the sharing indicator with the image
indicator. An article with a high sharing rate is one that has the same
amount of shares or more than the median amount of shares for all
articles in the dataset, whereas an article with a low sharing rate is
one that has less than the median amount of shares for all articles in
the dataset. Additionally, if the article has no pictures then it is
labeled no pictures, or if the article has a singular picture then it is
labeled one picture, or if the article has more than one picture then it
is labeled two or more pictures Thus, these groups provide insight into
how sharing is affected when the amount of pictures are changed on a
categorical scale.

``` r
channelTrain_summary <- channelTrain_summary %>% 
  mutate(video_indicator = case_when(num_videos == 0 ~ "No Videos",
                           num_videos == 1 ~ "One Video",
                           num_videos > 1 ~ "Two or More Videos",))
#Creating a new variable for indicating when the number of videos in the article
#is zero, 1, or more than 1.

table(channelTrain_summary$share_indicator, 
      channelTrain_summary$video_indicator)
```

    ##              
    ##               No Videos One Video Two or More Videos
    ##   High Shares      1740       407                181
    ##   Low Shares       1703       263                 88

``` r
#Creating contingency table for sharing indicator with video indicator
```

We can inspect the distributions of the sharing indicator with the video
indicator. An article with a high sharing rate is one that has the same
amount of shares or more than the median amount of shares for all
articles in the dataset, whereas an article with a low sharing rate is
one that has less than the median amount of shares for all articles in
the dataset. Additionally, if the article has no videos then it is
labeled no videos, or if the article has a singular video then it is
labeled one video, or if the article has more than one video then it is
labeled two or more videos Thus, these groups provide insight into how
sharing is affected when the amount of videos are changed on a
categorical scale.

``` r
plot6 <- ggplot(channelTrain_summary, aes(x=num_hrefs, y=shares, 
                                          color=num_self_hrefs)) 
#Creating the base of the plot

plot6 + geom_point() + 
  scale_color_gradient(low="blue", high="red", 
                       name="Number of Links 
Published by 
Mashable") +
  #Here I am adding to the plot to create a scatterplot where the points 
  #are colored by amount of links to other articles published by Mashable
  labs(x = "Number of Links in Article", y = "Amount of Shares",
       title = paste0("Amount of Shares vs Number of Links in an Article for the "
                      ,channelOfInterest," Channel")) +
  #Adding descriptive labels
  theme(plot.title = element_text(hjust = 0.5), title= element_text(size=10), 
        legend.title=element_text(size=10)) 
```

![](bus_files/figure-gfm/Scatterplot%20for%20Links-1.png)<!-- -->

``` r
  #Centering the title and scaling text to fit properly
```

We can inspect the trend of the amount of shares as a function of the
number of links in an article. If the data points display an upward
trend, then we would find an association between a higher amount of
shares given a larger number of links in an article. If the data points
display a downward trend, then we would find an association between a
lower amount of shares given a smaller number of links in an article.
Finally, each datapoint has been colored based on the amount of links to
other articles published by Mashable. A high amount of links to other
articles published by Mashable are indicated with red datapoints and a
low amount of links to other articles published by Mashable are
indicated with blue datapoints.

``` r
plot7 <- ggplot(channelTrain_summary, aes(x=num_keywords, y=shares)) 
#Creating the base of the plot

plot7 + geom_point() +
  #Here I am adding to the plot to create a scatterplot 
  labs(x = "Number of Keywords in the Metadata", y="Amount of Shares",
       title = paste0("Amount of Shares vs Number of Keywords in the Metadata of 
  an Article for the ",channelOfInterest," Channel")) +
  #Adding descriptive labels
  theme(plot.title = element_text(hjust = 0.5)) 
```

![](bus_files/figure-gfm/Scatterplot%20for%20Keywords-1.png)<!-- -->

``` r
  #Centering the title
```

We can inspect the trend of the amount of shares as a function of the
number of keywords in the metadata. If the data points display an upward
trend, then we would find an association between a higher amount of
shares given a larger number of keywords in the metadata for an article.
If the data points display a downward trend, then we would find an
association between a lower amount of shares given a smaller number of
keywords in the metadata for an article.

``` r
channelTrain_summary %>% group_by(num_keywords) %>% summarise(Min = min(shares),
                                              firstQuartile = quantile(shares, 
                                                                       0.25),
                                              Avg=mean(shares),
                                              Med=median(shares),
                                              thirdQuartile = quantile(shares, 
                                                                       0.75),
                                              max = max(shares),
                                              stdDev = sd(shares)) 
```

    ## # A tibble: 9 × 8
    ##   num_keywords   Min firstQuartile   Avg   Med thirdQuartile    max stdDev
    ##          <dbl> <dbl>         <dbl> <dbl> <dbl>         <dbl>  <dbl>  <dbl>
    ## 1            2   200          834. 1510.  1017          1175   5800  1770.
    ## 2            3    81          958. 2074.  1300          2000  48700  4025.
    ## 3            4    28          934. 2231.  1200          2100  47800  3997.
    ## 4            5    22          927  2868.  1300          2300 306100 12514.
    ## 5            6   156          919. 3795.  1400          2600 690400 25734.
    ## 6            7    63          982. 3886.  1500          2700 652900 26385.
    ## 7            8   119          997. 2939.  1500          2700  94400  6300.
    ## 8            9   393          975. 3720.  1400          2500 310800 17728.
    ## 9           10     1         1100  3684.  1650          3200 110200  9327.

``` r
#Creating numerical summary of the amount of shares by number of keywords 
#in the metadata
```

We can inspect the numerical summaries of shares split by the number of
keywords in the metadata for the article. These summaries provide an
indication of how the amount of shares shift as the number of keywords
in the metadata for the article changes.

## Modeling

Now that we’ve explored some summaries and statistics using our data, we
will now fit some models using our train set then test them using the
test set.  
We will first fit two linear regression models. Linear regression is a
statistical modeling technique that is used to establish a relationship
between a dependent variable (in our case, the number of shares) and one
or more independent (predictor) variables. It assumes a linear
relationship between the variables, where the dependent variable can be
predicted as a linear combination of the independent variables. It
estimates the coefficients that represent the change in the response
variable for each unit change in the predictor variables, assuming all
other variables are held constant. The goal is to estimate the
coefficients that minimize the difference between the predicted values
and the actual values of the dependent variable.  
Our first linear regression model will explore a linear fit with
interaction terms. This model is used to test whether the relationship
between the dependent and the independent variable changes depending on
the value of another independent variable.  
Our second linear regression model will explore a linear fit without
interaction terms. This model assumes that the effect of each predictor
on the dependent variable is independent of other predictors in the
model.

``` r
simon_share_lmfit <- train(shares ~ .^2, data = channelTrain,
                           method = "lm",
                           preProcess = c("center", "scale"),
                           trControl = trainControl(method = "cv", number = 5))
#Fitting linear model with all interactions

summary(simon_share_lmfit)
```

    ## 
    ## Call:
    ## lm(formula = .outcome ~ ., data = dat)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ## -71519  -3699   -197   3350 168817 
    ## 
    ## Coefficients: (305 not defined because of singularities)
    ##                                                             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                                3.266e+03  1.600e+02  20.413  < 2e-16 ***
    ## timedelta                                                  1.380e+05  1.729e+05   0.798 0.424876    
    ## n_tokens_title                                            -2.095e+03  2.775e+04  -0.075 0.939830    
    ## n_tokens_content                                          -8.038e+03  2.025e+04  -0.397 0.691489    
    ## n_unique_tokens                                           -2.296e+04  2.328e+04  -0.986 0.324058    
    ## n_non_stop_words                                           1.717e+04  5.658e+04   0.303 0.761571    
    ## n_non_stop_unique_tokens                                   1.215e+04  1.865e+04   0.651 0.514810    
    ## num_hrefs                                                  1.800e+04  1.200e+04   1.499 0.133891    
    ## num_self_hrefs                                             1.198e+03  9.401e+03   0.127 0.898566    
    ## num_imgs                                                  -3.543e+04  1.044e+04  -3.393 0.000700 ***
    ## num_videos                                                 6.622e+04  9.747e+05   0.068 0.945842    
    ## average_token_length                                       7.714e+03  8.294e+03   0.930 0.352421    
    ## num_keywords                                               9.375e+02  8.368e+04   0.011 0.991062    
    ## data_channel_is_bus                                               NA         NA      NA       NA    
    ## kw_min_min                                                 6.217e+04  5.888e+04   1.056 0.291101    
    ## kw_max_min                                                -5.955e+03  6.533e+05  -0.009 0.992728    
    ## kw_avg_min                                                 9.631e+04  6.323e+05   0.152 0.878950    
    ## kw_min_max                                                -1.207e+02  2.246e+04  -0.005 0.995710    
    ## kw_max_max                                                 5.923e+04  7.769e+04   0.762 0.445882    
    ## kw_avg_max                                                 1.040e+05  1.101e+05   0.945 0.344922    
    ## kw_min_avg                                                 1.376e+04  2.138e+04   0.644 0.519907    
    ## kw_max_avg                                                 3.544e+05  9.172e+05   0.386 0.699268    
    ## kw_avg_avg                                                -1.604e+05  1.871e+05  -0.857 0.391358    
    ## self_reference_min_shares                                  7.709e+03  1.075e+05   0.072 0.942816    
    ## self_reference_max_shares                                 -9.878e+04  1.223e+05  -0.808 0.419304    
    ## self_reference_avg_sharess                                 9.595e+04  1.892e+05   0.507 0.612162    
    ## weekday_is_monday                                         -2.134e+04  1.589e+04  -1.343 0.179375    
    ## weekday_is_tuesday                                        -2.175e+04  1.625e+04  -1.339 0.180792    
    ## weekday_is_wednesday                                      -1.527e+04  1.635e+04  -0.934 0.350335    
    ## weekday_is_thursday                                       -1.424e+04  1.657e+04  -0.860 0.389995    
    ## weekday_is_friday                                         -1.913e+04  1.445e+04  -1.324 0.185672    
    ## weekday_is_saturday                                       -5.486e+03  1.186e+04  -0.462 0.643834    
    ## weekday_is_sunday                                                 NA         NA      NA       NA    
    ## is_weekend                                                        NA         NA      NA       NA    
    ## LDA_00                                                     3.801e+03  8.849e+03   0.430 0.667533    
    ## LDA_01                                                    -1.231e+04  7.500e+03  -1.641 0.100926    
    ## LDA_02                                                     3.046e+02  7.983e+03   0.038 0.969562    
    ## LDA_03                                                    -7.735e+03  8.614e+03  -0.898 0.369274    
    ## LDA_04                                                            NA         NA      NA       NA    
    ## global_subjectivity                                       -1.815e+03  8.215e+03  -0.221 0.825196    
    ## global_sentiment_polarity                                 -2.802e+04  1.717e+04  -1.632 0.102817    
    ## global_rate_positive_words                                -3.035e+03  1.702e+04  -0.178 0.858443    
    ## global_rate_negative_words                                -4.773e+02  1.422e+04  -0.034 0.973220    
    ## rate_positive_words                                        1.610e+04  1.831e+04   0.879 0.379425    
    ## rate_negative_words                                       -1.465e+03  5.012e+03  -0.292 0.770036    
    ## avg_positive_polarity                                      1.875e+04  1.274e+04   1.472 0.141159    
    ## min_positive_polarity                                      6.796e+02  8.942e+03   0.076 0.939419    
    ## max_positive_polarity                                     -1.780e+04  1.089e+04  -1.634 0.102401    
    ## avg_negative_polarity                                     -3.826e+03  2.103e+04  -0.182 0.855641    
    ## min_negative_polarity                                      2.010e+04  1.818e+04   1.106 0.268797    
    ## max_negative_polarity                                      6.843e+03  1.336e+04   0.512 0.608536    
    ## title_subjectivity                                         7.981e+03  1.299e+04   0.614 0.539048    
    ## title_sentiment_polarity                                  -1.566e+04  9.601e+03  -1.631 0.103024    
    ## abs_title_subjectivity                                     2.305e+03  9.586e+03   0.241 0.809953    
    ## abs_title_sentiment_polarity                               3.323e+03  1.324e+04   0.251 0.801846    
    ## `timedelta:n_tokens_title`                                 3.133e+03  1.422e+03   2.203 0.027674 *  
    ## `timedelta:n_tokens_content`                               2.108e+03  1.531e+03   1.377 0.168651    
    ## `timedelta:n_unique_tokens`                                3.070e+03  6.632e+03   0.463 0.643392    
    ## `timedelta:n_non_stop_words`                              -1.159e+05  1.731e+05  -0.669 0.503440    
    ## `timedelta:n_non_stop_unique_tokens`                       4.047e+03  6.764e+03   0.598 0.549694    
    ## `timedelta:num_hrefs`                                     -1.862e+03  1.038e+03  -1.793 0.073010 .  
    ## `timedelta:num_self_hrefs`                                 1.033e+03  8.040e+02   1.284 0.199079    
    ## `timedelta:num_imgs`                                       3.107e+03  9.508e+02   3.268 0.001095 ** 
    ## `timedelta:num_videos`                                    -5.754e+03  2.725e+03  -2.111 0.034819 *  
    ## `timedelta:average_token_length`                           3.444e+03  5.868e+03   0.587 0.557366    
    ## `timedelta:num_keywords`                                  -3.745e+03  1.744e+03  -2.148 0.031827 *  
    ## `timedelta:data_channel_is_bus`                                   NA         NA      NA       NA    
    ## `timedelta:kw_min_min`                                    -6.574e+04  1.619e+04  -4.059 5.04e-05 ***
    ## `timedelta:kw_max_min`                                     9.951e+03  8.988e+03   1.107 0.268349    
    ## `timedelta:kw_avg_min`                                     5.895e+03  6.505e+03   0.906 0.364859    
    ## `timedelta:kw_min_max`                                    -1.148e+03  1.081e+03  -1.062 0.288275    
    ## `timedelta:kw_max_max`                                    -3.613e+04  1.002e+04  -3.606 0.000316 ***
    ## `timedelta:kw_avg_max`                                    -2.799e+01  1.637e+03  -0.017 0.986362    
    ## `timedelta:kw_min_avg`                                    -5.042e+02  9.744e+02  -0.517 0.604875    
    ## `timedelta:kw_max_avg`                                     3.231e+03  4.640e+03   0.696 0.486197    
    ## `timedelta:kw_avg_avg`                                     9.784e+01  3.888e+03   0.025 0.979923    
    ## `timedelta:self_reference_min_shares`                     -5.536e+03  1.064e+04  -0.520 0.602787    
    ## `timedelta:self_reference_max_shares`                     -1.256e+04  1.081e+04  -1.162 0.245192    
    ## `timedelta:self_reference_avg_sharess`                     1.097e+04  1.790e+04   0.613 0.539942    
    ## `timedelta:weekday_is_monday`                              2.596e+02  1.356e+03   0.192 0.848126    
    ## `timedelta:weekday_is_tuesday`                             6.568e+02  1.405e+03   0.467 0.640235    
    ## `timedelta:weekday_is_wednesday`                           1.964e+02  1.422e+03   0.138 0.890120    
    ## `timedelta:weekday_is_thursday`                            7.696e+02  1.471e+03   0.523 0.601019    
    ## `timedelta:weekday_is_friday`                              9.752e+02  1.219e+03   0.800 0.423845    
    ## `timedelta:weekday_is_saturday`                           -8.804e+01  8.821e+02  -0.100 0.920503    
    ## `timedelta:weekday_is_sunday`                                     NA         NA      NA       NA    
    ## `timedelta:is_weekend`                                            NA         NA      NA       NA    
    ## `timedelta:LDA_00`                                         8.748e+02  1.540e+03   0.568 0.570000    
    ## `timedelta:LDA_01`                                        -7.379e+02  8.467e+02  -0.872 0.383515    
    ## `timedelta:LDA_02`                                        -1.217e+03  7.489e+02  -1.625 0.104275    
    ## `timedelta:LDA_03`                                         9.282e+02  7.240e+02   1.282 0.199877    
    ## `timedelta:LDA_04`                                                NA         NA      NA       NA    
    ## `timedelta:global_subjectivity`                           -1.152e+03  2.081e+03  -0.553 0.580079    
    ## `timedelta:global_sentiment_polarity`                      1.238e+03  2.282e+03   0.543 0.587371    
    ## `timedelta:global_rate_positive_words`                    -3.048e+03  2.301e+03  -1.325 0.185349    
    ## `timedelta:global_rate_negative_words`                     4.448e+03  2.222e+03   2.002 0.045348 *  
    ## `timedelta:rate_positive_words`                            5.973e+03  5.942e+03   1.005 0.314876    
    ## `timedelta:rate_negative_words`                                   NA         NA      NA       NA    
    ## `timedelta:avg_positive_polarity`                         -2.557e+03  2.832e+03  -0.903 0.366730    
    ## `timedelta:min_positive_polarity`                          8.219e+01  9.974e+02   0.082 0.934326    
    ## `timedelta:max_positive_polarity`                          1.819e+03  1.878e+03   0.968 0.332911    
    ## `timedelta:avg_negative_polarity`                          3.558e+03  2.336e+03   1.523 0.127796    
    ## `timedelta:min_negative_polarity`                         -2.529e+03  1.707e+03  -1.482 0.138513    
    ## `timedelta:max_negative_polarity`                         -2.962e+03  1.320e+03  -2.244 0.024876 *  
    ## `timedelta:title_subjectivity`                             2.081e+03  1.169e+03   1.779 0.075282 .  
    ## `timedelta:title_sentiment_polarity`                      -5.004e+02  8.066e+02  -0.620 0.535071    
    ## `timedelta:abs_title_subjectivity`                         1.678e+03  1.099e+03   1.528 0.126715    
    ## `timedelta:abs_title_sentiment_polarity`                   5.180e+02  1.109e+03   0.467 0.640541    
    ## `n_tokens_title:n_tokens_content`                          2.995e+03  2.423e+03   1.236 0.216635    
    ## `n_tokens_title:n_unique_tokens`                          -1.785e+03  5.757e+03  -0.310 0.756483    
    ## `n_tokens_title:n_non_stop_words`                          1.184e+04  2.952e+04   0.401 0.688312    
    ## `n_tokens_title:n_non_stop_unique_tokens`                  5.564e+03  5.660e+03   0.983 0.325700    
    ## `n_tokens_title:num_hrefs`                                -2.925e+03  1.657e+03  -1.765 0.077664 .  
    ## `n_tokens_title:num_self_hrefs`                            1.898e+03  1.228e+03   1.546 0.122138    
    ## `n_tokens_title:num_imgs`                                  2.120e+03  1.494e+03   1.419 0.156015    
    ## `n_tokens_title:num_videos`                               -3.007e+02  2.643e+03  -0.114 0.909420    
    ## `n_tokens_title:average_token_length`                     -7.449e+03  4.250e+03  -1.753 0.079756 .  
    ## `n_tokens_title:num_keywords`                              6.395e+01  1.786e+03   0.036 0.971447    
    ## `n_tokens_title:data_channel_is_bus`                              NA         NA      NA       NA    
    ## `n_tokens_title:kw_min_min`                               -7.098e+02  2.103e+03  -0.338 0.735721    
    ## `n_tokens_title:kw_max_min`                                1.013e+04  1.302e+04   0.778 0.436549    
    ## `n_tokens_title:kw_avg_min`                               -3.020e+03  9.439e+03  -0.320 0.749075    
    ## `n_tokens_title:kw_min_max`                                1.398e+03  1.465e+03   0.954 0.340046    
    ## `n_tokens_title:kw_max_max`                                5.455e+02  2.874e+03   0.190 0.849500    
    ## `n_tokens_title:kw_avg_max`                               -1.283e+03  2.875e+03  -0.446 0.655530    
    ## `n_tokens_title:kw_min_avg`                                4.501e+02  1.800e+03   0.250 0.802556    
    ## `n_tokens_title:kw_max_avg`                               -9.964e+03  7.488e+03  -1.331 0.183395    
    ## `n_tokens_title:kw_avg_avg`                                2.376e+03  5.998e+03   0.396 0.692087    
    ## `n_tokens_title:self_reference_min_shares`                -1.141e+04  1.517e+04  -0.752 0.452091    
    ## `n_tokens_title:self_reference_max_shares`                -8.472e+03  1.628e+04  -0.520 0.602830    
    ## `n_tokens_title:self_reference_avg_sharess`                6.018e+03  2.682e+04   0.224 0.822488    
    ## `n_tokens_title:weekday_is_monday`                         2.084e+03  1.990e+03   1.047 0.294997    
    ## `n_tokens_title:weekday_is_tuesday`                       -1.532e+03  2.010e+03  -0.762 0.445888    
    ## `n_tokens_title:weekday_is_wednesday`                     -1.003e+03  2.040e+03  -0.492 0.623045    
    ## `n_tokens_title:weekday_is_thursday`                       3.182e+02  2.055e+03   0.155 0.876971    
    ## `n_tokens_title:weekday_is_friday`                        -1.566e+03  1.825e+03  -0.858 0.390849    
    ## `n_tokens_title:weekday_is_saturday`                      -1.860e+02  1.450e+03  -0.128 0.897934    
    ## `n_tokens_title:weekday_is_sunday`                                NA         NA      NA       NA    
    ## `n_tokens_title:is_weekend`                                       NA         NA      NA       NA    
    ## `n_tokens_title:LDA_00`                                    2.734e+03  1.613e+03   1.696 0.090062 .  
    ## `n_tokens_title:LDA_01`                                    2.471e+02  1.173e+03   0.211 0.833183    
    ## `n_tokens_title:LDA_02`                                    1.411e+03  1.236e+03   1.142 0.253704    
    ## `n_tokens_title:LDA_03`                                    1.894e+03  1.302e+03   1.455 0.145833    
    ## `n_tokens_title:LDA_04`                                           NA         NA      NA       NA    
    ## `n_tokens_title:global_subjectivity`                       7.112e+02  1.859e+03   0.383 0.702035    
    ## `n_tokens_title:global_sentiment_polarity`                -9.516e+02  3.023e+03  -0.315 0.752903    
    ## `n_tokens_title:global_rate_positive_words`                3.612e+03  2.590e+03   1.394 0.163285    
    ## `n_tokens_title:global_rate_negative_words`               -4.944e+03  3.097e+03  -1.596 0.110501    
    ## `n_tokens_title:rate_positive_words`                      -1.275e+04  5.030e+03  -2.535 0.011282 *  
    ## `n_tokens_title:rate_negative_words`                              NA         NA      NA       NA    
    ## `n_tokens_title:avg_positive_polarity`                     1.416e+03  2.834e+03   0.500 0.617288    
    ## `n_tokens_title:min_positive_polarity`                    -1.261e+03  1.409e+03  -0.895 0.370981    
    ## `n_tokens_title:max_positive_polarity`                    -5.872e+02  1.942e+03  -0.302 0.762384    
    ## `n_tokens_title:avg_negative_polarity`                     4.817e+03  3.008e+03   1.601 0.109434    
    ## `n_tokens_title:min_negative_polarity`                    -2.249e+03  2.429e+03  -0.926 0.354546    
    ## `n_tokens_title:max_negative_polarity`                    -5.037e+03  1.935e+03  -2.603 0.009296 ** 
    ## `n_tokens_title:title_subjectivity`                        1.693e+03  1.801e+03   0.940 0.347064    
    ## `n_tokens_title:title_sentiment_polarity`                  1.642e+03  1.508e+03   1.089 0.276219    
    ## `n_tokens_title:abs_title_subjectivity`                   -2.313e+02  1.396e+03  -0.166 0.868413    
    ## `n_tokens_title:abs_title_sentiment_polarity`             -1.897e+03  1.952e+03  -0.972 0.330989    
    ## `n_tokens_content:n_unique_tokens`                        -8.136e+03  4.744e+03  -1.715 0.086410 .  
    ## `n_tokens_content:n_non_stop_words`                               NA         NA      NA       NA    
    ## `n_tokens_content:n_non_stop_unique_tokens`                1.272e+04  7.028e+03   1.809 0.070504 .  
    ## `n_tokens_content:num_hrefs`                              -2.931e+03  1.331e+03  -2.203 0.027686 *  
    ## `n_tokens_content:num_self_hrefs`                         -2.475e+03  1.113e+03  -2.224 0.026223 *  
    ## `n_tokens_content:num_imgs`                                1.428e+03  1.133e+03   1.260 0.207669    
    ## `n_tokens_content:num_videos`                             -1.107e+04  3.187e+03  -3.474 0.000519 ***
    ## `n_tokens_content:average_token_length`                   -8.115e+03  1.034e+04  -0.785 0.432468    
    ## `n_tokens_content:num_keywords`                            9.029e+02  2.584e+03   0.349 0.726802    
    ## `n_tokens_content:data_channel_is_bus`                            NA         NA      NA       NA    
    ## `n_tokens_content:kw_min_min`                              4.316e+03  1.596e+03   2.705 0.006864 ** 
    ## `n_tokens_content:kw_max_min`                              2.148e+03  5.804e+03   0.370 0.711335    
    ## `n_tokens_content:kw_avg_min`                             -4.169e+03  5.203e+03  -0.801 0.423004    
    ## `n_tokens_content:kw_min_max`                              5.454e+02  1.072e+03   0.509 0.611013    
    ## `n_tokens_content:kw_max_max`                              1.118e+04  4.662e+03   2.398 0.016562 *  
    ## `n_tokens_content:kw_avg_max`                             -1.132e+03  2.886e+03  -0.392 0.694903    
    ## `n_tokens_content:kw_min_avg`                              2.324e+03  1.594e+03   1.458 0.145040    
    ## `n_tokens_content:kw_max_avg`                             -5.910e+03  4.338e+03  -1.362 0.173146    
    ## `n_tokens_content:kw_avg_avg`                              2.474e+03  5.455e+03   0.453 0.650267    
    ## `n_tokens_content:self_reference_min_shares`               1.085e+04  9.940e+03   1.092 0.275123    
    ## `n_tokens_content:self_reference_max_shares`               2.069e+04  1.210e+04   1.710 0.087397 .  
    ## `n_tokens_content:self_reference_avg_sharess`             -1.504e+04  1.722e+04  -0.873 0.382656    
    ## `n_tokens_content:weekday_is_monday`                       6.978e+02  1.433e+03   0.487 0.626363    
    ## `n_tokens_content:weekday_is_tuesday`                      3.742e+02  1.264e+03   0.296 0.767226    
    ## `n_tokens_content:weekday_is_wednesday`                    2.359e+03  1.243e+03   1.897 0.057924 .  
    ## `n_tokens_content:weekday_is_thursday`                     2.572e+02  1.235e+03   0.208 0.835051    
    ## `n_tokens_content:weekday_is_friday`                       1.456e+03  1.206e+03   1.207 0.227612    
    ## `n_tokens_content:weekday_is_saturday`                     1.455e+03  1.099e+03   1.324 0.185604    
    ## `n_tokens_content:weekday_is_sunday`                              NA         NA      NA       NA    
    ## `n_tokens_content:is_weekend`                                     NA         NA      NA       NA    
    ## `n_tokens_content:LDA_00`                                  1.564e+03  2.972e+03   0.526 0.598732    
    ## `n_tokens_content:LDA_01`                                  1.254e+03  9.476e+02   1.323 0.185774    
    ## `n_tokens_content:LDA_02`                                  8.930e+02  1.027e+03   0.869 0.384779    
    ## `n_tokens_content:LDA_03`                                  2.042e+02  1.094e+03   0.187 0.851888    
    ## `n_tokens_content:LDA_04`                                         NA         NA      NA       NA    
    ## `n_tokens_content:global_subjectivity`                    -1.081e+04  4.489e+03  -2.407 0.016135 *  
    ## `n_tokens_content:global_sentiment_polarity`               1.506e+04  4.499e+03   3.348 0.000822 ***
    ## `n_tokens_content:global_rate_positive_words`             -5.887e+03  5.257e+03  -1.120 0.262886    
    ## `n_tokens_content:global_rate_negative_words`              3.919e+03  4.641e+03   0.844 0.398578    
    ## `n_tokens_content:rate_positive_words`                    -2.867e+03  1.505e+04  -0.190 0.848953    
    ##  [ reached getOption("max.print") -- omitted 1286 rows ]
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 10590 on 3201 degrees of freedom
    ## Multiple R-squared:  0.7345, Adjusted R-squared:  0.6367 
    ## F-statistic: 7.506 on 1180 and 3201 DF,  p-value: < 2.2e-16

``` r
#viewing the resulting model

pred <- predict(simon_share_lmfit, newdata = channelTest)
#Creating predictions based on the channelTest data
l1 <- postResample(pred, obs = channelTest$shares)
#creating first linear model postResample results object
l1 #viewing how well the first linear model did 
```

    ##         RMSE     Rsquared          MAE 
    ## 9.331223e+04 2.905815e-03 1.349812e+04

``` r
sandra_share_lmfit <- train(shares ~ ., data = channelTrain,
                           method = "lm",
                           preProcess = c("center", "scale"),
                           trControl = trainControl(method = "cv", number = 5))
#Fitting linear model with no interactions

summary(sandra_share_lmfit)
```

    ## 
    ## Call:
    ## lm(formula = .outcome ~ ., data = dat)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ## -53661  -2574   -921    620 675608 
    ## 
    ## Coefficients: (4 not defined because of singularities)
    ##                              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                   3265.62     262.43  12.444  < 2e-16 ***
    ## timedelta                      677.49     377.30   1.796 0.072623 .  
    ## n_tokens_title                 309.81     279.23   1.110 0.267267    
    ## n_tokens_content               -59.69     537.53  -0.111 0.911579    
    ## n_unique_tokens                133.00     955.18   0.139 0.889268    
    ## n_non_stop_words                12.27    1076.37   0.011 0.990903    
    ## n_non_stop_unique_tokens       773.85     774.97   0.999 0.318069    
    ## num_hrefs                      208.25     366.87   0.568 0.570296    
    ## num_self_hrefs                 265.33     295.75   0.897 0.369699    
    ## num_imgs                       469.35     290.74   1.614 0.106529    
    ## num_videos                     563.65     291.16   1.936 0.052947 .  
    ## average_token_length          -655.93     416.78  -1.574 0.115608    
    ## num_keywords                   522.42     351.32   1.487 0.137082    
    ## data_channel_is_bus                NA         NA      NA       NA    
    ## kw_min_min                     438.09     536.00   0.817 0.413783    
    ## kw_max_min                      47.67    1694.38   0.028 0.977557    
    ## kw_avg_min                     504.87    1696.04   0.298 0.765963    
    ## kw_min_max                    -553.12     321.47  -1.721 0.085399 .  
    ## kw_max_max                      10.74     588.78   0.018 0.985450    
    ## kw_avg_max                     487.88     608.31   0.802 0.422577    
    ## kw_min_avg                     -84.57     377.93  -0.224 0.822945    
    ## kw_max_avg                   -2735.82     860.20  -3.180 0.001481 ** 
    ## kw_avg_avg                    3253.67     921.36   3.531 0.000418 ***
    ## self_reference_min_shares      846.20     864.72   0.979 0.327840    
    ## self_reference_max_shares    -1054.63     941.72  -1.120 0.262818    
    ## self_reference_avg_sharess    1571.34    1485.59   1.058 0.290242    
    ## weekday_is_monday              639.02     504.75   1.266 0.205571    
    ## weekday_is_tuesday             114.76     511.79   0.224 0.822595    
    ## weekday_is_wednesday            51.06     514.76   0.099 0.920986    
    ## weekday_is_thursday            170.34     520.35   0.327 0.743406    
    ## weekday_is_friday              -68.12     463.72  -0.147 0.883220    
    ## weekday_is_saturday            211.49     342.92   0.617 0.537456    
    ## weekday_is_sunday                  NA         NA      NA       NA    
    ## is_weekend                         NA         NA      NA       NA    
    ## LDA_00                         222.61     369.71   0.602 0.547134    
    ## LDA_01                         125.10     303.58   0.412 0.680291    
    ## LDA_02                          77.90     319.80   0.244 0.807572    
    ## LDA_03                         746.63     312.24   2.391 0.016836 *  
    ## LDA_04                             NA         NA      NA       NA    
    ## global_subjectivity            241.11     337.32   0.715 0.474777    
    ## global_sentiment_polarity     -259.22     697.27  -0.372 0.710086    
    ## global_rate_positive_words    -149.54     551.89  -0.271 0.786443    
    ## global_rate_negative_words     236.85     679.03   0.349 0.727257    
    ## rate_positive_words            490.76    2565.20   0.191 0.848289    
    ## rate_negative_words            150.13    2526.25   0.059 0.952613    
    ## avg_positive_polarity         -290.23     519.33  -0.559 0.576286    
    ## min_positive_polarity          -93.95     363.57  -0.258 0.796103    
    ## max_positive_polarity          257.05     423.72   0.607 0.544120    
    ## avg_negative_polarity          518.65     712.74   0.728 0.466844    
    ## min_negative_polarity        -1207.94     615.07  -1.964 0.049606 *  
    ## max_negative_polarity         -427.79     467.46  -0.915 0.360164    
    ## title_subjectivity            -218.09     429.80  -0.507 0.611875    
    ## title_sentiment_polarity        26.22     313.99   0.084 0.933456    
    ## abs_title_subjectivity         313.36     344.88   0.909 0.363616    
    ## abs_title_sentiment_polarity    97.65     426.22   0.229 0.818797    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 17370 on 4331 degrees of freedom
    ## Multiple R-squared:  0.03341,    Adjusted R-squared:  0.02225 
    ## F-statistic: 2.994 on 50 and 4331 DF,  p-value: 1.258e-11

``` r
#viewing the resulting model

pred <- predict(simon_share_lmfit, newdata = channelTest)
#Creating predictions based on the channelTest data
l2 <- postResample(pred, obs = channelTest$shares)
#creating second linear model postResample results object
l2 #viewing how well the second linear model did 
```

    ##         RMSE     Rsquared          MAE 
    ## 9.331223e+04 2.905815e-03 1.349812e+04

The next model we will explore is a boosted tree model. A boosted tree
is an ensemble based model that treats the errors created by previous
decision trees. By considering the errors of trees in previous rounds,
the boosted tree model forms new trees sequentially where each tree is
dependent on the previous tree. These kinds of models are useful for
non-linear data, which make them great for real world applications.

``` r
allTuningParameters <- expand.grid(n.trees=c(25,50,100,150,200), 
                                   interaction.depth=c(1,2,3,4), 
                                   shrinkage=0.1, n.minobsinnode = 10)
#Creating dataframe for the tuneGrid arguement using expand.grid

boostTreeFit <- train(shares ~ ., data = channelTrain,
                      method = "gbm",
                      preProcess = c("center", "scale"),
                      trControl = trainControl(method = "repeatedcv", 
                                               number = 5, repeats=3),
                      tuneGrid=allTuningParameters,
                      verbose=FALSE)
#Fitting the boosted tree model

boostTreeFit
```

    ## Stochastic Gradient Boosting 
    ## 
    ## 4382 samples
    ##   54 predictor
    ## 
    ## Pre-processing: centered (54), scaled (54) 
    ## Resampling: Cross-Validated (5 fold, repeated 3 times) 
    ## Summary of sample sizes: 3506, 3506, 3505, 3505, 3506, 3506, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE      Rsquared     MAE     
    ##   1                   25      16084.48  0.002209675  3204.049
    ##   1                   50      16244.46  0.005478640  3282.169
    ##   1                  100      16436.37  0.005642290  3368.364
    ##   1                  150      16536.52  0.006300035  3385.842
    ##   1                  200      16588.74  0.007177561  3392.088
    ##   2                   25      16167.22  0.001530715  3238.884
    ##   2                   50      16416.98  0.003876467  3306.065
    ##   2                  100      16905.18  0.003674973  3430.531
    ##   2                  150      17092.14  0.004136343  3488.765
    ##   2                  200      17219.50  0.004578214  3531.757
    ##   3                   25      16222.00  0.003297884  3276.068
    ##   3                   50      16518.16  0.004119169  3350.304
    ##   3                  100      17141.11  0.003669503  3504.812
    ##   3                  150      17677.99  0.004417146  3562.459
    ##   3                  200      18010.13  0.003897791  3644.380
    ##   4                   25      16139.73  0.003093872  3227.258
    ##   4                   50      16577.77  0.003251451  3371.714
    ##   4                  100      17257.81  0.001791626  3521.141
    ##   4                  150      17520.61  0.002826685  3579.146
    ##   4                  200      17983.29  0.002022818  3659.250
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## Tuning parameter 'n.minobsinnode' was
    ##  held constant at a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 25, interaction.depth = 1, shrinkage = 0.1
    ##  and n.minobsinnode = 10.

``` r
#viewing the resulting model

pred <- predict(boostTreeFit, newdata = channelTest)
#Creating predictions based on the channelTest data
bt <- postResample(pred, obs = channelTest$shares)
#creating boosted tree postResample results object
bt #viewing how well the boosted tree did 
```

    ##         RMSE     Rsquared          MAE 
    ## 6.335236e+03 2.316678e-02 2.479806e+03

The last model we will explore is a random forest model. It is an
ensemble tree-based model that trains a collection of decision trees
using bootstrap aggregating (bagging) and feature randomness. It
combines the predictions of multiple decision trees to make predictions
based on the majority vote or averaging of the individual trees.

``` r
randomForestFit <- train(shares ~ ., data = channelTrain,
                      method = "rf",
                      trControl = trainControl(method = "repeatedcv", 
                                               number = 5, repeats = 3),
                      tuneGrid = data.frame(mtry = 1:15))
# Fitting the random forests model

randomForestFit
```

    ## Random Forest 
    ## 
    ## 4382 samples
    ##   54 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold, repeated 3 times) 
    ## Summary of sample sizes: 3506, 3506, 3506, 3505, 3505, 3506, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  RMSE      Rsquared     MAE     
    ##    1    14546.05  0.023870660  2996.713
    ##    2    14807.51  0.018876300  3084.868
    ##    3    14944.31  0.017760706  3141.439
    ##    4    15206.78  0.016959902  3188.410
    ##    5    15343.61  0.015353531  3214.733
    ##    6    15483.08  0.013890983  3241.916
    ##    7    15632.49  0.013945712  3277.573
    ##    8    15739.42  0.013043224  3290.779
    ##    9    15721.29  0.011166122  3299.856
    ##   10    15813.75  0.012976803  3323.070
    ##   11    16021.80  0.011520849  3351.885
    ##   12    16131.60  0.010445814  3355.285
    ##   13    16169.20  0.009984846  3367.929
    ##   14    16267.47  0.008646846  3385.374
    ##   15    16413.38  0.007832135  3404.303
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final value used for the model was mtry = 1.

``` r
# Viewing the resulting model

pred <- predict(randomForestFit, newdata = channelTest)
#Creating predictions based on the channelTest data
rf <- postResample(pred, obs = channelTest$shares)
#creating random forest postResample results object
rf #viewing how well the random forest did
```

    ##         RMSE     Rsquared          MAE 
    ## 5516.2466693    0.1205796 2237.5166598

## Comparison

Now that we’ve fit and tested the above four models using our train and
test sets. We will now compare all four models to pick the best one
overall.

``` r
modelList <- list(firstLinearModel = l1[[1]], secondLinearModel = l2[[1]], 
                  boostedTree = bt[[1]], randomForest = rf[[1]])
# Storing all RMSE in one list

bestModel <- names(modelList)[which.min(modelList)]
# Finding model with lowest RMSE

paste("The best model with the lowest RMSE was", bestModel, sep=" ")
```

    ## [1] "The best model with the lowest RMSE was randomForest"

``` r
# Printing model with lowest RMSE
```

The model printed above has the lowest RMSE and, thus, is the best model
to use for predicted new values.
