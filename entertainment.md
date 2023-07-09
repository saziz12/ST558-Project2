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
    ##   url                  timedelta n_tokens_title n_tokens_content n_unique_tokens
    ##   <chr>                    <dbl>          <dbl>            <dbl>           <dbl>
    ## 1 http://mashable.com…       731             12              219           0.664
    ## 2 http://mashable.com…       731              9              255           0.605
    ## 3 http://mashable.com…       731              9              211           0.575
    ## 4 http://mashable.com…       731              9              531           0.504
    ## 5 http://mashable.com…       731             13             1072           0.416
    ## 6 http://mashable.com…       731             10              370           0.560
    ## # ℹ 56 more variables: n_non_stop_words <dbl>, n_non_stop_unique_tokens <dbl>,
    ## #   num_hrefs <dbl>, num_self_hrefs <dbl>, num_imgs <dbl>, num_videos <dbl>,
    ## #   average_token_length <dbl>, num_keywords <dbl>,
    ## #   data_channel_is_lifestyle <dbl>, data_channel_is_entertainment <dbl>,
    ## #   data_channel_is_bus <dbl>, data_channel_is_socmed <dbl>,
    ## #   data_channel_is_tech <dbl>, data_channel_is_world <dbl>, kw_min_min <dbl>,
    ## #   kw_max_min <dbl>, kw_avg_min <dbl>, kw_min_max <dbl>, kw_max_max <dbl>, …

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
    ##   timedelta n_tokens_title n_tokens_content n_unique_tokens n_non_stop_words
    ##       <dbl>          <dbl>            <dbl>           <dbl>            <dbl>
    ## 1       731             12              219           0.664             1.00
    ## 2       731              9              531           0.504             1.00
    ## 3       731             14              194           0.765             1.00
    ## 4       731             12              161           0.669             1.00
    ## 5       731             11              454           0.566             1.00
    ## 6       731             12              177           0.741             1.00
    ## # ℹ 50 more variables: n_non_stop_unique_tokens <dbl>, num_hrefs <dbl>,
    ## #   num_self_hrefs <dbl>, num_imgs <dbl>, num_videos <dbl>,
    ## #   average_token_length <dbl>, num_keywords <dbl>,
    ## #   data_channel_is_entertainment <dbl>, kw_min_min <dbl>, kw_max_min <dbl>,
    ## #   kw_avg_min <dbl>, kw_min_max <dbl>, kw_max_max <dbl>, kw_avg_max <dbl>,
    ## #   kw_min_avg <dbl>, kw_max_avg <dbl>, kw_avg_avg <dbl>,
    ## #   self_reference_min_shares <dbl>, self_reference_max_shares <dbl>, …

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

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##      47     833    1200    3015    2100  210300

The statistics above explore the spread of the `shares` variable. If the
mean is less than the median, the data is skewed to the left. If the
median is less than the mean, then the data is skewed to the right.

``` r
# contingency table for number of videos
videos_table <- table(channelTrain_summary$num_videos)
videos_table
```

    ## 
    ##    0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15 
    ## 2221 1550  439  112   68   39   28   25   20   26   37   31   21   22   23   26 
    ##   16   17   18   19   20   21   22   23   24   25   26   27   28   29   30   32 
    ##   20    9    9    2   16   29   12    4    2   39   62   17    6    1    1    4 
    ##   33   34   35   36   38   46   50   53   58   73 
    ##    9    2    1    1    1    1    2    1    1    1

The contingency table above shows the frequency of the number of videos
in each observation of our data.

``` r
ggplot(channelTrain_summary, aes(x = n_tokens_content, y = shares)) +
geom_point() +
labs(x = "Number of words in the content", y = "Number of Shares") +
ggtitle("Shares vs. Number of words in the content")
```

![](entertainment_files/figure-gfm/summaries%203-1.png)<!-- -->

``` r
# Creating ggplot of number of words in the content vs shares

ggplot(channelTrain_summary, aes(x = rate_positive_words, y = shares)) +
geom_point() +
labs(x = "Positive Word Rate", y = "Number of Shares") +
ggtitle("Shares vs. Positive Word Rate")
```

![](entertainment_files/figure-gfm/summaries%203-2.png)<!-- -->

``` r
# Creating ggplot of positive word rate vs shares

ggplot(channelTrain_summary, aes(x = rate_negative_words, y = shares)) +
geom_point() +
labs(x = "Negative Word Rate", y = "Number of Shares") +
ggtitle("Shares vs. Negative Word Rate")
```

![](entertainment_files/figure-gfm/summaries%203-3.png)<!-- -->

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
    ## 1 Sunday      171         1200  3864.  1700          3900  69500  6563.
    ## 2 Monday       64          802  3015.  1100          2000 112600  7808.
    ## 3 Tuesday      47          801. 2415.  1100          1900  53100  4701.
    ## 4 Wednesday    49          771  3124.  1100          2000 138700  9394.
    ## 5 Thursday     57          799  2900.  1100          1900 197600 10454.
    ## 6 Friday       58          850. 3089.  1200          2000 210300 10192.
    ## 7 Saturday     65         1200  3719.  1600          2900  68300  7277.

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

![](entertainment_files/figure-gfm/Bar%20Chart%20by%20Day-1.png)<!-- -->

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

![](entertainment_files/figure-gfm/Histogram%20of%20Shares%20by%20Part%20of%20Week-1.png)<!-- -->

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
    ##   High Shares           1714             791
    ##   Low Shares            1727             709

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
    ##   High Shares         466         973                 1066
    ##   Low Shares          450        1037                  949

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
    ##   High Shares      1046       829                630
    ##   Low Shares       1175       721                540

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

![](entertainment_files/figure-gfm/Scatterplot%20for%20Links-1.png)<!-- -->

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

![](entertainment_files/figure-gfm/Scatterplot%20for%20Keywords-1.png)<!-- -->

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
    ## 1            2   731          731   731    731           731    731    NA 
    ## 2            3    57          802  2023.  1000          1800  20000  3324.
    ## 3            4   183          768. 2225.  1000          1500  53100  5152.
    ## 4            5    58          796  2682.  1100          1900  82200  6095.
    ## 5            6    49          806. 2823.  1100          2000 197600  8588.
    ## 6            7   109          868  3469.  1200          2300 138700  9435.
    ## 7            8    88          827. 3066.  1200          2375 112600  7165.
    ## 8            9    47          884. 4120.  1300          2600 193400 12457.
    ## 9           10    80          862  2870.  1300          2500 210300  8813.

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
    ## -26176  -2801   -381   1899  81208 
    ## 
    ## Coefficients: (290 not defined because of singularities)
    ##                                                                Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                                   3.015e+03  9.821e+01  30.702  < 2e-16 ***
    ## timedelta                                                     5.800e+03  5.584e+03   1.039 0.299078    
    ## n_tokens_title                                                4.779e+03  2.207e+03   2.166 0.030402 *  
    ## n_tokens_content                                             -8.268e+02  9.732e+03  -0.085 0.932296    
    ## n_unique_tokens                                              -1.828e+05  1.266e+06  -0.144 0.885167    
    ## n_non_stop_words                                             -3.019e+06  1.539e+06  -1.962 0.049796 *  
    ## n_non_stop_unique_tokens                                      9.135e+05  9.686e+05   0.943 0.345718    
    ## num_hrefs                                                    -6.103e+03  6.300e+03  -0.969 0.332734    
    ## num_self_hrefs                                               -2.986e+03  6.534e+03  -0.457 0.647747    
    ## num_imgs                                                      5.930e+03  3.609e+03   1.643 0.100432    
    ## num_videos                                                   -2.708e+03  4.447e+03  -0.609 0.542517    
    ## average_token_length                                          1.851e+04  1.079e+04   1.715 0.086349 .  
    ## num_keywords                                                 -2.710e+02  2.675e+03  -0.101 0.919316    
    ## data_channel_is_entertainment                                        NA         NA      NA       NA    
    ## kw_min_min                                                    1.691e+03  1.733e+04   0.098 0.922276    
    ## kw_max_min                                                    1.639e+04  2.068e+04   0.793 0.427955    
    ## kw_avg_min                                                   -2.021e+04  2.460e+04  -0.822 0.411342    
    ## kw_min_max                                                   -1.374e+04  6.631e+03  -2.072 0.038322 *  
    ## kw_max_max                                                    1.031e+03  1.449e+04   0.071 0.943312    
    ## kw_avg_max                                                   -7.703e+03  5.425e+03  -1.420 0.155717    
    ## kw_min_avg                                                    6.528e+03  3.902e+03   1.673 0.094448 .  
    ## kw_max_avg                                                   -1.201e+03  8.526e+03  -0.141 0.887956    
    ## kw_avg_avg                                                    8.603e+01  7.901e+03   0.011 0.991313    
    ## self_reference_min_shares                                    -9.991e+03  1.777e+04  -0.562 0.573941    
    ## self_reference_max_shares                                    -7.466e+03  4.155e+04  -0.180 0.857414    
    ## self_reference_avg_sharess                                    7.891e+03  4.206e+04   0.188 0.851179    
    ## weekday_is_monday                                             3.677e+03  3.808e+03   0.966 0.334317    
    ## weekday_is_tuesday                                            2.835e+03  3.724e+03   0.761 0.446547    
    ## weekday_is_wednesday                                          3.500e+03  3.781e+03   0.926 0.354608    
    ## weekday_is_thursday                                           1.036e+03  3.777e+03   0.274 0.783883    
    ## weekday_is_friday                                             1.120e+03  3.510e+03   0.319 0.749660    
    ## weekday_is_saturday                                          -6.594e+02  2.928e+03  -0.225 0.821842    
    ## weekday_is_sunday                                                    NA         NA      NA       NA    
    ## is_weekend                                                           NA         NA      NA       NA    
    ## LDA_00                                                       -1.810e+08  7.971e+08  -0.227 0.820399    
    ## LDA_01                                                       -6.565e+08  2.891e+09  -0.227 0.820398    
    ## LDA_02                                                       -2.692e+08  1.186e+09  -0.227 0.820398    
    ## LDA_03                                                       -6.824e+08  3.006e+09  -0.227 0.820399    
    ## LDA_04                                                       -1.903e+08  8.381e+08  -0.227 0.820399    
    ## global_subjectivity                                           1.284e+04  6.699e+03   1.917 0.055350 .  
    ## global_sentiment_polarity                                    -2.556e+04  1.148e+04  -2.226 0.026061 *  
    ## global_rate_positive_words                                   -1.258e+03  1.127e+04  -0.112 0.911063    
    ## global_rate_negative_words                                    7.134e+03  1.196e+04   0.596 0.551050    
    ## rate_positive_words                                           1.541e+04  1.533e+04   1.005 0.315012    
    ## rate_negative_words                                                  NA         NA      NA       NA    
    ## avg_positive_polarity                                        -3.095e+02  9.568e+03  -0.032 0.974195    
    ## min_positive_polarity                                         5.587e+03  5.677e+03   0.984 0.325104    
    ## max_positive_polarity                                         2.052e+04  7.600e+03   2.700 0.006973 ** 
    ## avg_negative_polarity                                        -6.208e+03  1.117e+04  -0.556 0.578359    
    ## min_negative_polarity                                         1.656e+04  9.917e+03   1.669 0.095110 .  
    ## max_negative_polarity                                        -4.714e+02  7.022e+03  -0.067 0.946482    
    ## title_subjectivity                                           -3.647e+03  3.503e+03  -1.041 0.297952    
    ## title_sentiment_polarity                                      5.018e+03  2.761e+03   1.818 0.069211 .  
    ## abs_title_subjectivity                                       -1.530e+03  2.619e+03  -0.584 0.559104    
    ## abs_title_sentiment_polarity                                 -7.148e+03  3.482e+03  -2.053 0.040141 *  
    ## `timedelta:n_tokens_title`                                   -8.881e+02  8.885e+02  -1.000 0.317601    
    ## `timedelta:n_tokens_content`                                  3.934e+03  7.411e+02   5.308 1.17e-07 ***
    ## `timedelta:n_unique_tokens`                                   1.217e+05  4.001e+04   3.042 0.002369 ** 
    ## `timedelta:n_non_stop_words`                                  4.317e+04  5.384e+04   0.802 0.422756    
    ## `timedelta:n_non_stop_unique_tokens`                         -7.332e+04  3.108e+04  -2.359 0.018373 *  
    ## `timedelta:num_hrefs`                                         7.211e+02  4.732e+02   1.524 0.127612    
    ## `timedelta:num_self_hrefs`                                   -2.038e+03  5.117e+02  -3.982 6.95e-05 ***
    ## `timedelta:num_imgs`                                         -1.010e+03  5.419e+02  -1.864 0.062389 .  
    ## `timedelta:num_videos`                                        1.706e+02  3.954e+02   0.431 0.666237    
    ## `timedelta:average_token_length`                             -4.549e+03  3.730e+03  -1.219 0.222764    
    ## `timedelta:num_keywords`                                      1.091e+03  9.196e+02   1.187 0.235428    
    ## `timedelta:data_channel_is_entertainment`                            NA         NA      NA       NA    
    ## `timedelta:kw_min_min`                                       -7.737e+03  7.380e+03  -1.048 0.294544    
    ## `timedelta:kw_max_min`                                       -2.637e+03  2.816e+03  -0.936 0.349235    
    ## `timedelta:kw_avg_min`                                        5.705e+03  3.635e+03   1.570 0.116584    
    ## `timedelta:kw_min_max`                                        2.177e+02  3.751e+02   0.580 0.561629    
    ## `timedelta:kw_max_max`                                       -3.722e+03  3.826e+03  -0.973 0.330739    
    ## `timedelta:kw_avg_max`                                        2.313e+02  7.015e+02   0.330 0.741583    
    ## `timedelta:kw_min_avg`                                        1.696e+02  5.264e+02   0.322 0.747366    
    ## `timedelta:kw_max_avg`                                       -2.149e+03  1.269e+03  -1.694 0.090407 .  
    ## `timedelta:kw_avg_avg`                                       -1.271e+03  1.757e+03  -0.723 0.469699    
    ## `timedelta:self_reference_min_shares`                         4.076e+02  1.256e+03   0.324 0.745583    
    ## `timedelta:self_reference_max_shares`                         1.697e+03  2.816e+03   0.603 0.546810    
    ## `timedelta:self_reference_avg_sharess`                       -1.177e+03  3.478e+03  -0.339 0.735004    
    ## `timedelta:weekday_is_monday`                                 2.048e+02  6.094e+02   0.336 0.736811    
    ## `timedelta:weekday_is_tuesday`                                1.554e+02  6.000e+02   0.259 0.795665    
    ## `timedelta:weekday_is_wednesday`                              1.760e+01  6.133e+02   0.029 0.977105    
    ## `timedelta:weekday_is_thursday`                               4.047e+02  5.930e+02   0.682 0.495041    
    ## `timedelta:weekday_is_friday`                                 5.268e+02  5.430e+02   0.970 0.331982    
    ## `timedelta:weekday_is_saturday`                               5.710e+02  4.305e+02   1.327 0.184741    
    ## `timedelta:weekday_is_sunday`                                        NA         NA      NA       NA    
    ## `timedelta:is_weekend`                                               NA         NA      NA       NA    
    ## `timedelta:LDA_00`                                           -2.004e+02  5.618e+02  -0.357 0.721286    
    ## `timedelta:LDA_01`                                            3.939e+02  1.657e+03   0.238 0.812099    
    ## `timedelta:LDA_02`                                           -5.903e+01  5.700e+02  -0.104 0.917517    
    ## `timedelta:LDA_03`                                            4.643e+02  1.539e+03   0.302 0.762858    
    ## `timedelta:LDA_04`                                                   NA         NA      NA       NA    
    ## `timedelta:global_subjectivity`                              -6.333e+02  1.254e+03  -0.505 0.613574    
    ## `timedelta:global_sentiment_polarity`                         1.336e+03  1.246e+03   1.072 0.283658    
    ## `timedelta:global_rate_positive_words`                       -2.012e+02  1.300e+03  -0.155 0.876958    
    ## `timedelta:global_rate_negative_words`                       -1.577e+03  1.344e+03  -1.173 0.240682    
    ## `timedelta:rate_positive_words`                              -3.822e+03  2.936e+03  -1.302 0.193131    
    ## `timedelta:rate_negative_words`                                      NA         NA      NA       NA    
    ## `timedelta:avg_positive_polarity`                             1.739e+03  1.676e+03   1.037 0.299631    
    ## `timedelta:min_positive_polarity`                            -2.674e+02  5.528e+02  -0.484 0.628552    
    ## `timedelta:max_positive_polarity`                            -2.012e+03  1.106e+03  -1.819 0.068942 .  
    ## `timedelta:avg_negative_polarity`                             2.599e+02  1.289e+03   0.202 0.840212    
    ## `timedelta:min_negative_polarity`                            -7.523e+02  9.428e+02  -0.798 0.424943    
    ## `timedelta:max_negative_polarity`                            -8.536e+02  6.337e+02  -1.347 0.178051    
    ## `timedelta:title_subjectivity`                                2.997e+02  5.270e+02   0.569 0.569559    
    ## `timedelta:title_sentiment_polarity`                         -2.209e+02  3.913e+02  -0.565 0.572385    
    ## `timedelta:abs_title_subjectivity`                            5.880e+01  4.911e+02   0.120 0.904699    
    ## `timedelta:abs_title_sentiment_polarity`                      5.192e+02  5.379e+02   0.965 0.334509    
    ## `n_tokens_title:n_tokens_content`                            -9.124e+02  1.381e+03  -0.661 0.508885    
    ## `n_tokens_title:n_unique_tokens`                              2.163e+04  1.797e+05   0.120 0.904206    
    ## `n_tokens_title:n_non_stop_words`                            -1.633e+05  2.419e+05  -0.675 0.499603    
    ## `n_tokens_title:n_non_stop_unique_tokens`                    -3.391e+04  1.425e+05  -0.238 0.811893    
    ## `n_tokens_title:num_hrefs`                                   -6.165e+02  9.439e+02  -0.653 0.513673    
    ## `n_tokens_title:num_self_hrefs`                               3.355e+02  9.880e+02   0.340 0.734229    
    ## `n_tokens_title:num_imgs`                                     3.063e+02  9.893e+02   0.310 0.756886    
    ## `n_tokens_title:num_videos`                                   5.098e+02  9.064e+02   0.562 0.573811    
    ## `n_tokens_title:average_token_length`                         2.265e+03  3.458e+03   0.655 0.512569    
    ## `n_tokens_title:num_keywords`                                -1.599e+03  9.795e+02  -1.632 0.102768    
    ## `n_tokens_title:data_channel_is_entertainment`                       NA         NA      NA       NA    
    ## `n_tokens_title:kw_min_min`                                  -2.629e+02  1.361e+03  -0.193 0.846797    
    ## `n_tokens_title:kw_max_min`                                  -3.793e+03  6.393e+03  -0.593 0.553008    
    ## `n_tokens_title:kw_avg_min`                                   6.252e+03  6.938e+03   0.901 0.367617    
    ## `n_tokens_title:kw_min_max`                                  -4.775e+02  9.389e+02  -0.509 0.611069    
    ## `n_tokens_title:kw_max_max`                                  -3.788e+03  2.038e+03  -1.858 0.063193 .  
    ## `n_tokens_title:kw_avg_max`                                   2.429e+03  1.526e+03   1.592 0.111473    
    ## `n_tokens_title:kw_min_avg`                                   3.165e+02  1.102e+03   0.287 0.773993    
    ## `n_tokens_title:kw_max_avg`                                   1.893e+03  2.634e+03   0.719 0.472375    
    ## `n_tokens_title:kw_avg_avg`                                  -1.538e+03  2.651e+03  -0.580 0.561784    
    ## `n_tokens_title:self_reference_min_shares`                    6.838e+02  2.341e+03   0.292 0.770290    
    ## `n_tokens_title:self_reference_max_shares`                    5.514e+03  5.840e+03   0.944 0.345147    
    ## `n_tokens_title:self_reference_avg_sharess`                  -3.634e+03  5.234e+03  -0.694 0.487495    
    ## `n_tokens_title:weekday_is_monday`                           -2.937e+03  1.126e+03  -2.608 0.009131 ** 
    ## `n_tokens_title:weekday_is_tuesday`                          -2.755e+03  1.116e+03  -2.469 0.013576 *  
    ## `n_tokens_title:weekday_is_wednesday`                        -3.144e+03  1.109e+03  -2.836 0.004587 ** 
    ## `n_tokens_title:weekday_is_thursday`                         -2.811e+03  1.092e+03  -2.575 0.010053 *  
    ## `n_tokens_title:weekday_is_friday`                           -1.870e+03  1.025e+03  -1.824 0.068247 .  
    ## `n_tokens_title:weekday_is_saturday`                         -1.631e+03  8.363e+02  -1.950 0.051243 .  
    ## `n_tokens_title:weekday_is_sunday`                                   NA         NA      NA       NA    
    ## `n_tokens_title:is_weekend`                                          NA         NA      NA       NA    
    ## `n_tokens_title:LDA_00`                                      -8.706e+02  8.990e+02  -0.968 0.332926    
    ## `n_tokens_title:LDA_01`                                      -1.670e+03  2.453e+03  -0.681 0.496108    
    ## `n_tokens_title:LDA_02`                                      -1.449e+03  1.182e+03  -1.226 0.220245    
    ## `n_tokens_title:LDA_03`                                      -2.770e+03  2.615e+03  -1.060 0.289420    
    ## `n_tokens_title:LDA_04`                                              NA         NA      NA       NA    
    ## `n_tokens_title:global_subjectivity`                          8.995e+02  1.419e+03   0.634 0.526204    
    ## `n_tokens_title:global_sentiment_polarity`                    8.269e+02  2.063e+03   0.401 0.688516    
    ## `n_tokens_title:global_rate_positive_words`                  -1.454e+03  1.592e+03  -0.913 0.361084    
    ## `n_tokens_title:global_rate_negative_words`                   9.756e+02  2.165e+03   0.451 0.652235    
    ## `n_tokens_title:rate_positive_words`                          2.238e+03  3.130e+03   0.715 0.474674    
    ## `n_tokens_title:rate_negative_words`                                 NA         NA      NA       NA    
    ## `n_tokens_title:avg_positive_polarity`                        3.007e+03  1.943e+03   1.548 0.121803    
    ## `n_tokens_title:min_positive_polarity`                       -1.163e+03  9.904e+02  -1.174 0.240290    
    ## `n_tokens_title:max_positive_polarity`                       -2.431e+03  1.454e+03  -1.672 0.094618 .  
    ## `n_tokens_title:avg_negative_polarity`                        2.174e+02  1.892e+03   0.115 0.908550    
    ## `n_tokens_title:min_negative_polarity`                        7.902e+01  1.635e+03   0.048 0.961456    
    ## `n_tokens_title:max_negative_polarity`                        2.719e+02  1.114e+03   0.244 0.807203    
    ## `n_tokens_title:title_subjectivity`                          -2.914e+02  1.046e+03  -0.278 0.780654    
    ## `n_tokens_title:title_sentiment_polarity`                     6.737e+02  7.617e+02   0.884 0.376492    
    ## `n_tokens_title:abs_title_subjectivity`                      -4.197e+02  7.833e+02  -0.536 0.592113    
    ## `n_tokens_title:abs_title_sentiment_polarity`                 4.364e+02  1.001e+03   0.436 0.662787    
    ## `n_tokens_content:n_unique_tokens`                            4.843e+04  2.158e+05   0.224 0.822435    
    ## `n_tokens_content:n_non_stop_words`                                  NA         NA      NA       NA    
    ## `n_tokens_content:n_non_stop_unique_tokens`                  -9.010e+04  1.659e+05  -0.543 0.586976    
    ## `n_tokens_content:num_hrefs`                                 -6.302e+02  6.536e+02  -0.964 0.334993    
    ## `n_tokens_content:num_self_hrefs`                            -2.837e+03  8.079e+02  -3.512 0.000450 ***
    ## `n_tokens_content:num_imgs`                                  -1.838e+03  6.555e+02  -2.804 0.005079 ** 
    ## `n_tokens_content:num_videos`                                 1.438e+03  6.731e+02   2.136 0.032733 *  
    ## `n_tokens_content:average_token_length`                      -5.836e+03  5.652e+03  -1.033 0.301897    
    ## `n_tokens_content:num_keywords`                               4.446e+03  1.251e+03   3.553 0.000386 ***
    ## `n_tokens_content:data_channel_is_entertainment`                     NA         NA      NA       NA    
    ## `n_tokens_content:kw_min_min`                                 1.424e+02  9.087e+02   0.157 0.875482    
    ## `n_tokens_content:kw_max_min`                                -2.230e+03  2.621e+03  -0.851 0.395090    
    ## `n_tokens_content:kw_avg_min`                                 4.983e+03  2.847e+03   1.750 0.080147 .  
    ## `n_tokens_content:kw_min_max`                                -3.503e+02  7.023e+02  -0.499 0.617991    
    ## `n_tokens_content:kw_max_max`                                 7.821e+03  3.458e+03   2.262 0.023765 *  
    ## `n_tokens_content:kw_avg_max`                                 4.856e+03  1.664e+03   2.918 0.003548 ** 
    ## `n_tokens_content:kw_min_avg`                                 3.786e+03  9.195e+02   4.118 3.91e-05 ***
    ## `n_tokens_content:kw_max_avg`                                 1.269e+04  1.607e+03   7.896 3.75e-15 ***
    ## `n_tokens_content:kw_avg_avg`                                -2.190e+04  2.974e+03  -7.364 2.18e-13 ***
    ## `n_tokens_content:self_reference_min_shares`                 -9.247e+01  9.804e+02  -0.094 0.924862    
    ## `n_tokens_content:self_reference_max_shares`                  1.459e+03  2.424e+03   0.602 0.547369    
    ## `n_tokens_content:self_reference_avg_sharess`                -1.939e+03  2.429e+03  -0.798 0.424683    
    ## `n_tokens_content:weekday_is_monday`                          2.048e+02  6.806e+02   0.301 0.763483    
    ## `n_tokens_content:weekday_is_tuesday`                         4.974e+02  6.728e+02   0.739 0.459826    
    ## `n_tokens_content:weekday_is_wednesday`                      -7.491e+02  6.420e+02  -1.167 0.243352    
    ## `n_tokens_content:weekday_is_thursday`                       -1.981e+01  7.152e+02  -0.028 0.977907    
    ## `n_tokens_content:weekday_is_friday`                          1.631e+03  6.343e+02   2.572 0.010154 *  
    ## `n_tokens_content:weekday_is_saturday`                        5.855e+02  5.071e+02   1.155 0.248307    
    ## `n_tokens_content:weekday_is_sunday`                                 NA         NA      NA       NA    
    ## `n_tokens_content:is_weekend`                                        NA         NA      NA       NA    
    ## `n_tokens_content:LDA_00`                                    -1.425e+02  6.435e+02  -0.221 0.824747    
    ## `n_tokens_content:LDA_01`                                     1.497e+03  1.989e+03   0.753 0.451618    
    ## `n_tokens_content:LDA_02`                                    -1.398e+03  1.062e+03  -1.317 0.187959    
    ## `n_tokens_content:LDA_03`                                    -4.681e+02  2.028e+03  -0.231 0.817479    
    ## `n_tokens_content:LDA_04`                                            NA         NA      NA       NA    
    ## `n_tokens_content:global_subjectivity`                        2.144e+03  2.532e+03   0.847 0.397167    
    ## `n_tokens_content:global_sentiment_polarity`                 -3.175e+03  1.865e+03  -1.702 0.088746 .  
    ## `n_tokens_content:global_rate_positive_words`                 1.381e+03  2.380e+03   0.580 0.561678    
    ## `n_tokens_content:global_rate_negative_words`                -6.260e+03  2.640e+03  -2.372 0.017762 *  
    ## `n_tokens_content:rate_positive_words`                       -1.100e+02  5.832e+03  -0.019 0.984959    
    ##  [ reached getOption("max.print") -- omitted 1286 rows ]
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 6903 on 3745 degrees of freedom
    ## Multiple R-squared:  0.4942, Adjusted R-squared:  0.3328 
    ## F-statistic: 3.062 on 1195 and 3745 DF,  p-value: < 2.2e-16

``` r
#viewing the resulting model

pred <- predict(simon_share_lmfit, newdata = channelTest)
#Creating predictions based on the channelTest data
l1 <- postResample(pred, obs = channelTest$shares)
#creating first linear model postResample results object
l1 #viewing how well the first linear model did 
```

    ##         RMSE     Rsquared          MAE 
    ## 1.018860e+04 7.231347e-03 5.455768e+03

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
    ## -36176  -2263  -1065    199 204835 
    ## 
    ## Coefficients: (4 not defined because of singularities)
    ##                                 Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                    3.015e+03  1.159e+02  26.010  < 2e-16 ***
    ## timedelta                      4.483e+02  1.648e+02   2.720 0.006552 ** 
    ## n_tokens_title                -1.203e+01  1.232e+02  -0.098 0.922194    
    ## n_tokens_content               8.818e+02  2.277e+02   3.874 0.000109 ***
    ## n_unique_tokens                1.125e+05  3.990e+04   2.820 0.004817 ** 
    ## n_non_stop_words              -1.148e+05  5.197e+04  -2.209 0.027248 *  
    ## n_non_stop_unique_tokens      -6.117e+04  3.124e+04  -1.958 0.050285 .  
    ## num_hrefs                      2.625e+02  1.441e+02   1.821 0.068627 .  
    ## num_self_hrefs                -3.042e+02  1.375e+02  -2.212 0.026992 *  
    ## num_imgs                       2.567e+01  1.632e+02   0.157 0.875042    
    ## num_videos                    -1.421e+02  1.388e+02  -1.024 0.306122    
    ## average_token_length           7.534e+02  4.240e+02   1.777 0.075671 .  
    ## num_keywords                  -2.764e+02  1.432e+02  -1.930 0.053627 .  
    ## data_channel_is_entertainment         NA         NA      NA       NA    
    ## kw_min_min                     2.889e+02  2.376e+02   1.216 0.224050    
    ## kw_max_min                     5.238e+03  5.274e+02   9.932  < 2e-16 ***
    ## kw_avg_min                    -4.815e+03  5.404e+02  -8.911  < 2e-16 ***
    ## kw_min_max                    -2.506e+02  1.539e+02  -1.628 0.103541    
    ## kw_max_max                    -1.571e+00  2.666e+02  -0.006 0.995299    
    ## kw_avg_max                    -3.169e+02  2.214e+02  -1.432 0.152255    
    ## kw_min_avg                    -7.069e+02  1.855e+02  -3.810 0.000141 ***
    ## kw_max_avg                    -1.001e+03  2.977e+02  -3.361 0.000782 ***
    ## kw_avg_avg                     2.674e+03  3.506e+02   7.626 2.89e-14 ***
    ## self_reference_min_shares      3.918e+02  1.828e+02   2.143 0.032162 *  
    ## self_reference_max_shares      2.656e+02  2.635e+02   1.008 0.313520    
    ## self_reference_avg_sharess    -6.109e+01  3.209e+02  -0.190 0.849033    
    ## weekday_is_monday             -2.417e+02  2.018e+02  -1.197 0.231225    
    ## weekday_is_tuesday            -4.974e+02  1.994e+02  -2.494 0.012647 *  
    ## weekday_is_wednesday          -1.564e+02  2.000e+02  -0.782 0.434335    
    ## weekday_is_thursday           -3.264e+02  1.972e+02  -1.655 0.097935 .  
    ## weekday_is_friday             -1.788e+02  1.838e+02  -0.973 0.330621    
    ## weekday_is_saturday            5.308e+00  1.513e+02   0.035 0.972025    
    ## weekday_is_sunday                     NA         NA      NA       NA    
    ## is_weekend                            NA         NA      NA       NA    
    ## LDA_00                        -3.966e+05  3.188e+05  -1.244 0.213473    
    ## LDA_01                        -1.438e+06  1.156e+06  -1.243 0.213755    
    ## LDA_02                        -5.897e+05  4.742e+05  -1.244 0.213678    
    ## LDA_03                        -1.495e+06  1.202e+06  -1.244 0.213711    
    ## LDA_04                        -4.169e+05  3.352e+05  -1.244 0.213610    
    ## global_subjectivity            3.404e+02  1.923e+02   1.771 0.076700 .  
    ## global_sentiment_polarity      3.755e+02  3.372e+02   1.113 0.265576    
    ## global_rate_positive_words    -4.732e+02  2.556e+02  -1.851 0.064189 .  
    ## global_rate_negative_words     3.662e+02  3.325e+02   1.101 0.270760    
    ## rate_positive_words            3.197e+02  4.374e+02   0.731 0.464879    
    ## rate_negative_words                   NA         NA      NA       NA    
    ## avg_positive_polarity         -1.914e+01  2.822e+02  -0.068 0.945918    
    ## min_positive_polarity         -3.247e+01  1.577e+02  -0.206 0.836859    
    ## max_positive_polarity         -1.215e+02  2.164e+02  -0.562 0.574404    
    ## avg_negative_polarity          2.308e+01  3.042e+02   0.076 0.939509    
    ## min_negative_polarity         -1.114e+02  2.601e+02  -0.428 0.668648    
    ## max_negative_polarity          9.509e+01  1.915e+02   0.497 0.619516    
    ## title_subjectivity             1.344e+02  1.714e+02   0.784 0.433258    
    ## title_sentiment_polarity      -1.119e+02  1.309e+02  -0.855 0.392523    
    ## abs_title_subjectivity         1.144e+02  1.358e+02   0.842 0.399724    
    ## abs_title_sentiment_polarity   1.599e+02  1.714e+02   0.933 0.350798    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 8148 on 4890 degrees of freedom
    ## Multiple R-squared:  0.07981,    Adjusted R-squared:  0.0704 
    ## F-statistic: 8.482 on 50 and 4890 DF,  p-value: < 2.2e-16

``` r
#viewing the resulting model

pred <- predict(simon_share_lmfit, newdata = channelTest)
#Creating predictions based on the channelTest data
l2 <- postResample(pred, obs = channelTest$shares)
#creating second linear model postResample results object
l2 #viewing how well the second linear model did 
```

    ##         RMSE     Rsquared          MAE 
    ## 1.018860e+04 7.231347e-03 5.455768e+03

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
    ## 4941 samples
    ##   54 predictor
    ## 
    ## Pre-processing: centered (54), scaled (54) 
    ## Resampling: Cross-Validated (5 fold, repeated 3 times) 
    ## Summary of sample sizes: 3953, 3952, 3952, 3953, 3954, 3952, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE      Rsquared    MAE     
    ##   1                   25      8304.200  0.01205521  2999.486
    ##   1                   50      8337.694  0.01451705  3006.373
    ##   1                  100      8331.337  0.01645466  2985.411
    ##   1                  150      8351.514  0.01708823  3000.141
    ##   1                  200      8348.764  0.01784977  3000.969
    ##   2                   25      8317.117  0.01216535  2995.978
    ##   2                   50      8366.825  0.01298060  3012.196
    ##   2                  100      8430.425  0.01268539  3026.223
    ##   2                  150      8464.871  0.01176790  3037.972
    ##   2                  200      8482.093  0.01070558  3042.682
    ##   3                   25      8288.728  0.01728632  2985.655
    ##   3                   50      8362.547  0.01623233  3005.770
    ##   3                  100      8423.548  0.01489097  3033.617
    ##   3                  150      8471.002  0.01343346  3048.344
    ##   3                  200      8522.215  0.01181443  3071.012
    ##   4                   25      8313.630  0.01534112  2990.343
    ##   4                   50      8377.917  0.01638336  3016.139
    ##   4                  100      8444.750  0.01502385  3040.133
    ##   4                  150      8487.972  0.01573254  3070.453
    ##   4                  200      8569.487  0.01339700  3100.814
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## Tuning parameter 'n.minobsinnode' was
    ##  held constant at a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 25, interaction.depth = 3, shrinkage = 0.1
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
    ## 6.306592e+03 1.742281e-02 2.869811e+03

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
    ## 4941 samples
    ##   54 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold, repeated 3 times) 
    ## Summary of sample sizes: 3953, 3951, 3954, 3953, 3953, 3952, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  RMSE      Rsquared    MAE     
    ##    1    8110.656  0.03246518  2959.215
    ##    2    8105.694  0.03374430  3016.902
    ##    3    8116.899  0.03394329  3044.612
    ##    4    8153.300  0.03112732  3068.421
    ##    5    8169.887  0.03133951  3080.890
    ##    6    8203.117  0.02841825  3100.478
    ##    7    8219.494  0.02814047  3107.506
    ##    8    8231.628  0.02910897  3112.662
    ##    9    8274.790  0.02684543  3132.044
    ##   10    8282.729  0.02677560  3138.268
    ##   11    8305.773  0.02542389  3145.122
    ##   12    8320.913  0.02462961  3153.190
    ##   13    8317.124  0.02587477  3156.875
    ##   14    8350.779  0.02495553  3163.525
    ##   15    8361.444  0.02417852  3168.427
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final value used for the model was mtry = 2.

``` r
# Viewing the resulting model

pred <- predict(randomForestFit, newdata = channelTest)
#Creating predictions based on the channelTest data
rf <- postResample(pred, obs = channelTest$shares)
#creating random forest postResample results object
rf #viewing how well the random forest did
```

    ##         RMSE     Rsquared          MAE 
    ## 6.177260e+03 3.349775e-02 2.867899e+03

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
