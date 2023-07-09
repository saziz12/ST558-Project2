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
    ## 1       731              8              257           0.568             1.00                  0.671         9
    ## 2       731              8              218           0.663             1.00                  0.688        14
    ## 3       731              9             1226           0.410             1.00                  0.617        10
    ## 4       731             10             1121           0.451             1.00                  0.629        15
    ## 5       729              9              168           0.778             1.00                  0.865         6
    ## 6       729              9              100           0.760             1.00                  0.803         3
    ## # ℹ abbreviated name: ¹​n_non_stop_unique_tokens
    ## # ℹ 48 more variables: num_self_hrefs <dbl>, num_imgs <dbl>, num_videos <dbl>, average_token_length <dbl>,
    ## #   num_keywords <dbl>, data_channel_is_socmed <dbl>, kw_min_min <dbl>, kw_max_min <dbl>, kw_avg_min <dbl>,
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

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##      23    1400    2100    3732    3800  122800

The statistics above explore the spread of the `shares` variable. If the
mean is less than the median, the data is skewed to the left. If the
median is less than the mean, then the data is skewed to the right.

``` r
# contingency table for number of videos
videos_table <- table(channelTrain_summary$num_videos)
videos_table
```

    ## 
    ##    0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19   20 
    ## 1159  224  106   34   11    5    4    6    5    5   12   14    6    4    3    5    6    7    1    2    1 
    ##   23   26   27   28   31   34   58   73 
    ##    1    1    1    1    1    1    1    1

The contingency table above shows the frequency of the number of videos
in each observation of our data.

``` r
ggplot(channelTrain_summary, aes(x = n_tokens_content, y = shares)) +
geom_point() +
labs(x = "Number of words in the content", y = "Number of Shares") +
ggtitle("Shares vs. Number of words in the content")
```

![](socmed_files/figure-gfm/summaries%203-1.png)<!-- -->

``` r
# Creating ggplot of number of words in the content vs shares

ggplot(channelTrain_summary, aes(x = rate_positive_words, y = shares)) +
geom_point() +
labs(x = "Positive Word Rate", y = "Number of Shares") +
ggtitle("Shares vs. Positive Word Rate")
```

![](socmed_files/figure-gfm/summaries%203-2.png)<!-- -->

``` r
# Creating ggplot of positive word rate vs shares

ggplot(channelTrain_summary, aes(x = rate_negative_words, y = shares)) +
geom_point() +
labs(x = "Negative Word Rate", y = "Number of Shares") +
ggtitle("Shares vs. Negative Word Rate")
```

![](socmed_files/figure-gfm/summaries%203-3.png)<!-- -->

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
    ## 1 Sunday      455          1800 4449.  2550          4775  53100  6017.
    ## 2 Monday      200          1400 4384.  2400          4250  57600  6857.
    ## 3 Tuesday     238          1300 3539.  1900          3500 122800  7779.
    ## 4 Wednesday    23          1300 3399.  1900          3600  51900  4919.
    ## 5 Thursday    165          1300 3128.  2000          3500  26900  3258.
    ## 6 Friday      389          1400 4323.  2300          4200  47400  5842.
    ## 7 Saturday    837          1600 3652.  2400          3700  34500  4594.

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

![](socmed_files/figure-gfm/Bar%20Chart%20by%20Day-1.png)<!-- -->

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

![](socmed_files/figure-gfm/Histogram%20of%20Shares%20by%20Part%20of%20Week-1.png)<!-- -->

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
    ##   High Shares            642             201
    ##   Low Shares             489             296

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
    ##   High Shares          69         482                  292
    ##   Low Shares          115         429                  241

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
    ##   High Shares       634       117                 92
    ##   Low Shares        525       107                153

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

![](socmed_files/figure-gfm/Scatterplot%20for%20Links-1.png)<!-- -->

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

![](socmed_files/figure-gfm/Scatterplot%20for%20Keywords-1.png)<!-- -->

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

    ## # A tibble: 10 × 8
    ##    num_keywords   Min firstQuartile   Avg   Med thirdQuartile    max stdDev
    ##           <dbl> <dbl>         <dbl> <dbl> <dbl>         <dbl>  <dbl>  <dbl>
    ##  1            1    23          820. 1751. 1100           1675  12000  2238.
    ##  2            2   389          730. 2695. 1074.          2275  19700  4509.
    ##  3            3   200         1200  3550. 1800           3200  21300  4543.
    ##  4            4   262         1300  3401. 2100           3400  47400  5384.
    ##  5            5   238         1300  3547. 2200           3700  51900  5020.
    ##  6            6   366         1400  4318. 2300           4600  57600  5911.
    ##  7            7   165         1400  3991. 2200           3900 122800  8301.
    ##  8            8   595         1500  4299. 2100           3775  53100  6827.
    ##  9            9   414         1500  3149. 2250           3800  15200  2457.
    ## 10           10   251         1400  3476. 2100           3875  23800  3809.

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
    ## -15310  -1399     24   1354  45076 
    ## 
    ## Coefficients: (311 not defined because of singularities)
    ##                                                             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                                3.732e+03  1.297e+02  28.768  < 2e-16 ***
    ## timedelta                                                  5.856e+04  1.364e+05   0.429 0.667905    
    ## n_tokens_title                                            -1.465e+05  4.786e+04  -3.061 0.002337 ** 
    ## n_tokens_content                                          -3.267e+04  3.215e+04  -1.016 0.310161    
    ## n_unique_tokens                                            5.909e+04  3.416e+04   1.730 0.084343 .  
    ## n_non_stop_words                                          -5.548e+04  2.314e+04  -2.398 0.016907 *  
    ## n_non_stop_unique_tokens                                  -5.102e+04  2.962e+04  -1.722 0.085714 .  
    ## num_hrefs                                                  8.764e+04  2.381e+04   3.681 0.000260 ***
    ## num_self_hrefs                                            -3.265e+04  2.445e+04  -1.336 0.182337    
    ## num_imgs                                                   2.820e+05  2.120e+05   1.330 0.184162    
    ## num_videos                                                 9.215e+05  9.473e+05   0.973 0.331186    
    ## average_token_length                                       1.160e+03  1.377e+04   0.084 0.932905    
    ## num_keywords                                              -6.703e+04  1.474e+05  -0.455 0.649410    
    ## data_channel_is_socmed                                            NA         NA      NA       NA    
    ## kw_min_min                                                 1.437e+05  1.159e+05   1.240 0.215517    
    ## kw_max_min                                                -1.904e+06  8.410e+05  -2.264 0.024072 *  
    ## kw_avg_min                                                -5.069e+04  1.657e+05  -0.306 0.759788    
    ## kw_min_max                                                -9.886e+04  4.025e+04  -2.456 0.014417 *  
    ## kw_max_max                                                -5.766e+03  3.015e+04  -0.191 0.848440    
    ## kw_avg_max                                                -1.054e+04  2.684e+04  -0.393 0.694713    
    ## kw_min_avg                                                -2.444e+04  1.808e+04  -1.352 0.177084    
    ## kw_max_avg                                                 4.612e+04  5.188e+04   0.889 0.374485    
    ## kw_avg_avg                                                 1.106e+04  3.732e+04   0.296 0.767186    
    ## self_reference_min_shares                                  7.640e+04  7.180e+04   1.064 0.287912    
    ## self_reference_max_shares                                  1.305e+05  1.410e+05   0.926 0.354991    
    ## self_reference_avg_sharess                                -1.584e+05  1.344e+05  -1.179 0.239173    
    ## weekday_is_monday                                         -7.725e+03  1.876e+04  -0.412 0.680635    
    ## weekday_is_tuesday                                        -5.540e+03  2.102e+04  -0.264 0.792199    
    ## weekday_is_wednesday                                      -1.251e+04  1.994e+04  -0.627 0.530734    
    ## weekday_is_thursday                                       -7.217e+03  2.053e+04  -0.352 0.725360    
    ## weekday_is_friday                                         -7.035e+03  1.974e+04  -0.356 0.721725    
    ## weekday_is_saturday                                        1.092e+04  1.717e+04   0.636 0.525003    
    ## weekday_is_sunday                                                 NA         NA      NA       NA    
    ## is_weekend                                                        NA         NA      NA       NA    
    ## LDA_00                                                     1.853e+04  1.479e+04   1.253 0.210972    
    ## LDA_01                                                    -5.164e+03  1.287e+04  -0.401 0.688484    
    ## LDA_02                                                     2.113e+04  1.507e+04   1.402 0.161557    
    ## LDA_03                                                    -1.382e+04  1.546e+04  -0.894 0.371956    
    ## LDA_04                                                            NA         NA      NA       NA    
    ## global_subjectivity                                        1.378e+04  1.267e+04   1.087 0.277450    
    ## global_sentiment_polarity                                  1.394e+03  2.673e+04   0.052 0.958434    
    ## global_rate_positive_words                                -6.742e+03  2.495e+04  -0.270 0.787080    
    ## global_rate_negative_words                                 2.822e+03  2.534e+04   0.111 0.911361    
    ## rate_positive_words                                       -1.807e+04  2.785e+04  -0.649 0.516687    
    ## rate_negative_words                                       -1.089e+04  6.220e+03  -1.750 0.080759 .  
    ## avg_positive_polarity                                     -1.997e+04  2.239e+04  -0.892 0.373001    
    ## min_positive_polarity                                     -1.299e+04  1.238e+04  -1.049 0.294799    
    ## max_positive_polarity                                      6.655e+02  1.649e+04   0.040 0.967827    
    ## avg_negative_polarity                                     -5.922e+03  3.022e+04  -0.196 0.844720    
    ## min_negative_polarity                                      1.533e+04  2.811e+04   0.545 0.585893    
    ## max_negative_polarity                                     -6.799e+03  1.926e+04  -0.353 0.724241    
    ## title_subjectivity                                        -1.648e+04  1.939e+04  -0.850 0.395739    
    ## title_sentiment_polarity                                  -6.312e+03  2.004e+04  -0.315 0.752903    
    ## abs_title_subjectivity                                     6.117e+03  1.291e+04   0.474 0.635954    
    ## abs_title_sentiment_polarity                               4.750e+03  2.436e+04   0.195 0.845456    
    ## `timedelta:n_tokens_title`                                 2.580e+02  2.265e+03   0.114 0.909370    
    ## `timedelta:n_tokens_content`                               7.505e+03  3.830e+03   1.959 0.050681 .  
    ## `timedelta:n_unique_tokens`                                4.109e+02  9.427e+03   0.044 0.965249    
    ## `timedelta:n_non_stop_words`                              -6.105e+04  1.368e+05  -0.446 0.655616    
    ## `timedelta:n_non_stop_unique_tokens`                       9.812e+03  1.039e+04   0.944 0.345462    
    ## `timedelta:num_hrefs`                                     -3.515e+03  3.231e+03  -1.088 0.277148    
    ## `timedelta:num_self_hrefs`                                 2.808e+03  3.595e+03   0.781 0.435264    
    ## `timedelta:num_imgs`                                      -3.328e+03  2.347e+03  -1.418 0.156979    
    ## `timedelta:num_videos`                                     2.110e+03  1.641e+03   1.286 0.199241    
    ## `timedelta:average_token_length`                           1.117e+04  9.697e+03   1.152 0.249802    
    ## `timedelta:num_keywords`                                   2.623e+03  2.386e+03   1.099 0.272206    
    ## `timedelta:data_channel_is_socmed`                                NA         NA      NA       NA    
    ## `timedelta:kw_min_min`                                    -3.651e+03  2.072e+04  -0.176 0.860208    
    ## `timedelta:kw_max_min`                                     9.874e+03  1.351e+04   0.731 0.465372    
    ## `timedelta:kw_avg_min`                                    -9.036e+03  1.653e+04  -0.547 0.584906    
    ## `timedelta:kw_min_max`                                     5.314e+03  2.887e+03   1.841 0.066276 .  
    ## `timedelta:kw_max_max`                                    -1.613e+04  1.336e+04  -1.207 0.228008    
    ## `timedelta:kw_avg_max`                                    -2.409e+02  1.938e+03  -0.124 0.901149    
    ## `timedelta:kw_min_avg`                                    -4.240e+02  2.050e+03  -0.207 0.836234    
    ## `timedelta:kw_max_avg`                                    -3.563e+03  5.689e+03  -0.626 0.531496    
    ## `timedelta:kw_avg_avg`                                    -1.248e+03  5.551e+03  -0.225 0.822233    
    ## `timedelta:self_reference_min_shares`                      4.702e+03  6.210e+03   0.757 0.449396    
    ## `timedelta:self_reference_max_shares`                      4.727e+03  1.235e+04   0.383 0.702041    
    ## `timedelta:self_reference_avg_sharess`                    -7.737e+03  1.179e+04  -0.656 0.511944    
    ## `timedelta:weekday_is_monday`                              2.252e+03  1.966e+03   1.145 0.252653    
    ## `timedelta:weekday_is_tuesday`                             1.734e+02  2.124e+03   0.082 0.934981    
    ## `timedelta:weekday_is_wednesday`                           1.236e+03  1.999e+03   0.619 0.536532    
    ## `timedelta:weekday_is_thursday`                            2.187e+03  2.097e+03   1.043 0.297518    
    ## `timedelta:weekday_is_friday`                              1.147e+02  1.918e+03   0.060 0.952365    
    ## `timedelta:weekday_is_saturday`                            1.068e+02  1.898e+03   0.056 0.955171    
    ## `timedelta:weekday_is_sunday`                                     NA         NA      NA       NA    
    ## `timedelta:is_weekend`                                            NA         NA      NA       NA    
    ## `timedelta:LDA_00`                                        -6.509e+02  1.950e+03  -0.334 0.738705    
    ## `timedelta:LDA_01`                                         3.204e+02  1.491e+03   0.215 0.829878    
    ## `timedelta:LDA_02`                                        -2.371e+03  1.535e+03  -1.545 0.123125    
    ## `timedelta:LDA_03`                                         1.121e+03  1.767e+03   0.634 0.526171    
    ## `timedelta:LDA_04`                                                NA         NA      NA       NA    
    ## `timedelta:global_subjectivity`                            3.619e+02  3.760e+03   0.096 0.923358    
    ## `timedelta:global_sentiment_polarity`                      9.405e+03  3.851e+03   2.442 0.014978 *  
    ## `timedelta:global_rate_positive_words`                    -7.604e+03  3.373e+03  -2.254 0.024660 *  
    ## `timedelta:global_rate_negative_words`                     3.612e+03  3.984e+03   0.906 0.365187    
    ## `timedelta:rate_positive_words`                            7.338e+03  1.020e+04   0.720 0.472109    
    ## `timedelta:rate_negative_words`                                   NA         NA      NA       NA    
    ## `timedelta:avg_positive_polarity`                         -1.580e+04  5.672e+03  -2.785 0.005577 ** 
    ## `timedelta:min_positive_polarity`                          2.279e+03  2.050e+03   1.112 0.266821    
    ## `timedelta:max_positive_polarity`                          1.624e+03  3.223e+03   0.504 0.614485    
    ## `timedelta:avg_negative_polarity`                         -1.925e+03  4.502e+03  -0.428 0.669185    
    ## `timedelta:min_negative_polarity`                         -3.057e+03  3.402e+03  -0.899 0.369324    
    ## `timedelta:max_negative_polarity`                          3.341e+03  2.795e+03   1.196 0.232508    
    ## `timedelta:title_subjectivity`                            -9.075e+02  1.961e+03  -0.463 0.643690    
    ## `timedelta:title_sentiment_polarity`                       8.217e+02  2.208e+03   0.372 0.709921    
    ## `timedelta:abs_title_subjectivity`                        -8.172e+02  1.646e+03  -0.497 0.619740    
    ## `timedelta:abs_title_sentiment_polarity`                   7.517e+02  2.890e+03   0.260 0.794915    
    ## `n_tokens_title:n_tokens_content`                          1.014e+04  3.487e+03   2.908 0.003817 ** 
    ## `n_tokens_title:n_unique_tokens`                           9.621e+03  7.096e+03   1.356 0.175863    
    ## `n_tokens_title:n_non_stop_words`                          1.615e+05  5.096e+04   3.169 0.001632 ** 
    ## `n_tokens_title:n_non_stop_unique_tokens`                 -6.912e+03  7.249e+03  -0.954 0.340779    
    ## `n_tokens_title:num_hrefs`                                -6.369e+03  3.587e+03  -1.775 0.076521 .  
    ## `n_tokens_title:num_self_hrefs`                            2.445e+03  3.145e+03   0.777 0.437374    
    ## `n_tokens_title:num_imgs`                                 -4.152e+03  2.755e+03  -1.507 0.132487    
    ## `n_tokens_title:num_videos`                               -2.640e+03  2.805e+03  -0.941 0.347122    
    ## `n_tokens_title:average_token_length`                     -8.694e+03  6.569e+03  -1.323 0.186351    
    ## `n_tokens_title:num_keywords`                             -2.668e+03  2.185e+03  -1.221 0.222784    
    ## `n_tokens_title:data_channel_is_socmed`                           NA         NA      NA       NA    
    ## `n_tokens_title:kw_min_min`                               -1.097e+03  3.474e+03  -0.316 0.752246    
    ## `n_tokens_title:kw_max_min`                               -4.562e+04  2.086e+04  -2.187 0.029262 *  
    ## `n_tokens_title:kw_avg_min`                                5.957e+04  2.537e+04   2.348 0.019311 *  
    ## `n_tokens_title:kw_min_max`                                5.589e+03  3.760e+03   1.487 0.137830    
    ## `n_tokens_title:kw_max_max`                                1.486e+03  4.460e+03   0.333 0.739094    
    ## `n_tokens_title:kw_avg_max`                                1.678e+03  3.297e+03   0.509 0.611063    
    ## `n_tokens_title:kw_min_avg`                               -3.494e+03  2.569e+03  -1.360 0.174499    
    ## `n_tokens_title:kw_max_avg`                               -5.687e+03  8.422e+03  -0.675 0.499872    
    ## `n_tokens_title:kw_avg_avg`                                8.008e+03  6.408e+03   1.250 0.212053    
    ## `n_tokens_title:self_reference_min_shares`                 2.768e+03  7.165e+03   0.386 0.699399    
    ## `n_tokens_title:self_reference_max_shares`                 1.177e+04  1.210e+04   0.972 0.331460    
    ## `n_tokens_title:self_reference_avg_sharess`               -1.211e+04  1.240e+04  -0.977 0.329071    
    ## `n_tokens_title:weekday_is_monday`                         5.861e+03  2.537e+03   2.310 0.021315 *  
    ## `n_tokens_title:weekday_is_tuesday`                        8.080e+03  2.846e+03   2.839 0.004724 ** 
    ## `n_tokens_title:weekday_is_wednesday`                      4.399e+03  2.650e+03   1.660 0.097600 .  
    ## `n_tokens_title:weekday_is_thursday`                       6.266e+03  2.678e+03   2.340 0.019731 *  
    ## `n_tokens_title:weekday_is_friday`                         3.883e+03  2.525e+03   1.538 0.124796    
    ## `n_tokens_title:weekday_is_saturday`                       3.932e+03  2.309e+03   1.703 0.089220 .  
    ## `n_tokens_title:weekday_is_sunday`                                NA         NA      NA       NA    
    ## `n_tokens_title:is_weekend`                                       NA         NA      NA       NA    
    ## `n_tokens_title:LDA_00`                                   -4.204e+03  2.220e+03  -1.893 0.058960 .  
    ## `n_tokens_title:LDA_01`                                   -1.933e+03  1.870e+03  -1.034 0.301887    
    ## `n_tokens_title:LDA_02`                                   -4.190e+03  2.170e+03  -1.931 0.054137 .  
    ## `n_tokens_title:LDA_03`                                   -4.093e+03  2.454e+03  -1.668 0.096072 .  
    ## `n_tokens_title:LDA_04`                                           NA         NA      NA       NA    
    ## `n_tokens_title:global_subjectivity`                      -1.134e+03  2.904e+03  -0.390 0.696376    
    ## `n_tokens_title:global_sentiment_polarity`                -6.301e+03  4.353e+03  -1.447 0.148488    
    ## `n_tokens_title:global_rate_positive_words`                6.460e+03  3.320e+03   1.946 0.052320 .  
    ## `n_tokens_title:global_rate_negative_words`               -9.297e+03  4.381e+03  -2.122 0.034357 *  
    ## `n_tokens_title:rate_positive_words`                      -7.702e+03  7.037e+03  -1.094 0.274325    
    ## `n_tokens_title:rate_negative_words`                              NA         NA      NA       NA    
    ## `n_tokens_title:avg_positive_polarity`                     5.878e+03  4.109e+03   1.430 0.153276    
    ## `n_tokens_title:min_positive_polarity`                     1.229e+03  2.081e+03   0.591 0.555075    
    ## `n_tokens_title:max_positive_polarity`                     1.655e+02  2.619e+03   0.063 0.949654    
    ## `n_tokens_title:avg_negative_polarity`                    -4.930e+03  4.380e+03  -1.126 0.260948    
    ## `n_tokens_title:min_negative_polarity`                     1.916e+03  3.212e+03   0.596 0.551257    
    ## `n_tokens_title:max_negative_polarity`                     3.175e+03  2.703e+03   1.174 0.240848    
    ## `n_tokens_title:title_subjectivity`                       -5.697e+02  2.515e+03  -0.226 0.820927    
    ## `n_tokens_title:title_sentiment_polarity`                 -1.076e+03  2.739e+03  -0.393 0.694714    
    ## `n_tokens_title:abs_title_subjectivity`                   -2.458e+03  1.734e+03  -1.417 0.157068    
    ## `n_tokens_title:abs_title_sentiment_polarity`              9.279e+01  3.364e+03   0.028 0.978004    
    ## `n_tokens_content:n_unique_tokens`                        -1.042e+04  7.558e+03  -1.378 0.168844    
    ## `n_tokens_content:n_non_stop_words`                               NA         NA      NA       NA    
    ## `n_tokens_content:n_non_stop_unique_tokens`                1.211e+04  1.069e+04   1.133 0.257910    
    ## `n_tokens_content:num_hrefs`                               7.499e+03  4.134e+03   1.814 0.070302 .  
    ## `n_tokens_content:num_self_hrefs`                          4.808e+03  4.556e+03   1.056 0.291757    
    ## `n_tokens_content:num_imgs`                               -4.592e+03  3.555e+03  -1.292 0.197034    
    ## `n_tokens_content:num_videos`                              8.246e+03  4.103e+03   2.010 0.045049 *  
    ## `n_tokens_content:average_token_length`                    2.127e+04  1.781e+04   1.194 0.233055    
    ## `n_tokens_content:num_keywords`                           -4.702e+03  3.707e+03  -1.269 0.205245    
    ## `n_tokens_content:data_channel_is_socmed`                         NA         NA      NA       NA    
    ## `n_tokens_content:kw_min_min`                             -3.917e+03  2.442e+03  -1.604 0.109382    
    ## `n_tokens_content:kw_max_min`                              9.723e+03  9.641e+03   1.009 0.313745    
    ## `n_tokens_content:kw_avg_min`                             -9.339e+03  1.077e+04  -0.867 0.386280    
    ## `n_tokens_content:kw_min_max`                             -2.080e+03  2.845e+03  -0.731 0.464954    
    ## `n_tokens_content:kw_max_max`                             -8.480e+02  5.908e+03  -0.144 0.885933    
    ## `n_tokens_content:kw_avg_max`                              1.661e+03  3.800e+03   0.437 0.662346    
    ## `n_tokens_content:kw_min_avg`                             -2.325e+03  2.106e+03  -1.104 0.270171    
    ## `n_tokens_content:kw_max_avg`                             -3.400e+03  5.716e+03  -0.595 0.552320    
    ## `n_tokens_content:kw_avg_avg`                             -1.049e+02  8.114e+03  -0.013 0.989693    
    ## `n_tokens_content:self_reference_min_shares`              -1.766e+03  4.842e+03  -0.365 0.715409    
    ## `n_tokens_content:self_reference_max_shares`              -1.289e+04  1.222e+04  -1.055 0.291948    
    ## `n_tokens_content:self_reference_avg_sharess`              3.499e+03  8.788e+03   0.398 0.690710    
    ## `n_tokens_content:weekday_is_monday`                      -1.664e+03  1.708e+03  -0.974 0.330446    
    ## `n_tokens_content:weekday_is_tuesday`                      7.105e+02  2.230e+03   0.319 0.750131    
    ## `n_tokens_content:weekday_is_wednesday`                   -2.636e+03  1.698e+03  -1.552 0.121273    
    ## `n_tokens_content:weekday_is_thursday`                    -2.774e+03  2.043e+03  -1.358 0.175239    
    ## `n_tokens_content:weekday_is_friday`                      -1.965e+03  1.534e+03  -1.281 0.200711    
    ## `n_tokens_content:weekday_is_saturday`                    -2.706e+03  1.999e+03  -1.354 0.176376    
    ## `n_tokens_content:weekday_is_sunday`                              NA         NA      NA       NA    
    ## `n_tokens_content:is_weekend`                                     NA         NA      NA       NA    
    ## `n_tokens_content:LDA_00`                                 -2.183e+03  2.028e+03  -1.077 0.282238    
    ## `n_tokens_content:LDA_01`                                 -3.898e+03  1.506e+03  -2.588 0.009958 ** 
    ## `n_tokens_content:LDA_02`                                 -9.502e+03  3.152e+03  -3.014 0.002722 ** 
    ## `n_tokens_content:LDA_03`                                 -7.219e+02  1.648e+03  -0.438 0.661619    
    ## `n_tokens_content:LDA_04`                                         NA         NA      NA       NA    
    ## `n_tokens_content:global_subjectivity`                    -6.273e+03  8.021e+03  -0.782 0.434630    
    ## `n_tokens_content:global_sentiment_polarity`              -6.681e+03  6.210e+03  -1.076 0.282609    
    ## `n_tokens_content:global_rate_positive_words`             -4.675e+03  8.147e+03  -0.574 0.566323    
    ## `n_tokens_content:global_rate_negative_words`              1.131e+04  8.510e+03   1.329 0.184485    
    ## `n_tokens_content:rate_positive_words`                     2.506e+04  2.270e+04   1.104 0.270204    
    ##  [ reached getOption("max.print") -- omitted 1286 rows ]
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 5234 on 453 degrees of freedom
    ## Multiple R-squared:  0.7734, Adjusted R-squared:  0.186 
    ## F-statistic: 1.317 on 1174 and 453 DF,  p-value: 0.0003005

``` r
#viewing the resulting model

pred <- predict(simon_share_lmfit, newdata = channelTest)
#Creating predictions based on the channelTest data
l1 <- postResample(pred, obs = channelTest$shares)
#creating first linear model postResample results object
l1 #viewing how well the first linear model did 
```

    ##         RMSE     Rsquared          MAE 
    ## 3.197659e+05 7.982401e-03 3.076566e+04

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
    ## -10755  -2129   -929    467 116833 
    ## 
    ## Coefficients: (4 not defined because of singularities)
    ##                              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                   3731.60     139.11  26.825  < 2e-16 ***
    ## timedelta                      423.35     228.53   1.853  0.06414 .  
    ## n_tokens_title                 105.23     147.93   0.711  0.47697    
    ## n_tokens_content               372.54     296.61   1.256  0.20930    
    ## n_unique_tokens               -539.67     546.31  -0.988  0.32338    
    ## n_non_stop_words                63.98     475.68   0.135  0.89302    
    ## n_non_stop_unique_tokens       -31.40     469.28  -0.067  0.94666    
    ## num_hrefs                     -607.11     255.10  -2.380  0.01744 *  
    ## num_self_hrefs                -226.16     201.65  -1.122  0.26221    
    ## num_imgs                      -217.86     204.70  -1.064  0.28737    
    ## num_videos                     290.39     152.86   1.900  0.05765 .  
    ## average_token_length            64.50     259.56   0.249  0.80377    
    ## num_keywords                   238.23     187.63   1.270  0.20438    
    ## data_channel_is_socmed             NA         NA      NA       NA    
    ## kw_min_min                     295.30     281.44   1.049  0.29422    
    ## kw_max_min                   -1636.89     749.23  -2.185  0.02905 *  
    ## kw_avg_min                    1503.47     730.11   2.059  0.03963 *  
    ## kw_min_max                    -344.65     247.13  -1.395  0.16333    
    ## kw_max_max                      63.71     310.90   0.205  0.83765    
    ## kw_avg_max                     135.28     340.59   0.397  0.69129    
    ## kw_min_avg                     -18.31     250.19  -0.073  0.94166    
    ## kw_max_avg                    -881.80     578.09  -1.525  0.12737    
    ## kw_avg_avg                    1543.07     502.97   3.068  0.00219 ** 
    ## self_reference_min_shares      324.53     306.84   1.058  0.29037    
    ## self_reference_max_shares     -117.17     256.98  -0.456  0.64850    
    ## self_reference_avg_sharess     430.60     407.87   1.056  0.29125    
    ## weekday_is_monday              111.72     237.75   0.470  0.63848    
    ## weekday_is_tuesday            -177.27     254.46  -0.697  0.48612    
    ## weekday_is_wednesday          -256.94     251.82  -1.020  0.30773    
    ## weekday_is_thursday           -361.43     260.93  -1.385  0.16620    
    ## weekday_is_friday              106.28     237.38   0.448  0.65442    
    ## weekday_is_saturday            -92.01     208.31  -0.442  0.65877    
    ## weekday_is_sunday                  NA         NA      NA       NA    
    ## is_weekend                         NA         NA      NA       NA    
    ## LDA_00                         165.93     235.12   0.706  0.48047    
    ## LDA_01                        -237.10     167.40  -1.416  0.15686    
    ## LDA_02                        -208.69     218.30  -0.956  0.33923    
    ## LDA_03                        -260.06     229.34  -1.134  0.25699    
    ## LDA_04                             NA         NA      NA       NA    
    ## global_subjectivity             33.77     203.60   0.166  0.86827    
    ## global_sentiment_polarity      294.84     371.15   0.794  0.42709    
    ## global_rate_positive_words     -81.87     271.33  -0.302  0.76288    
    ## global_rate_negative_words    -240.56     375.97  -0.640  0.52238    
    ## rate_positive_words            184.81     828.46   0.223  0.82351    
    ## rate_negative_words            708.01     821.29   0.862  0.38878    
    ## avg_positive_polarity         -227.40     327.58  -0.694  0.48767    
    ## min_positive_polarity         -147.03     196.37  -0.749  0.45414    
    ## max_positive_polarity         -185.80     228.85  -0.812  0.41697    
    ## avg_negative_polarity          300.43     399.42   0.752  0.45206    
    ## min_negative_polarity         -518.24     343.71  -1.508  0.13181    
    ## max_negative_polarity          -82.02     264.03  -0.311  0.75612    
    ## title_subjectivity             137.94     225.37   0.612  0.54058    
    ## title_sentiment_polarity      -270.76     190.49  -1.421  0.15542    
    ## abs_title_subjectivity         108.97     167.12   0.652  0.51445    
    ## abs_title_sentiment_polarity   457.93     252.20   1.816  0.06959 .  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 5613 on 1577 degrees of freedom
    ## Multiple R-squared:  0.09263,    Adjusted R-squared:  0.06386 
    ## F-statistic:  3.22 on 50 and 1577 DF,  p-value: 9.122e-13

``` r
#viewing the resulting model

pred <- predict(simon_share_lmfit, newdata = channelTest)
#Creating predictions based on the channelTest data
l2 <- postResample(pred, obs = channelTest$shares)
#creating second linear model postResample results object
l2 #viewing how well the second linear model did 
```

    ##         RMSE     Rsquared          MAE 
    ## 3.197659e+05 7.982401e-03 3.076566e+04

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
    ## 1628 samples
    ##   54 predictor
    ## 
    ## Pre-processing: centered (54), scaled (54) 
    ## Resampling: Cross-Validated (5 fold, repeated 3 times) 
    ## Summary of sample sizes: 1302, 1302, 1303, 1303, 1302, 1302, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE      Rsquared    MAE     
    ##   1                   25      5600.286  0.01721500  2769.666
    ##   1                   50      5575.814  0.03094730  2733.272
    ##   1                  100      5588.235  0.03448102  2743.593
    ##   1                  150      5589.196  0.03831177  2747.229
    ##   1                  200      5592.447  0.04249062  2754.115
    ##   2                   25      5577.207  0.02878350  2744.799
    ##   2                   50      5562.439  0.04372176  2728.478
    ##   2                  100      5581.870  0.05147024  2757.076
    ##   2                  150      5625.866  0.05106374  2793.814
    ##   2                  200      5645.154  0.05180840  2835.115
    ##   3                   25      5564.512  0.03780802  2744.377
    ##   3                   50      5579.215  0.04844939  2745.669
    ##   3                  100      5638.335  0.05041897  2810.896
    ##   3                  150      5660.108  0.05338134  2861.958
    ##   3                  200      5696.204  0.05275146  2913.202
    ##   4                   25      5572.226  0.03766567  2748.235
    ##   4                   50      5605.316  0.04657562  2759.325
    ##   4                  100      5657.118  0.05239106  2839.013
    ##   4                  150      5683.805  0.05412978  2887.105
    ##   4                  200      5737.495  0.05122277  2941.728
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## Tuning parameter 'n.minobsinnode' was
    ##  held constant at a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 50, interaction.depth = 2, shrinkage = 0.1
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
    ## 4.643127e+03 7.092805e-02 2.297843e+03

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
    ## 1628 samples
    ##   54 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold, repeated 3 times) 
    ## Summary of sample sizes: 1302, 1303, 1302, 1302, 1303, 1303, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  RMSE      Rsquared    MAE     
    ##    1    5409.924  0.08289530  2710.272
    ##    2    5389.839  0.08109586  2734.821
    ##    3    5393.292  0.08187139  2743.353
    ##    4    5406.748  0.07925796  2753.625
    ##    5    5420.974  0.07495030  2767.173
    ##    6    5448.607  0.07041340  2779.503
    ##    7    5450.627  0.07044360  2782.800
    ##    8    5483.106  0.06518744  2803.303
    ##    9    5477.507  0.06739657  2797.749
    ##   10    5485.534  0.06714406  2803.624
    ##   11    5501.662  0.06351856  2810.829
    ##   12    5529.106  0.06134008  2817.143
    ##   13    5532.457  0.06033490  2820.672
    ##   14    5568.311  0.05583149  2830.534
    ##   15    5561.845  0.05814492  2830.525
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
    ## 4.623452e+03 8.841557e-02 2.396884e+03

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
