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
    ## 1       731             10              231           0.636             1.00                  0.797         4
    ## 2       731              9             1248           0.490             1.00                  0.732        11
    ## 3       731             12              682           0.460             1.00                  0.635        10
    ## 4       731              9              391           0.510             1.00                  0.650         9
    ## 5       731             11              125           0.675             1.00                  0.797         1
    ## 6       731             11              799           0.504             1.00                  0.738         8
    ## # ℹ abbreviated name: ¹​n_non_stop_unique_tokens
    ## # ℹ 48 more variables: num_self_hrefs <dbl>, num_imgs <dbl>, num_videos <dbl>, average_token_length <dbl>,
    ## #   num_keywords <dbl>, data_channel_is_world <dbl>, kw_min_min <dbl>, kw_max_min <dbl>, kw_avg_min <dbl>,
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
    ##      41     827    1100    2209    1900  141400

The statistics above explore the spread of the `shares` variable. If the
mean is less than the median, the data is skewed to the left. If the
median is less than the mean, then the data is skewed to the right.

``` r
# contingency table for number of videos
videos_table <- table(channelTrain_summary$num_videos)
videos_table
```

    ## 
    ##    0    1    2    3    4    5    6    7    8    9   10   11   13   14   15   16   17   18   20   21   23 
    ## 4026 1328  312  104   42   22   13    8    7    7   13    5    2    2    1    1    1    1    2    2    1

The contingency table above shows the frequency of the number of videos
in each observation of our data.

``` r
ggplot(channelTrain_summary, aes(x = n_tokens_content, y = shares)) +
geom_point() +
labs(x = "Number of words in the content", y = "Number of Shares") +
ggtitle("Shares vs. Number of words in the content")
```

![](world_files/figure-gfm/summaries%203-1.png)<!-- -->

``` r
# Creating ggplot of number of words in the content vs shares

ggplot(channelTrain_summary, aes(x = rate_positive_words, y = shares)) +
geom_point() +
labs(x = "Positive Word Rate", y = "Number of Shares") +
ggtitle("Shares vs. Positive Word Rate")
```

![](world_files/figure-gfm/summaries%203-2.png)<!-- -->

``` r
# Creating ggplot of positive word rate vs shares

ggplot(channelTrain_summary, aes(x = rate_negative_words, y = shares)) +
geom_point() +
labs(x = "Negative Word Rate", y = "Number of Shares") +
ggtitle("Shares vs. Negative Word Rate")
```

![](world_files/figure-gfm/summaries%203-3.png)<!-- -->

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
    ## 1 Sunday       91         1100  2754.  1400          2300  55600  5045.
    ## 2 Monday       43          845. 2485.  1100          1800 141400  7552.
    ## 3 Tuesday      45          768. 2101.  1100          1700  84800  4535.
    ## 4 Wednesday    48          781. 1908.  1100          1700  53500  3322.
    ## 5 Thursday     41          776. 2293.  1100          1700  67700  5012.
    ## 6 Friday       70          853  2032.  1100          1900  64300  3716.
    ## 7 Saturday     43         1000  2350.  1500          2475  39700  3128.

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

![](world_files/figure-gfm/Bar%20Chart%20by%20Day-1.png)<!-- -->

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

![](world_files/figure-gfm/Histogram%20of%20Shares%20by%20Part%20of%20Week-1.png)<!-- -->

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
    ##   High Shares           2819             486
    ##   Low Shares            2266             329

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
    ##   High Shares         369        1710                 1226
    ##   Low Shares          253        1657                  685

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
    ##   High Shares      2211       740                354
    ##   Low Shares       1815       588                192

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

![](world_files/figure-gfm/Scatterplot%20for%20Links-1.png)<!-- -->

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

![](world_files/figure-gfm/Scatterplot%20for%20Keywords-1.png)<!-- -->

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

    ## # A tibble: 8 × 8
    ##   num_keywords   Min firstQuartile   Avg   Med thirdQuartile    max stdDev
    ##          <dbl> <dbl>         <dbl> <dbl> <dbl>         <dbl>  <dbl>  <dbl>
    ## 1            3    43          883. 2429.  1100          1775  48000  5557.
    ## 2            4    48          798. 1780.  1100          1700  19700  2483.
    ## 3            5    43          850. 1885.  1100          1700  27800  2628.
    ## 4            6    42          815  2197.  1100          1600 141400  6370.
    ## 5            7   129          850. 2265.  1100          1900  53500  4712.
    ## 6            8    45          815  2101.  1100          1900  64300  3699.
    ## 7            9    57          800. 2249.  1200          1900  84800  5143.
    ## 8           10    41          829  2575.  1200          2100 108400  5792.

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
    ## -11009  -1662   -302    993 128623 
    ## 
    ## Coefficients: (291 not defined because of singularities)
    ##                                                             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                                2.209e+03  5.973e+01  36.985  < 2e-16 ***
    ## timedelta                                                  1.214e+03  3.795e+03   0.320 0.749128    
    ## n_tokens_title                                            -1.282e+03  1.313e+03  -0.976 0.328968    
    ## n_tokens_content                                          -4.148e+03  5.423e+03  -0.765 0.444375    
    ## n_unique_tokens                                           -4.204e+03  9.559e+03  -0.440 0.660099    
    ## n_non_stop_words                                          -7.893e+03  8.711e+03  -0.906 0.364957    
    ## n_non_stop_unique_tokens                                   1.602e+03  9.576e+03   0.167 0.867129    
    ## num_hrefs                                                  1.367e+03  3.733e+03   0.366 0.714307    
    ## num_self_hrefs                                            -5.999e+03  3.319e+03  -1.807 0.070773 .  
    ## num_imgs                                                   4.237e+03  1.814e+03   2.336 0.019554 *  
    ## num_videos                                                -7.916e+01  1.862e+03  -0.043 0.966089    
    ## average_token_length                                       7.894e+03  7.206e+03   1.095 0.273359    
    ## num_keywords                                               1.279e+03  1.561e+03   0.819 0.412739    
    ## data_channel_is_world                                             NA         NA      NA       NA    
    ## kw_min_min                                                 5.034e+02  4.254e+03   0.118 0.905817    
    ## kw_max_min                                                 4.772e+02  7.518e+03   0.063 0.949387    
    ## kw_avg_min                                                -1.861e+03  6.613e+03  -0.281 0.778448    
    ## kw_min_max                                                -1.345e+03  3.530e+03  -0.381 0.703136    
    ## kw_max_max                                                -9.506e+02  3.458e+03  -0.275 0.783385    
    ## kw_avg_max                                                 1.735e+03  2.534e+03   0.685 0.493607    
    ## kw_min_avg                                                 2.046e+03  2.136e+03   0.958 0.338062    
    ## kw_max_avg                                                 5.871e+03  4.068e+03   1.443 0.149029    
    ## kw_avg_avg                                                -6.843e+03  3.683e+03  -1.858 0.063244 .  
    ## self_reference_min_shares                                 -1.180e+04  4.404e+04  -0.268 0.788780    
    ## self_reference_max_shares                                  1.890e+04  4.962e+04   0.381 0.703325    
    ## self_reference_avg_sharess                                 7.122e+02  7.916e+04   0.009 0.992822    
    ## weekday_is_monday                                         -8.322e+02  2.354e+03  -0.353 0.723729    
    ## weekday_is_tuesday                                         6.392e+02  2.471e+03   0.259 0.795901    
    ## weekday_is_wednesday                                       7.753e+02  2.526e+03   0.307 0.758881    
    ## weekday_is_thursday                                       -2.742e+02  2.499e+03  -0.110 0.912619    
    ## weekday_is_friday                                          1.127e+03  2.359e+03   0.478 0.632950    
    ## weekday_is_saturday                                       -4.590e+02  1.890e+03  -0.243 0.808099    
    ## weekday_is_sunday                                                 NA         NA      NA       NA    
    ## is_weekend                                                        NA         NA      NA       NA    
    ## LDA_00                                                    -2.171e+03  1.495e+03  -1.452 0.146481    
    ## LDA_01                                                    -1.786e+03  1.418e+03  -1.260 0.207874    
    ## LDA_02                                                    -1.148e+03  1.717e+03  -0.668 0.503909    
    ## LDA_03                                                    -3.646e+03  1.568e+03  -2.325 0.020088 *  
    ## LDA_04                                                            NA         NA      NA       NA    
    ## global_subjectivity                                        1.413e+02  3.808e+03   0.037 0.970397    
    ## global_sentiment_polarity                                 -7.360e+02  6.758e+03  -0.109 0.913279    
    ## global_rate_positive_words                                 7.677e+03  6.558e+03   1.171 0.241808    
    ## global_rate_negative_words                                 1.240e+03  5.892e+03   0.210 0.833340    
    ## rate_positive_words                                        3.908e+03  8.906e+03   0.439 0.660785    
    ## rate_negative_words                                               NA         NA      NA       NA    
    ## avg_positive_polarity                                      2.409e+03  5.714e+03   0.422 0.673289    
    ## min_positive_polarity                                      4.524e+02  3.204e+03   0.141 0.887701    
    ## max_positive_polarity                                     -2.485e+03  4.402e+03  -0.565 0.572399    
    ## avg_negative_polarity                                     -1.545e+03  6.277e+03  -0.246 0.805655    
    ## min_negative_polarity                                     -1.679e+03  5.402e+03  -0.311 0.756037    
    ## max_negative_polarity                                      2.889e+03  3.706e+03   0.779 0.435749    
    ## title_subjectivity                                         4.697e+03  2.164e+03   2.171 0.030002 *  
    ## title_sentiment_polarity                                   8.117e+02  1.431e+03   0.567 0.570595    
    ## abs_title_subjectivity                                     8.142e+02  1.523e+03   0.535 0.593018    
    ## abs_title_sentiment_polarity                              -2.522e+03  2.149e+03  -1.174 0.240567    
    ## `timedelta:n_tokens_title`                                 2.701e+02  5.067e+02   0.533 0.594058    
    ## `timedelta:n_tokens_content`                               3.754e+02  3.919e+02   0.958 0.338233    
    ## `timedelta:n_unique_tokens`                                1.781e+02  2.294e+03   0.078 0.938122    
    ## `timedelta:n_non_stop_words`                               5.425e+03  3.314e+03   1.637 0.101651    
    ## `timedelta:n_non_stop_unique_tokens`                       1.623e+03  2.410e+03   0.673 0.500810    
    ## `timedelta:num_hrefs`                                      2.808e+01  2.490e+02   0.113 0.910201    
    ## `timedelta:num_self_hrefs`                                -5.150e+02  2.569e+02  -2.004 0.045087 *  
    ## `timedelta:num_imgs`                                       4.377e+02  2.580e+02   1.696 0.089919 .  
    ## `timedelta:num_videos`                                     2.461e+02  2.474e+02   0.995 0.319763    
    ## `timedelta:average_token_length`                          -3.326e+03  2.378e+03  -1.399 0.162023    
    ## `timedelta:num_keywords`                                  -3.795e+02  5.710e+02  -0.665 0.506263    
    ## `timedelta:data_channel_is_world`                                 NA         NA      NA       NA    
    ## `timedelta:kw_min_min`                                    -1.216e+03  3.610e+03  -0.337 0.736259    
    ## `timedelta:kw_max_min`                                    -3.436e+03  1.198e+03  -2.869 0.004134 ** 
    ## `timedelta:kw_avg_min`                                     1.490e+03  1.055e+03   1.412 0.157876    
    ## `timedelta:kw_min_max`                                     6.212e+01  1.900e+02   0.327 0.743764    
    ## `timedelta:kw_max_max`                                    -1.904e+03  2.847e+03  -0.669 0.503770    
    ## `timedelta:kw_avg_max`                                     1.354e+02  4.433e+02   0.306 0.759998    
    ## `timedelta:kw_min_avg`                                    -5.892e+02  2.895e+02  -2.035 0.041861 *  
    ## `timedelta:kw_max_avg`                                    -1.222e+03  6.391e+02  -1.913 0.055832 .  
    ## `timedelta:kw_avg_avg`                                     1.774e+03  9.718e+02   1.826 0.067967 .  
    ## `timedelta:self_reference_min_shares`                     -4.857e+03  2.893e+03  -1.679 0.093246 .  
    ## `timedelta:self_reference_max_shares`                     -9.537e+02  3.242e+03  -0.294 0.768676    
    ## `timedelta:self_reference_avg_sharess`                     6.546e+03  5.070e+03   1.291 0.196744    
    ## `timedelta:weekday_is_monday`                              4.078e+02  3.200e+02   1.274 0.202556    
    ## `timedelta:weekday_is_tuesday`                             5.215e+01  3.287e+02   0.159 0.873949    
    ## `timedelta:weekday_is_wednesday`                           2.060e+02  3.409e+02   0.604 0.545724    
    ## `timedelta:weekday_is_thursday`                            2.326e+02  3.228e+02   0.721 0.471111    
    ## `timedelta:weekday_is_friday`                             -1.051e+02  3.160e+02  -0.333 0.739500    
    ## `timedelta:weekday_is_saturday`                            1.650e+02  2.145e+02   0.769 0.441749    
    ## `timedelta:weekday_is_sunday`                                     NA         NA      NA       NA    
    ## `timedelta:is_weekend`                                            NA         NA      NA       NA    
    ## `timedelta:LDA_00`                                         2.803e+02  2.305e+02   1.216 0.224005    
    ## `timedelta:LDA_01`                                         1.214e+02  1.979e+02   0.613 0.539641    
    ## `timedelta:LDA_02`                                        -6.878e+01  5.445e+02  -0.126 0.899479    
    ## `timedelta:LDA_03`                                        -2.435e+01  2.653e+02  -0.092 0.926870    
    ## `timedelta:LDA_04`                                                NA         NA      NA       NA    
    ## `timedelta:global_subjectivity`                           -8.472e+01  7.253e+02  -0.117 0.907018    
    ## `timedelta:global_sentiment_polarity`                     -8.899e+01  6.477e+02  -0.137 0.890716    
    ## `timedelta:global_rate_positive_words`                     1.223e+03  7.706e+02   1.587 0.112589    
    ## `timedelta:global_rate_negative_words`                    -1.155e+03  7.193e+02  -1.606 0.108294    
    ## `timedelta:rate_positive_words`                           -3.373e+03  1.666e+03  -2.025 0.042897 *  
    ## `timedelta:rate_negative_words`                                   NA         NA      NA       NA    
    ## `timedelta:avg_positive_polarity`                          2.648e+02  9.421e+02   0.281 0.778641    
    ## `timedelta:min_positive_polarity`                         -2.441e+02  3.411e+02  -0.716 0.474156    
    ## `timedelta:max_positive_polarity`                         -1.738e+02  5.948e+02  -0.292 0.770145    
    ## `timedelta:avg_negative_polarity`                          9.113e+01  6.679e+02   0.136 0.891474    
    ## `timedelta:min_negative_polarity`                          5.125e+02  4.975e+02   1.030 0.303001    
    ## `timedelta:max_negative_polarity`                         -1.312e+02  3.669e+02  -0.358 0.720606    
    ## `timedelta:title_subjectivity`                            -6.044e+00  2.952e+02  -0.020 0.983668    
    ## `timedelta:title_sentiment_polarity`                       4.508e+01  1.933e+02   0.233 0.815653    
    ## `timedelta:abs_title_subjectivity`                        -7.231e+01  3.219e+02  -0.225 0.822279    
    ## `timedelta:abs_title_sentiment_polarity`                  -2.938e+02  2.877e+02  -1.021 0.307255    
    ## `n_tokens_title:n_tokens_content`                         -1.165e+03  8.193e+02  -1.422 0.154979    
    ## `n_tokens_title:n_unique_tokens`                           1.241e+03  2.281e+03   0.544 0.586514    
    ## `n_tokens_title:n_non_stop_words`                          7.616e+03  3.038e+03   2.507 0.012199 *  
    ## `n_tokens_title:n_non_stop_unique_tokens`                 -2.674e+03  2.332e+03  -1.147 0.251624    
    ## `n_tokens_title:num_hrefs`                                -3.871e+02  5.239e+02  -0.739 0.459995    
    ## `n_tokens_title:num_self_hrefs`                           -7.563e+02  4.679e+02  -1.616 0.106108    
    ## `n_tokens_title:num_imgs`                                 -7.714e+02  5.229e+02  -1.475 0.140232    
    ## `n_tokens_title:num_videos`                               -7.966e+01  5.010e+02  -0.159 0.873667    
    ## `n_tokens_title:average_token_length`                     -6.387e+03  2.203e+03  -2.899 0.003755 ** 
    ## `n_tokens_title:num_keywords`                              7.921e+01  5.774e+02   0.137 0.890882    
    ## `n_tokens_title:data_channel_is_world`                            NA         NA      NA       NA    
    ## `n_tokens_title:kw_min_min`                                9.867e-03  9.876e+02   0.000 0.999992    
    ## `n_tokens_title:kw_max_min`                               -9.497e+03  2.310e+03  -4.111 4.01e-05 ***
    ## `n_tokens_title:kw_avg_min`                                7.065e+03  2.028e+03   3.483 0.000500 ***
    ## `n_tokens_title:kw_min_max`                               -1.367e+02  4.950e+02  -0.276 0.782362    
    ## `n_tokens_title:kw_max_max`                                1.057e+03  1.524e+03   0.694 0.487984    
    ## `n_tokens_title:kw_avg_max`                                9.605e+01  7.326e+02   0.131 0.895689    
    ## `n_tokens_title:kw_min_avg`                               -7.867e+02  6.000e+02  -1.311 0.189919    
    ## `n_tokens_title:kw_max_avg`                               -1.498e+03  1.330e+03  -1.126 0.260116    
    ## `n_tokens_title:kw_avg_avg`                                1.313e+03  1.309e+03   1.003 0.315848    
    ## `n_tokens_title:self_reference_min_shares`                -2.617e+03  5.301e+03  -0.494 0.621577    
    ## `n_tokens_title:self_reference_max_shares`                 2.171e+03  5.492e+03   0.395 0.692564    
    ## `n_tokens_title:self_reference_avg_sharess`                6.980e+02  9.295e+03   0.075 0.940142    
    ## `n_tokens_title:weekday_is_monday`                        -2.561e+02  6.232e+02  -0.411 0.681159    
    ## `n_tokens_title:weekday_is_tuesday`                       -5.427e+02  6.547e+02  -0.829 0.407174    
    ## `n_tokens_title:weekday_is_wednesday`                     -7.617e+02  6.554e+02  -1.162 0.245237    
    ## `n_tokens_title:weekday_is_thursday`                      -3.552e+02  6.389e+02  -0.556 0.578295    
    ## `n_tokens_title:weekday_is_friday`                        -8.402e+02  6.218e+02  -1.351 0.176728    
    ## `n_tokens_title:weekday_is_saturday`                      -2.926e+02  5.193e+02  -0.563 0.573128    
    ## `n_tokens_title:weekday_is_sunday`                                NA         NA      NA       NA    
    ## `n_tokens_title:is_weekend`                                       NA         NA      NA       NA    
    ## `n_tokens_title:LDA_00`                                    3.172e+02  4.333e+02   0.732 0.464169    
    ## `n_tokens_title:LDA_01`                                    1.555e+03  4.699e+02   3.308 0.000946 ***
    ## `n_tokens_title:LDA_02`                                    1.352e+02  6.061e+02   0.223 0.823488    
    ## `n_tokens_title:LDA_03`                                    9.569e+02  5.092e+02   1.879 0.060265 .  
    ## `n_tokens_title:LDA_04`                                           NA         NA      NA       NA    
    ## `n_tokens_title:global_subjectivity`                       8.213e+02  7.773e+02   1.057 0.290743    
    ## `n_tokens_title:global_sentiment_polarity`                -3.788e+02  1.169e+03  -0.324 0.746005    
    ## `n_tokens_title:global_rate_positive_words`               -5.630e+02  1.056e+03  -0.533 0.594162    
    ## `n_tokens_title:global_rate_negative_words`                2.285e+02  1.122e+03   0.204 0.838628    
    ## `n_tokens_title:rate_positive_words`                       3.278e+02  1.827e+03   0.179 0.857609    
    ## `n_tokens_title:rate_negative_words`                              NA         NA      NA       NA    
    ## `n_tokens_title:avg_positive_polarity`                    -5.493e+02  1.075e+03  -0.511 0.609565    
    ## `n_tokens_title:min_positive_polarity`                    -2.183e+02  5.281e+02  -0.413 0.679275    
    ## `n_tokens_title:max_positive_polarity`                     9.493e+02  7.478e+02   1.269 0.204329    
    ## `n_tokens_title:avg_negative_polarity`                     6.294e+02  9.930e+02   0.634 0.526236    
    ## `n_tokens_title:min_negative_polarity`                    -4.708e+02  8.401e+02  -0.560 0.575210    
    ## `n_tokens_title:max_negative_polarity`                    -6.590e+02  5.844e+02  -1.128 0.259484    
    ## `n_tokens_title:title_subjectivity`                       -6.078e+02  6.408e+02  -0.949 0.342912    
    ## `n_tokens_title:title_sentiment_polarity`                  4.355e+02  4.288e+02   1.016 0.309856    
    ## `n_tokens_title:abs_title_subjectivity`                    5.769e+02  4.820e+02   1.197 0.231328    
    ## `n_tokens_title:abs_title_sentiment_polarity`              4.306e+02  6.028e+02   0.714 0.475052    
    ## `n_tokens_content:n_unique_tokens`                        -2.276e+03  1.492e+03  -1.526 0.127200    
    ## `n_tokens_content:n_non_stop_words`                               NA         NA      NA       NA    
    ## `n_tokens_content:n_non_stop_unique_tokens`                7.045e+02  1.950e+03   0.361 0.717879    
    ## `n_tokens_content:num_hrefs`                               3.847e+02  4.428e+02   0.869 0.384960    
    ## `n_tokens_content:num_self_hrefs`                         -1.285e+02  3.962e+02  -0.324 0.745752    
    ## `n_tokens_content:num_imgs`                               -6.249e+02  3.694e+02  -1.692 0.090730 .  
    ## `n_tokens_content:num_videos`                              2.696e+02  2.624e+02   1.028 0.304221    
    ## `n_tokens_content:average_token_length`                    7.680e+03  3.138e+03   2.447 0.014430 *  
    ## `n_tokens_content:num_keywords`                           -8.184e+00  7.425e+02  -0.011 0.991207    
    ## `n_tokens_content:data_channel_is_world`                          NA         NA      NA       NA    
    ## `n_tokens_content:kw_min_min`                             -5.441e+02  6.243e+02  -0.871 0.383556    
    ## `n_tokens_content:kw_max_min`                             -4.875e+03  1.472e+03  -3.313 0.000930 ***
    ## `n_tokens_content:kw_avg_min`                              4.544e+03  1.293e+03   3.515 0.000443 ***
    ## `n_tokens_content:kw_min_max`                              2.953e+02  3.791e+02   0.779 0.436097    
    ## `n_tokens_content:kw_max_max`                              5.482e+02  2.244e+03   0.244 0.806969    
    ## `n_tokens_content:kw_avg_max`                             -3.530e+02  7.234e+02  -0.488 0.625624    
    ## `n_tokens_content:kw_min_avg`                              7.348e+02  4.182e+02   1.757 0.078930 .  
    ## `n_tokens_content:kw_max_avg`                              7.139e+02  1.015e+03   0.704 0.481684    
    ## `n_tokens_content:kw_avg_avg`                             -1.283e+03  1.439e+03  -0.892 0.372584    
    ## `n_tokens_content:self_reference_min_shares`              -1.797e+03  2.560e+03  -0.702 0.482852    
    ## `n_tokens_content:self_reference_max_shares`               6.919e+02  3.513e+03   0.197 0.843877    
    ## `n_tokens_content:self_reference_avg_sharess`              7.941e+02  4.579e+03   0.173 0.862339    
    ## `n_tokens_content:weekday_is_monday`                      -7.847e+02  3.965e+02  -1.979 0.047843 *  
    ## `n_tokens_content:weekday_is_tuesday`                      5.499e+02  4.188e+02   1.313 0.189261    
    ## `n_tokens_content:weekday_is_wednesday`                    6.679e+02  3.938e+02   1.696 0.089947 .  
    ## `n_tokens_content:weekday_is_thursday`                     4.070e+01  3.887e+02   0.105 0.916602    
    ## `n_tokens_content:weekday_is_friday`                       3.146e+02  3.734e+02   0.842 0.399605    
    ## `n_tokens_content:weekday_is_saturday`                     1.667e+02  3.336e+02   0.500 0.617246    
    ## `n_tokens_content:weekday_is_sunday`                              NA         NA      NA       NA    
    ## `n_tokens_content:is_weekend`                                     NA         NA      NA       NA    
    ## `n_tokens_content:LDA_00`                                  2.369e+02  2.968e+02   0.798 0.424859    
    ## `n_tokens_content:LDA_01`                                  2.060e+02  2.652e+02   0.777 0.437311    
    ## `n_tokens_content:LDA_02`                                 -2.058e+02  7.627e+02  -0.270 0.787250    
    ## `n_tokens_content:LDA_03`                                 -3.237e+01  2.978e+02  -0.109 0.913435    
    ## `n_tokens_content:LDA_04`                                         NA         NA      NA       NA    
    ## `n_tokens_content:global_subjectivity`                     4.879e+01  1.286e+03   0.038 0.969741    
    ## `n_tokens_content:global_sentiment_polarity`               2.357e+03  1.087e+03   2.167 0.030252 *  
    ## `n_tokens_content:global_rate_positive_words`             -1.525e+03  1.362e+03  -1.119 0.263045    
    ## `n_tokens_content:global_rate_negative_words`              1.881e+03  1.357e+03   1.386 0.165910    
    ## `n_tokens_content:rate_positive_words`                    -1.098e+03  3.303e+03  -0.332 0.739535    
    ##  [ reached getOption("max.print") -- omitted 1286 rows ]
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 4588 on 4705 degrees of freedom
    ## Multiple R-squared:  0.304,  Adjusted R-squared:  0.1273 
    ## F-statistic: 1.721 on 1194 and 4705 DF,  p-value: < 2.2e-16

``` r
#viewing the resulting model

pred <- predict(simon_share_lmfit, newdata = channelTest)
#Creating predictions based on the channelTest data
l1 <- postResample(pred, obs = channelTest$shares)
#creating first linear model postResample results object
l1 #viewing how well the first linear model did 
```

    ##         RMSE     Rsquared          MAE 
    ## 1.010877e+04 1.686800e-03 3.472136e+03

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
    ##  -6113  -1411   -693     84 137826 
    ## 
    ## Coefficients: (5 not defined because of singularities)
    ##                               Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                   2208.994     62.654  35.257  < 2e-16 ***
    ## timedelta                       83.849     89.769   0.934 0.350317    
    ## n_tokens_title                 238.270     66.434   3.587 0.000338 ***
    ## n_tokens_content              -135.855    112.287  -1.210 0.226370    
    ## n_unique_tokens                184.234    298.445   0.617 0.537052    
    ## n_non_stop_words               896.293    341.809   2.622 0.008759 ** 
    ## n_non_stop_unique_tokens       -21.322    296.055  -0.072 0.942587    
    ## num_hrefs                      186.412     77.360   2.410 0.015998 *  
    ## num_self_hrefs                 -14.666     69.713  -0.210 0.833377    
    ## num_imgs                       343.339     74.409   4.614 4.03e-06 ***
    ## num_videos                      82.399     65.533   1.257 0.208672    
    ## average_token_length         -1332.739    265.389  -5.022 5.27e-07 ***
    ## num_keywords                    21.575     78.848   0.274 0.784376    
    ## data_channel_is_world               NA         NA      NA       NA    
    ## kw_min_min                     149.691    130.409   1.148 0.251073    
    ## kw_max_min                     644.670    239.418   2.693 0.007109 ** 
    ## kw_avg_min                    -623.416    241.186  -2.585 0.009768 ** 
    ## kw_min_max                     -20.474     81.090  -0.252 0.800671    
    ## kw_max_max                      93.198    146.376   0.637 0.524344    
    ## kw_avg_max                    -144.235    104.998  -1.374 0.169588    
    ## kw_min_avg                    -289.771     96.074  -3.016 0.002571 ** 
    ## kw_max_avg                    -673.013    154.796  -4.348 1.40e-05 ***
    ## kw_avg_avg                     983.477    172.202   5.711 1.18e-08 ***
    ## self_reference_min_shares     -160.480    168.076  -0.955 0.339715    
    ## self_reference_max_shares      -58.825    145.652  -0.404 0.686323    
    ## self_reference_avg_sharess     239.365    249.416   0.960 0.337245    
    ## weekday_is_monday              -48.181    106.029  -0.454 0.649549    
    ## weekday_is_tuesday            -212.735    109.880  -1.936 0.052909 .  
    ## weekday_is_wednesday          -296.003    110.873  -2.670 0.007612 ** 
    ## weekday_is_thursday           -143.541    109.808  -1.307 0.191198    
    ## weekday_is_friday             -264.576    105.105  -2.517 0.011854 *  
    ## weekday_is_saturday            -66.169     83.254  -0.795 0.426773    
    ## weekday_is_sunday                   NA         NA      NA       NA    
    ## is_weekend                          NA         NA      NA       NA    
    ## LDA_00                          60.149     70.751   0.850 0.395278    
    ## LDA_01                          -1.090     67.877  -0.016 0.987183    
    ## LDA_02                        -168.551     86.042  -1.959 0.050168 .  
    ## LDA_03                         184.373     79.559   2.317 0.020514 *  
    ## LDA_04                              NA         NA      NA       NA    
    ## global_subjectivity            330.570    106.920   3.092 0.001999 ** 
    ## global_sentiment_polarity      -83.610    185.363  -0.451 0.651961    
    ## global_rate_positive_words     105.278    154.391   0.682 0.495336    
    ## global_rate_negative_words     -35.888    166.480  -0.216 0.829331    
    ## rate_positive_words            -86.917    245.747  -0.354 0.723589    
    ## rate_negative_words                 NA         NA      NA       NA    
    ## avg_positive_polarity          -19.108    149.585  -0.128 0.898361    
    ## min_positive_polarity          -20.777     85.556  -0.243 0.808136    
    ## max_positive_polarity            3.639    111.707   0.033 0.974015    
    ## avg_negative_polarity          161.674    148.990   1.085 0.277908    
    ## min_negative_polarity         -119.241    128.541  -0.928 0.353629    
    ## max_negative_polarity         -107.119     91.282  -1.174 0.240643    
    ## title_subjectivity              25.457    100.750   0.253 0.800529    
    ## title_sentiment_polarity        73.768     67.076   1.100 0.271478    
    ## abs_title_subjectivity         106.156     76.065   1.396 0.162888    
    ## abs_title_sentiment_polarity   158.085     94.386   1.675 0.094013 .  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 4813 on 5850 degrees of freedom
    ## Multiple R-squared:  0.04769,    Adjusted R-squared:  0.03972 
    ## F-statistic: 5.979 on 49 and 5850 DF,  p-value: < 2.2e-16

``` r
#viewing the resulting model

pred <- predict(simon_share_lmfit, newdata = channelTest)
#Creating predictions based on the channelTest data
l2 <- postResample(pred, obs = channelTest$shares)
#creating second linear model postResample results object
l2 #viewing how well the second linear model did 
```

    ##         RMSE     Rsquared          MAE 
    ## 1.010877e+04 1.686800e-03 3.472136e+03

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
    ## 5900 samples
    ##   54 predictor
    ## 
    ## Pre-processing: centered (54), scaled (54) 
    ## Resampling: Cross-Validated (5 fold, repeated 3 times) 
    ## Summary of sample sizes: 4720, 4719, 4720, 4721, 4720, 4719, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE      Rsquared    MAE     
    ##   1                   25      4741.148  0.02874822  1764.225
    ##   1                   50      4732.785  0.03240416  1756.340
    ##   1                  100      4742.185  0.03300623  1756.507
    ##   1                  150      4752.480  0.03222956  1776.188
    ##   1                  200      4755.657  0.03198868  1780.599
    ##   2                   25      4749.815  0.02537407  1764.010
    ##   2                   50      4783.764  0.02415272  1767.263
    ##   2                  100      4822.242  0.02379542  1791.140
    ##   2                  150      4836.646  0.02375432  1807.823
    ##   2                  200      4857.527  0.02384991  1831.272
    ##   3                   25      4773.200  0.02139776  1769.997
    ##   3                   50      4819.937  0.02234675  1779.252
    ##   3                  100      4868.948  0.02210695  1822.051
    ##   3                  150      4904.121  0.02055774  1856.238
    ##   3                  200      4927.897  0.01968876  1887.702
    ##   4                   25      4790.990  0.02068245  1776.551
    ##   4                   50      4834.299  0.02322455  1796.680
    ##   4                  100      4890.403  0.02212244  1841.935
    ##   4                  150      4921.739  0.02095808  1879.246
    ##   4                  200      4951.887  0.02048835  1920.454
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## Tuning parameter 'n.minobsinnode' was
    ##  held constant at a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 50, interaction.depth = 1, shrinkage = 0.1
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
    ## 8113.8556474    0.0244643 2028.6794876

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
    ## 5900 samples
    ##   54 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold, repeated 3 times) 
    ## Summary of sample sizes: 4719, 4721, 4720, 4721, 4719, 4721, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  RMSE      Rsquared    MAE     
    ##    1    4783.097  0.03532491  1762.733
    ##    2    4777.879  0.03620650  1803.213
    ##    3    4790.490  0.03418357  1821.954
    ##    4    4799.948  0.03345594  1836.818
    ##    5    4811.891  0.03183968  1852.115
    ##    6    4818.873  0.03152368  1860.188
    ##    7    4834.597  0.02872568  1872.847
    ##    8    4838.203  0.02945960  1875.253
    ##    9    4843.643  0.02927571  1881.738
    ##   10    4847.185  0.02891356  1885.117
    ##   11    4858.558  0.02785137  1891.257
    ##   12    4868.208  0.02662107  1896.465
    ##   13    4864.306  0.02770111  1897.078
    ##   14    4875.148  0.02630161  1903.639
    ##   15    4881.252  0.02581093  1907.098
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
    ## 8.080788e+03 3.281664e-02 2.052579e+03

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
