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
    ## 1       731             13             1072           0.416             1.00                  0.541        19
    ## 2       731             10              370           0.560             1.00                  0.698         2
    ## 3       731             12              989           0.434             1.00                  0.572        20
    ## 4       731             11               97           0.670             1.00                  0.837         2
    ## 5       731              8             1207           0.411             1.00                  0.549        24
    ## 6       731             13             1248           0.391             1.00                  0.523        21
    ## # ℹ abbreviated name: ¹​n_non_stop_unique_tokens
    ## # ℹ 48 more variables: num_self_hrefs <dbl>, num_imgs <dbl>, num_videos <dbl>, average_token_length <dbl>,
    ## #   num_keywords <dbl>, data_channel_is_tech <dbl>, kw_min_min <dbl>, kw_max_min <dbl>, kw_avg_min <dbl>,
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
    ##      82    1100    1700    3198    3000  663600

The statistics above explore the spread of the `shares` variable. If the
mean is less than the median, the data is skewed to the left. If the
median is less than the mean, then the data is skewed to the right.

``` r
# contingency table for number of videos
videos_table <- table(channelTrain_summary$num_videos)
videos_table
```

    ## 
    ##    0    1    2    3    4    5    6    7    8    9   10   11   12   14   15   17   59   73 
    ## 3698 1130  225   32   10    6   10    7    3    6    6    6    1    1    1    1    1    1

The contingency table above shows the frequency of the number of videos
in each observation of our data.

``` r
ggplot(channelTrain_summary, aes(x = n_tokens_content, y = shares)) +
geom_point() +
labs(x = "Number of words in the content", y = "Number of Shares") +
ggtitle("Shares vs. Number of words in the content")
```

![](tech_files/figure-gfm/summaries%203-1.png)<!-- -->

``` r
# Creating ggplot of number of words in the content vs shares

ggplot(channelTrain_summary, aes(x = rate_positive_words, y = shares)) +
geom_point() +
labs(x = "Positive Word Rate", y = "Number of Shares") +
ggtitle("Shares vs. Positive Word Rate")
```

![](tech_files/figure-gfm/summaries%203-2.png)<!-- -->

``` r
# Creating ggplot of positive word rate vs shares

ggplot(channelTrain_summary, aes(x = rate_negative_words, y = shares)) +
geom_point() +
labs(x = "Negative Word Rate", y = "Number of Shares") +
ggtitle("Shares vs. Negative Word Rate")
```

![](tech_files/figure-gfm/summaries%203-3.png)<!-- -->

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
    ## 1 Sunday      206          1600 4087.  2500          4000  83300  6410.
    ## 2 Monday      192          1100 2899.  1600          3100  51000  4176.
    ## 3 Tuesday     104          1100 2973.  1600          2800  88500  5093.
    ## 4 Wednesday   218          1000 3658.  1600          2800 663600 21432.
    ## 5 Thursday     86          1000 2706.  1500          2600  52600  4073.
    ## 6 Friday       82          1200 3257.  1800          3200 104100  6142.
    ## 7 Saturday    119          1600 3696.  2200          3700  96100  6047.

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

![](tech_files/figure-gfm/Bar%20Chart%20by%20Day-1.png)<!-- -->

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

![](tech_files/figure-gfm/Histogram%20of%20Shares%20by%20Part%20of%20Week-1.png)<!-- -->

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
    ##   High Shares           1931             740
    ##   Low Shares            1827             647

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
    ##   High Shares         410        1110                 1151
    ##   Low Shares          420        1159                  895

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
    ##   High Shares      1847       633                191
    ##   Low Shares       1851       497                126

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

![](tech_files/figure-gfm/Scatterplot%20for%20Links-1.png)<!-- -->

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

![](tech_files/figure-gfm/Scatterplot%20for%20Keywords-1.png)<!-- -->

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
    ## 1            2  1700          1700 1700   1700          1700   1700    NA 
    ## 2            3   119          1300 2262   1700          2900  10900  1970.
    ## 3            4   498          1250 3431.  1800          4000  20700  3792.
    ## 4            5   413          1100 3109.  1700          2925  52600  5020.
    ## 5            6   206          1125 2856.  1700          2700  48000  4046.
    ## 6            7   116          1100 3030.  1700          3000  67800  4644.
    ## 7            8   211          1100 3012.  1600          3000  96100  5447.
    ## 8            9   104          1100 3146.  1600          2800  83300  6167.
    ## 9           10    82          1200 3756.  1800          3400 663600 19907.

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
    ## -46408  -3084    -95   2805 342510 
    ## 
    ## Coefficients: (307 not defined because of singularities)
    ##                                                             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                                3.198e+03  1.240e+02  25.783  < 2e-16 ***
    ## timedelta                                                  1.385e+04  5.880e+04   0.236 0.813756    
    ## n_tokens_title                                            -3.196e+04  4.031e+04  -0.793 0.428028    
    ## n_tokens_content                                           9.118e+04  1.534e+04   5.945 3.01e-09 ***
    ## n_unique_tokens                                            1.347e+04  1.844e+04   0.731 0.464968    
    ## n_non_stop_words                                           7.885e+03  3.944e+04   0.200 0.841534    
    ## n_non_stop_unique_tokens                                  -4.798e+03  1.549e+04  -0.310 0.756705    
    ## num_hrefs                                                  3.316e+04  1.056e+04   3.142 0.001692 ** 
    ## num_self_hrefs                                            -3.961e+04  1.033e+04  -3.833 0.000128 ***
    ## num_imgs                                                  -2.956e+04  9.494e+03  -3.113 0.001863 ** 
    ## num_videos                                                -2.240e+04  9.900e+04  -0.226 0.820997    
    ## average_token_length                                      -1.130e+04  5.767e+03  -1.960 0.050077 .  
    ## num_keywords                                               3.535e+04  1.447e+05   0.244 0.806966    
    ## data_channel_is_tech                                              NA         NA      NA       NA    
    ## kw_min_min                                                -3.885e+04  2.651e+04  -1.466 0.142792    
    ## kw_max_min                                                -1.737e+05  2.689e+05  -0.646 0.518407    
    ## kw_avg_min                                                 1.738e+05  3.822e+05   0.455 0.649307    
    ## kw_min_max                                                 7.742e+04  3.557e+05   0.218 0.827736    
    ## kw_max_max                                                -9.271e+03  1.593e+04  -0.582 0.560649    
    ## kw_avg_max                                                 8.316e+04  1.685e+05   0.494 0.621574    
    ## kw_min_avg                                                -3.036e+04  8.499e+04  -0.357 0.720916    
    ## kw_max_avg                                                -9.664e+04  2.282e+05  -0.423 0.671974    
    ## kw_avg_avg                                                -1.142e+04  1.555e+04  -0.735 0.462612    
    ## self_reference_min_shares                                 -1.388e+05  6.400e+04  -2.168 0.030185 *  
    ## self_reference_max_shares                                 -3.936e+04  9.050e+04  -0.435 0.663638    
    ## self_reference_avg_sharess                                 1.967e+05  1.274e+05   1.544 0.122724    
    ## weekday_is_monday                                         -8.024e+03  1.048e+04  -0.765 0.444045    
    ## weekday_is_tuesday                                        -4.589e+03  1.096e+04  -0.419 0.675533    
    ## weekday_is_wednesday                                      -3.626e+03  1.092e+04  -0.332 0.739929    
    ## weekday_is_thursday                                       -9.232e+03  1.079e+04  -0.855 0.392496    
    ## weekday_is_friday                                         -9.715e+03  9.715e+03  -1.000 0.317373    
    ## weekday_is_saturday                                       -1.441e+03  8.728e+03  -0.165 0.868833    
    ## weekday_is_sunday                                                 NA         NA      NA       NA    
    ## is_weekend                                                        NA         NA      NA       NA    
    ## LDA_00                                                     1.956e+04  5.827e+03   3.356 0.000798 ***
    ## LDA_01                                                     1.796e+03  5.383e+03   0.334 0.738601    
    ## LDA_02                                                    -4.783e+03  5.675e+03  -0.843 0.399449    
    ## LDA_03                                                    -5.357e+03  5.812e+03  -0.922 0.356715    
    ## LDA_04                                                            NA         NA      NA       NA    
    ## global_subjectivity                                       -4.474e+03  5.462e+03  -0.819 0.412744    
    ## global_sentiment_polarity                                  2.187e+04  1.146e+04   1.907 0.056542 .  
    ## global_rate_positive_words                                 1.746e+03  1.293e+04   0.135 0.892571    
    ## global_rate_negative_words                                -1.691e+04  1.147e+04  -1.474 0.140486    
    ## rate_positive_words                                       -1.518e+04  1.320e+04  -1.150 0.250099    
    ## rate_negative_words                                               NA         NA      NA       NA    
    ## avg_positive_polarity                                      2.015e+03  9.170e+03   0.220 0.826096    
    ## min_positive_polarity                                      2.026e+03  6.261e+03   0.324 0.746212    
    ## max_positive_polarity                                     -7.487e+03  8.095e+03  -0.925 0.355121    
    ## avg_negative_polarity                                     -3.396e+04  1.494e+04  -2.273 0.023075 *  
    ## min_negative_polarity                                      2.505e+04  1.362e+04   1.839 0.065988 .  
    ## max_negative_polarity                                      1.173e+04  9.467e+03   1.239 0.215339    
    ## title_subjectivity                                        -1.886e+03  7.693e+03  -0.245 0.806342    
    ## title_sentiment_polarity                                  -9.065e+02  8.057e+03  -0.113 0.910431    
    ## abs_title_subjectivity                                    -9.708e+03  6.735e+03  -1.441 0.149542    
    ## abs_title_sentiment_polarity                               4.894e+03  9.821e+03   0.498 0.618313    
    ## `timedelta:n_tokens_title`                                -2.005e+03  1.469e+03  -1.365 0.172444    
    ## `timedelta:n_tokens_content`                              -4.646e+03  1.550e+03  -2.998 0.002737 ** 
    ## `timedelta:n_unique_tokens`                               -1.252e+04  6.295e+03  -1.989 0.046780 *  
    ## `timedelta:n_non_stop_words`                              -1.526e+04  5.992e+04  -0.255 0.798979    
    ## `timedelta:n_non_stop_unique_tokens`                       1.118e+04  6.311e+03   1.772 0.076514 .  
    ## `timedelta:num_hrefs`                                     -1.557e+03  1.252e+03  -1.244 0.213708    
    ## `timedelta:num_self_hrefs`                                 1.033e+03  1.117e+03   0.924 0.355394    
    ## `timedelta:num_imgs`                                       1.949e+03  1.049e+03   1.857 0.063326 .  
    ## `timedelta:num_videos`                                     6.941e+02  1.429e+03   0.486 0.627219    
    ## `timedelta:average_token_length`                          -8.660e+02  5.836e+03  -0.148 0.882033    
    ## `timedelta:num_keywords`                                   2.293e+02  1.628e+03   0.141 0.888019    
    ## `timedelta:data_channel_is_tech`                                  NA         NA      NA       NA    
    ## `timedelta:kw_min_min`                                     4.119e+03  8.934e+03   0.461 0.644799    
    ## `timedelta:kw_max_min`                                     1.087e+03  3.339e+03   0.325 0.744890    
    ## `timedelta:kw_avg_min`                                    -1.141e+03  3.306e+03  -0.345 0.729967    
    ## `timedelta:kw_min_max`                                     1.311e+03  9.065e+02   1.447 0.148060    
    ## `timedelta:kw_max_max`                                     3.282e+03  4.879e+03   0.673 0.501114    
    ## `timedelta:kw_avg_max`                                     2.507e+02  6.335e+02   0.396 0.692312    
    ## `timedelta:kw_min_avg`                                    -2.325e+02  9.851e+02  -0.236 0.813450    
    ## `timedelta:kw_max_avg`                                     2.427e+03  2.078e+03   1.168 0.242758    
    ## `timedelta:kw_avg_avg`                                    -2.419e+03  3.167e+03  -0.764 0.445045    
    ## `timedelta:self_reference_min_shares`                      5.873e+03  4.348e+03   1.351 0.176920    
    ## `timedelta:self_reference_max_shares`                      4.586e+03  5.927e+03   0.774 0.439169    
    ## `timedelta:self_reference_avg_sharess`                    -1.132e+04  8.747e+03  -1.294 0.195704    
    ## `timedelta:weekday_is_monday`                              5.173e+02  1.317e+03   0.393 0.694528    
    ## `timedelta:weekday_is_tuesday`                             7.536e+02  1.366e+03   0.552 0.581129    
    ## `timedelta:weekday_is_wednesday`                           1.023e+03  1.405e+03   0.728 0.466798    
    ## `timedelta:weekday_is_thursday`                            1.935e+02  1.322e+03   0.146 0.883612    
    ## `timedelta:weekday_is_friday`                              7.483e+02  1.249e+03   0.599 0.549253    
    ## `timedelta:weekday_is_saturday`                            4.287e+02  9.948e+02   0.431 0.666569    
    ## `timedelta:weekday_is_sunday`                                     NA         NA      NA       NA    
    ## `timedelta:is_weekend`                                            NA         NA      NA       NA    
    ## `timedelta:LDA_00`                                        -9.195e+02  6.431e+02  -1.430 0.152861    
    ## `timedelta:LDA_01`                                        -5.795e+02  6.576e+02  -0.881 0.378271    
    ## `timedelta:LDA_02`                                         6.239e+02  7.150e+02   0.873 0.382946    
    ## `timedelta:LDA_03`                                         1.407e+03  7.413e+02   1.899 0.057687 .  
    ## `timedelta:LDA_04`                                                NA         NA      NA       NA    
    ## `timedelta:global_subjectivity`                            1.451e+03  2.305e+03   0.630 0.529015    
    ## `timedelta:global_sentiment_polarity`                     -3.555e+03  2.279e+03  -1.560 0.118939    
    ## `timedelta:global_rate_positive_words`                     2.156e+03  2.444e+03   0.882 0.377883    
    ## `timedelta:global_rate_negative_words`                    -6.903e+02  2.417e+03  -0.286 0.775236    
    ## `timedelta:rate_positive_words`                            1.060e+03  6.619e+03   0.160 0.872809    
    ## `timedelta:rate_negative_words`                                   NA         NA      NA       NA    
    ## `timedelta:avg_positive_polarity`                          1.852e+03  2.974e+03   0.623 0.533447    
    ## `timedelta:min_positive_polarity`                         -1.886e+02  1.099e+03  -0.172 0.863728    
    ## `timedelta:max_positive_polarity`                         -5.686e+02  1.786e+03  -0.318 0.750218    
    ## `timedelta:avg_negative_polarity`                          1.457e+03  2.156e+03   0.675 0.499419    
    ## `timedelta:min_negative_polarity`                         -9.417e+02  1.664e+03  -0.566 0.571527    
    ## `timedelta:max_negative_polarity`                         -3.130e+02  1.178e+03  -0.266 0.790413    
    ## `timedelta:title_subjectivity`                             3.270e+02  9.682e+02   0.338 0.735569    
    ## `timedelta:title_sentiment_polarity`                       5.634e+02  8.972e+02   0.628 0.530050    
    ## `timedelta:abs_title_subjectivity`                         1.375e+03  1.027e+03   1.339 0.180760    
    ## `timedelta:abs_title_sentiment_polarity`                  -1.097e+03  1.138e+03  -0.964 0.335220    
    ## `n_tokens_title:n_tokens_content`                          3.466e+03  1.590e+03   2.180 0.029306 *  
    ## `n_tokens_title:n_unique_tokens`                           7.449e+03  4.153e+03   1.793 0.072971 .  
    ## `n_tokens_title:n_non_stop_words`                          3.370e+04  4.166e+04   0.809 0.418567    
    ## `n_tokens_title:n_non_stop_unique_tokens`                 -1.138e+04  4.002e+03  -2.843 0.004497 ** 
    ## `n_tokens_title:num_hrefs`                                 4.242e+03  1.241e+03   3.419 0.000635 ***
    ## `n_tokens_title:num_self_hrefs`                           -1.606e+03  1.163e+03  -1.381 0.167280    
    ## `n_tokens_title:num_imgs`                                 -1.794e+03  1.087e+03  -1.651 0.098797 .  
    ## `n_tokens_title:num_videos`                                1.308e+03  1.767e+03   0.740 0.459061    
    ## `n_tokens_title:average_token_length`                      3.103e+03  3.127e+03   0.992 0.321160    
    ## `n_tokens_title:num_keywords`                             -1.896e+02  1.255e+03  -0.151 0.879926    
    ## `n_tokens_title:data_channel_is_tech`                             NA         NA      NA       NA    
    ## `n_tokens_title:kw_min_min`                                4.127e+02  1.407e+03   0.293 0.769371    
    ## `n_tokens_title:kw_max_min`                               -1.476e+03  4.450e+03  -0.332 0.740049    
    ## `n_tokens_title:kw_avg_min`                                4.766e+02  3.662e+03   0.130 0.896447    
    ## `n_tokens_title:kw_min_max`                               -4.024e+02  1.831e+03  -0.220 0.826058    
    ## `n_tokens_title:kw_max_max`                               -1.303e+03  2.018e+03  -0.646 0.518378    
    ## `n_tokens_title:kw_avg_max`                               -2.497e+03  1.919e+03  -1.301 0.193348    
    ## `n_tokens_title:kw_min_avg`                                1.954e+01  1.289e+03   0.015 0.987899    
    ## `n_tokens_title:kw_max_avg`                               -3.491e+03  3.107e+03  -1.124 0.261245    
    ## `n_tokens_title:kw_avg_avg`                                5.079e+03  2.793e+03   1.818 0.069089 .  
    ## `n_tokens_title:self_reference_min_shares`                -4.329e+04  1.065e+04  -4.063 4.93e-05 ***
    ## `n_tokens_title:self_reference_max_shares`                -4.968e+04  1.162e+04  -4.276 1.95e-05 ***
    ## `n_tokens_title:self_reference_avg_sharess`                8.538e+04  1.916e+04   4.457 8.55e-06 ***
    ## `n_tokens_title:weekday_is_monday`                         3.085e+03  1.549e+03   1.992 0.046479 *  
    ## `n_tokens_title:weekday_is_tuesday`                        3.379e+03  1.607e+03   2.103 0.035527 *  
    ## `n_tokens_title:weekday_is_wednesday`                      4.041e+03  1.595e+03   2.534 0.011326 *  
    ## `n_tokens_title:weekday_is_thursday`                       2.404e+03  1.554e+03   1.547 0.121904    
    ## `n_tokens_title:weekday_is_friday`                         2.299e+03  1.419e+03   1.621 0.105199    
    ## `n_tokens_title:weekday_is_saturday`                       7.981e+01  1.224e+03   0.065 0.948028    
    ## `n_tokens_title:weekday_is_sunday`                                NA         NA      NA       NA    
    ## `n_tokens_title:is_weekend`                                       NA         NA      NA       NA    
    ## `n_tokens_title:LDA_00`                                    6.997e+02  8.519e+02   0.821 0.411487    
    ## `n_tokens_title:LDA_01`                                    1.085e+03  7.952e+02   1.364 0.172661    
    ## `n_tokens_title:LDA_02`                                   -2.095e+03  8.606e+02  -2.434 0.014965 *  
    ## `n_tokens_title:LDA_03`                                   -9.836e+02  8.588e+02  -1.145 0.252144    
    ## `n_tokens_title:LDA_04`                                           NA         NA      NA       NA    
    ## `n_tokens_title:global_subjectivity`                      -8.350e+02  1.518e+03  -0.550 0.582376    
    ## `n_tokens_title:global_sentiment_polarity`                 1.309e+03  2.097e+03   0.624 0.532432    
    ## `n_tokens_title:global_rate_positive_words`               -1.321e+03  1.945e+03  -0.679 0.497158    
    ## `n_tokens_title:global_rate_negative_words`                2.667e+03  2.564e+03   1.040 0.298292    
    ## `n_tokens_title:rate_positive_words`                       9.609e+02  4.278e+03   0.225 0.822278    
    ## `n_tokens_title:rate_negative_words`                              NA         NA      NA       NA    
    ## `n_tokens_title:avg_positive_polarity`                    -1.252e+03  2.115e+03  -0.592 0.553911    
    ## `n_tokens_title:min_positive_polarity`                    -1.793e+00  1.142e+03  -0.002 0.998747    
    ## `n_tokens_title:max_positive_polarity`                    -1.140e+03  1.445e+03  -0.789 0.430296    
    ## `n_tokens_title:avg_negative_polarity`                    -6.940e+03  2.310e+03  -3.005 0.002677 ** 
    ## `n_tokens_title:min_negative_polarity`                     7.322e+03  1.910e+03   3.834 0.000128 ***
    ## `n_tokens_title:max_negative_polarity`                     2.062e+03  1.442e+03   1.430 0.152812    
    ## `n_tokens_title:title_subjectivity`                       -2.515e+03  1.145e+03  -2.196 0.028137 *  
    ## `n_tokens_title:title_sentiment_polarity`                  9.449e+02  1.099e+03   0.860 0.390074    
    ## `n_tokens_title:abs_title_subjectivity`                   -3.209e+03  9.766e+02  -3.286 0.001027 ** 
    ## `n_tokens_title:abs_title_sentiment_polarity`             -1.275e+03  1.317e+03  -0.968 0.333075    
    ## `n_tokens_content:n_unique_tokens`                        -5.200e+03  3.557e+03  -1.462 0.143852    
    ## `n_tokens_content:n_non_stop_words`                               NA         NA      NA       NA    
    ## `n_tokens_content:n_non_stop_unique_tokens`               -1.599e+04  4.607e+03  -3.472 0.000523 ***
    ## `n_tokens_content:num_hrefs`                              -1.301e+03  9.132e+02  -1.425 0.154199    
    ## `n_tokens_content:num_self_hrefs`                         -5.604e+03  1.292e+03  -4.338 1.48e-05 ***
    ## `n_tokens_content:num_imgs`                               -1.749e+03  9.433e+02  -1.854 0.063772 .  
    ## `n_tokens_content:num_videos`                             -1.689e+02  1.708e+03  -0.099 0.921234    
    ## `n_tokens_content:average_token_length`                    2.684e+04  7.453e+03   3.602 0.000320 ***
    ## `n_tokens_content:num_keywords`                            2.547e+03  2.282e+03   1.116 0.264393    
    ## `n_tokens_content:data_channel_is_tech`                           NA         NA      NA       NA    
    ## `n_tokens_content:kw_min_min`                              1.517e+03  1.080e+03   1.404 0.160250    
    ## `n_tokens_content:kw_max_min`                              4.704e+03  3.672e+03   1.281 0.200245    
    ## `n_tokens_content:kw_avg_min`                             -7.484e+03  3.003e+03  -2.492 0.012736 *  
    ## `n_tokens_content:kw_min_max`                             -6.664e+02  1.244e+03  -0.536 0.592255    
    ## `n_tokens_content:kw_max_max`                              5.068e+03  2.763e+03   1.834 0.066710 .  
    ## `n_tokens_content:kw_avg_max`                             -1.016e+04  1.567e+03  -6.483 1.01e-10 ***
    ## `n_tokens_content:kw_min_avg`                              8.246e+02  8.704e+02   0.947 0.343488    
    ## `n_tokens_content:kw_max_avg`                             -7.355e+03  2.286e+03  -3.218 0.001303 ** 
    ## `n_tokens_content:kw_avg_avg`                              8.661e+03  3.635e+03   2.382 0.017253 *  
    ## `n_tokens_content:self_reference_min_shares`              -2.485e+04  4.193e+03  -5.926 3.36e-09 ***
    ## `n_tokens_content:self_reference_max_shares`              -2.189e+04  5.937e+03  -3.686 0.000231 ***
    ## `n_tokens_content:self_reference_avg_sharess`              4.373e+04  8.093e+03   5.404 6.92e-08 ***
    ## `n_tokens_content:weekday_is_monday`                       2.067e+02  7.996e+02   0.259 0.796012    
    ## `n_tokens_content:weekday_is_tuesday`                     -1.022e+03  7.798e+02  -1.311 0.190092    
    ## `n_tokens_content:weekday_is_wednesday`                    3.079e+03  8.012e+02   3.843 0.000124 ***
    ## `n_tokens_content:weekday_is_thursday`                    -1.082e+03  7.501e+02  -1.442 0.149284    
    ## `n_tokens_content:weekday_is_friday`                      -2.964e+00  6.959e+02  -0.004 0.996602    
    ## `n_tokens_content:weekday_is_saturday`                    -8.774e+02  6.569e+02  -1.336 0.181730    
    ## `n_tokens_content:weekday_is_sunday`                              NA         NA      NA       NA    
    ## `n_tokens_content:is_weekend`                                     NA         NA      NA       NA    
    ## `n_tokens_content:LDA_00`                                  4.000e+03  5.201e+02   7.690 1.84e-14 ***
    ## `n_tokens_content:LDA_01`                                  7.942e+02  5.442e+02   1.459 0.144578    
    ## `n_tokens_content:LDA_02`                                 -2.618e+03  5.571e+02  -4.699 2.71e-06 ***
    ## `n_tokens_content:LDA_03`                                 -1.664e+02  6.374e+02  -0.261 0.794037    
    ## `n_tokens_content:LDA_04`                                         NA         NA      NA       NA    
    ## `n_tokens_content:global_subjectivity`                    -8.442e+03  3.596e+03  -2.347 0.018959 *  
    ## `n_tokens_content:global_sentiment_polarity`               5.646e+03  3.133e+03   1.802 0.071644 .  
    ## `n_tokens_content:global_rate_positive_words`              9.883e+03  3.837e+03   2.575 0.010047 *  
    ## `n_tokens_content:global_rate_negative_words`             -1.322e+04  3.529e+03  -3.747 0.000182 ***
    ## `n_tokens_content:rate_positive_words`                    -7.333e+04  1.122e+04  -6.535 7.17e-11 ***
    ##  [ reached getOption("max.print") -- omitted 1286 rows ]
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 8896 on 3966 degrees of freedom
    ## Multiple R-squared:  0.4504, Adjusted R-squared:  0.2871 
    ## F-statistic: 2.759 on 1178 and 3966 DF,  p-value: < 2.2e-16

``` r
#viewing the resulting model

pred <- predict(simon_share_lmfit, newdata = channelTest)
#Creating predictions based on the channelTest data
l1 <- postResample(pred, obs = channelTest$shares)
#creating first linear model postResample results object
l1 #viewing how well the first linear model did 
```

    ##         RMSE     Rsquared          MAE 
    ## 1.662282e+04 1.461373e-03 6.643900e+03

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
    ## -15339  -2096  -1005    459 648855 
    ## 
    ## Coefficients: (5 not defined because of singularities)
    ##                              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                   3197.76     145.74  21.942  < 2e-16 ***
    ## timedelta                     -276.39     281.70  -0.981 0.326556    
    ## n_tokens_title                 186.69     156.83   1.190 0.233931    
    ## n_tokens_content              1043.63     279.10   3.739 0.000187 ***
    ## n_unique_tokens                162.66     551.61   0.295 0.768099    
    ## n_non_stop_words               238.34     264.82   0.900 0.368168    
    ## n_non_stop_unique_tokens      -358.41     476.78  -0.752 0.452250    
    ## num_hrefs                      709.98     203.60   3.487 0.000492 ***
    ## num_self_hrefs                -574.77     190.13  -3.023 0.002515 ** 
    ## num_imgs                      -464.68     196.84  -2.361 0.018276 *  
    ## num_videos                     191.61     149.46   1.282 0.199896    
    ## average_token_length           -78.74     217.81  -0.361 0.717749    
    ## num_keywords                   -79.74     175.12  -0.455 0.648871    
    ## data_channel_is_tech               NA         NA      NA       NA    
    ## kw_min_min                     436.32     270.41   1.614 0.106689    
    ## kw_max_min                     528.28     522.72   1.011 0.312234    
    ## kw_avg_min                    -723.51     510.39  -1.418 0.156382    
    ## kw_min_max                     -52.01     163.94  -0.317 0.751075    
    ## kw_max_max                     222.40     296.28   0.751 0.452908    
    ## kw_avg_max                    -843.28     312.56  -2.698 0.006998 ** 
    ## kw_min_avg                    -162.17     225.92  -0.718 0.472901    
    ## kw_max_avg                    -729.67     357.17  -2.043 0.041112 *  
    ## kw_avg_avg                    1256.38     386.78   3.248 0.001168 ** 
    ## self_reference_min_shares     -145.63     358.24  -0.407 0.684379    
    ## self_reference_max_shares     -274.07     459.77  -0.596 0.551125    
    ## self_reference_avg_sharess     502.36     656.30   0.765 0.444048    
    ## weekday_is_monday             -224.45     272.16  -0.825 0.409585    
    ## weekday_is_tuesday            -256.95     286.38  -0.897 0.369640    
    ## weekday_is_wednesday            13.90     285.52   0.049 0.961170    
    ## weekday_is_thursday           -382.54     278.69  -1.373 0.169917    
    ## weekday_is_friday             -126.94     256.58  -0.495 0.620824    
    ## weekday_is_saturday            -72.30     216.22  -0.334 0.738112    
    ## weekday_is_sunday                  NA         NA      NA       NA    
    ## is_weekend                         NA         NA      NA       NA    
    ## LDA_00                         362.95     150.76   2.408 0.016097 *  
    ## LDA_01                         -27.03     150.56  -0.180 0.857545    
    ## LDA_02                         -44.85     158.02  -0.284 0.776554    
    ## LDA_03                          13.92     152.30   0.091 0.927164    
    ## LDA_04                             NA         NA      NA       NA    
    ## global_subjectivity             74.72     181.85   0.411 0.681190    
    ## global_sentiment_polarity      120.18     382.23   0.314 0.753212    
    ## global_rate_positive_words    -280.06     291.76  -0.960 0.337142    
    ## global_rate_negative_words     -91.96     406.41  -0.226 0.821005    
    ## rate_positive_words           -447.83     473.91  -0.945 0.344726    
    ## rate_negative_words                NA         NA      NA       NA    
    ## avg_positive_polarity         -150.19     292.53  -0.513 0.607692    
    ## min_positive_polarity          -38.76     197.20  -0.197 0.844181    
    ## max_positive_polarity          -43.29     226.99  -0.191 0.848770    
    ## avg_negative_polarity        -1410.30     377.17  -3.739 0.000187 ***
    ## min_negative_polarity         1419.79     330.82   4.292 1.81e-05 ***
    ## max_negative_polarity          731.15     250.21   2.922 0.003491 ** 
    ## title_subjectivity             -73.64     218.55  -0.337 0.736157    
    ## title_sentiment_polarity       206.81     194.32   1.064 0.287272    
    ## abs_title_subjectivity        -228.69     185.34  -1.234 0.217295    
    ## abs_title_sentiment_polarity   -38.22     245.73  -0.156 0.876416    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 10450 on 5095 degrees of freedom
    ## Multiple R-squared:  0.02504,    Adjusted R-squared:  0.01567 
    ## F-statistic: 2.671 on 49 and 5095 DF,  p-value: 3.09e-09

``` r
#viewing the resulting model

pred <- predict(simon_share_lmfit, newdata = channelTest)
#Creating predictions based on the channelTest data
l2 <- postResample(pred, obs = channelTest$shares)
#creating second linear model postResample results object
l2 #viewing how well the second linear model did 
```

    ##         RMSE     Rsquared          MAE 
    ## 1.662282e+04 1.461373e-03 6.643900e+03

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
    ## 5145 samples
    ##   54 predictor
    ## 
    ## Pre-processing: centered (54), scaled (54) 
    ## Resampling: Cross-Validated (5 fold, repeated 3 times) 
    ## Summary of sample sizes: 4115, 4116, 4116, 4116, 4117, 4116, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE      Rsquared     MAE     
    ##   1                   25      8601.652  0.001340706  2613.062
    ##   1                   50      8721.798  0.001811878  2628.937
    ##   1                  100      8741.750  0.002000311  2631.433
    ##   1                  150      8723.770  0.002219155  2628.369
    ##   1                  200      8796.016  0.002471006  2637.084
    ##   2                   25      8543.037  0.003348676  2566.862
    ##   2                   50      8589.567  0.004202854  2576.271
    ##   2                  100      8745.371  0.005504639  2587.327
    ##   2                  150      8930.045  0.006125841  2604.244
    ##   2                  200      9034.368  0.007005275  2621.836
    ##   3                   25      8545.470  0.006168109  2579.081
    ##   3                   50      8625.404  0.007360068  2585.419
    ##   3                  100      8748.095  0.010462192  2601.076
    ##   3                  150      8901.707  0.011364847  2618.008
    ##   3                  200      9181.064  0.011662143  2653.393
    ##   4                   25      8505.011  0.004709660  2562.159
    ##   4                   50      8622.812  0.006245644  2585.546
    ##   4                  100      8824.396  0.007378395  2611.604
    ##   4                  150      9020.628  0.007649544  2654.759
    ##   4                  200      9317.999  0.007567333  2704.979
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## Tuning parameter 'n.minobsinnode' was
    ##  held constant at a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 25, interaction.depth = 4, shrinkage = 0.1
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
    ## 3.672422e+03 2.609189e-02 2.105377e+03

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
    ## 5145 samples
    ##   54 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold, repeated 3 times) 
    ## Summary of sample sizes: 4115, 4116, 4116, 4115, 4118, 4117, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  RMSE      Rsquared    MAE     
    ##    1    8266.131  0.02061497  2510.578
    ##    2    8294.359  0.02242574  2545.665
    ##    3    8396.628  0.01763870  2580.167
    ##    4    8398.101  0.01801980  2594.710
    ##    5    8486.536  0.01714298  2612.719
    ##    6    8537.629  0.01597046  2628.946
    ##    7    8591.097  0.01519914  2644.887
    ##    8    8686.792  0.01474042  2657.289
    ##    9    8807.560  0.01369019  2675.393
    ##   10    8823.726  0.01363331  2682.636
    ##   11    8954.966  0.01175276  2693.260
    ##   12    9012.880  0.01148164  2713.027
    ##   13    9057.973  0.01187415  2709.189
    ##   14    9080.152  0.01248915  2712.342
    ##   15    9164.615  0.01100111  2726.472
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
    ## 3.460401e+03 3.591637e-02 2.087114e+03

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
