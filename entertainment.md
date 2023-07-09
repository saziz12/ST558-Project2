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
library(tidyverse)
library(ggplot2) # for creating ggplots
library(stringr) # for str_to_title()
library(caret) # for model training and evaluation
library(gbm) #Needed this library to run the training function for boosted tree
library(rmarkdown)
```

## Reading in Dataset

Now, we will read in our data set.

``` r
# reading in the data
data <- read_csv("../data/OnlineNewsPopularity.csv", show_col_types=FALSE)
head(data) #viewing the first 6 observations of the data
```

    ## # A tibble: 6 × 61
    ##   url           timedelta n_tokens_title n_tokens_content n_unique_tokens n_non_stop_words n_non_stop_unique_to…¹ num_hrefs num_self_hrefs num_imgs
    ##   <chr>             <dbl>          <dbl>            <dbl>           <dbl>            <dbl>                  <dbl>     <dbl>          <dbl>    <dbl>
    ## 1 http://masha…       731             12              219           0.664             1.00                  0.815         4              2        1
    ## 2 http://masha…       731              9              255           0.605             1.00                  0.792         3              1        1
    ## 3 http://masha…       731              9              211           0.575             1.00                  0.664         3              1        1
    ## 4 http://masha…       731              9              531           0.504             1.00                  0.666         9              0        1
    ## 5 http://masha…       731             13             1072           0.416             1.00                  0.541        19             19       20
    ## 6 http://masha…       731             10              370           0.560             1.00                  0.698         2              2        0
    ## # ℹ abbreviated name: ¹​n_non_stop_unique_tokens
    ## # ℹ 51 more variables: num_videos <dbl>, average_token_length <dbl>, num_keywords <dbl>, data_channel_is_lifestyle <dbl>,
    ## #   data_channel_is_entertainment <dbl>, data_channel_is_bus <dbl>, data_channel_is_socmed <dbl>, data_channel_is_tech <dbl>,
    ## #   data_channel_is_world <dbl>, kw_min_min <dbl>, kw_max_min <dbl>, kw_avg_min <dbl>, kw_min_max <dbl>, kw_max_max <dbl>, kw_avg_max <dbl>,
    ## #   kw_min_avg <dbl>, kw_max_avg <dbl>, kw_avg_avg <dbl>, self_reference_min_shares <dbl>, self_reference_max_shares <dbl>,
    ## #   self_reference_avg_sharess <dbl>, weekday_is_monday <dbl>, weekday_is_tuesday <dbl>, weekday_is_wednesday <dbl>, weekday_is_thursday <dbl>,
    ## #   weekday_is_friday <dbl>, weekday_is_saturday <dbl>, weekday_is_sunday <dbl>, is_weekend <dbl>, LDA_00 <dbl>, LDA_01 <dbl>, LDA_02 <dbl>, …

``` r
channel_vars <- data %>% select(starts_with("data_channel_is_")) %>% 
  pivot_longer(cols=everything(), names_to = "Channels")
#Selecting all of the variables in the imported data that start with 
#data_channel_is_ and pivoting the data to long format so that the 
#channel names can be extracted

strings <- unlist(strsplit(unique(channel_vars$Channels),split = '_'))
#Utilizing strsplit to split the data channel names by underscore

channel_names <- c()
#Creating empty atomic vector for loop below

for(i in 1:length(strings)){
  if(!(strings[i] %in% c("data", "channel", "is"))){
    #Checking to see if string is not equal to the uninterested strings above
    channel_names[i] <- strings[i]
    #saving the strings of interest in previously created atomic vector
  }
}
channel_names <- channel_names[!is.na(channel_names)] #removing NA values

output_file <- paste0(channel_names, ".html") 
#adding .html to end of channel names
params <- lapply(channel_names, FUN = function(x){list(channel = x)})
#creating params list for rendering step later on
reports <- tibble(output_file, params)
#creating report tibble for the output_file names and params
```

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
    ##   timedelta n_tokens_title n_tokens_content n_unique_tokens n_non_stop_words n_non_stop_unique_tokens num_hrefs num_self_hrefs num_imgs num_videos
    ##       <dbl>          <dbl>            <dbl>           <dbl>            <dbl>                    <dbl>     <dbl>          <dbl>    <dbl>      <dbl>
    ## 1       731              8              960           0.418             1.00                    0.550        21             20       20          0
    ## 2       731             10              187           0.667             1.00                    0.800         7              0        1          0
    ## 3       731             11              103           0.689             1.00                    0.806         3              1        1          0
    ## 4       731             10              243           0.619             1.00                    0.824         1              1        0          0
    ## 5       731              8              204           0.586             1.00                    0.698         7              2        1          0
    ## 6       731             11              315           0.551             1.00                    0.702         4              4        1          0
    ## # ℹ 45 more variables: average_token_length <dbl>, num_keywords <dbl>, data_channel_is_lifestyle <dbl>, kw_min_min <dbl>, kw_max_min <dbl>,
    ## #   kw_avg_min <dbl>, kw_min_max <dbl>, kw_max_max <dbl>, kw_avg_max <dbl>, kw_min_avg <dbl>, kw_max_avg <dbl>, kw_avg_avg <dbl>,
    ## #   self_reference_min_shares <dbl>, self_reference_max_shares <dbl>, self_reference_avg_sharess <dbl>, weekday_is_monday <dbl>,
    ## #   weekday_is_tuesday <dbl>, weekday_is_wednesday <dbl>, weekday_is_thursday <dbl>, weekday_is_friday <dbl>, weekday_is_saturday <dbl>,
    ## #   weekday_is_sunday <dbl>, is_weekend <dbl>, LDA_00 <dbl>, LDA_01 <dbl>, LDA_02 <dbl>, LDA_03 <dbl>, LDA_04 <dbl>, global_subjectivity <dbl>,
    ## #   global_sentiment_polarity <dbl>, global_rate_positive_words <dbl>, global_rate_negative_words <dbl>, rate_positive_words <dbl>,
    ## #   rate_negative_words <dbl>, avg_positive_polarity <dbl>, min_positive_polarity <dbl>, max_positive_polarity <dbl>, …

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

#### The following summarizations were created by Sandra Aziz

``` r
# summary stats for shares
summary_stats <- summary(channelTrain_summary$shares)
summary_stats
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##      28    1100    1700    3645    3225  196700

The statistics above explore the spread of the `shares` variable. If the
mean is less than the median, the data is skewed to the left. If the
median is less than the mean, then the data is skewed to the right.

``` r
# contingency table for number of videos
videos_table <- table(channelTrain_summary$num_videos)
videos_table
```

    ## 
    ##    0    1    2    3    4    5    6    7    8    9   10   11   12   15   26   50 
    ## 1140  230   50   18    5    7    2    4    1    1    5    5    1    1    1    1

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

#### The following summarizations were created by Simon Weisenhorn

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
    ## 1 Sunday      613          1300 3773.  2100          4075  33100  4718.
    ## 2 Monday      109          1000 4528.  1550          3125 196700 16088.
    ## 3 Tuesday     241          1000 3183.  1500          2800  54900  6145.
    ## 4 Wednesday    78          1000 3376.  1600          3000  73100  6229.
    ## 5 Thursday     28          1100 3666.  1700          3450  56000  6094.
    ## 6 Friday      127          1000 3078.  1500          2900  40400  4659.
    ## 7 Saturday    446          1500 4303.  2350          4100  43000  5869.

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
    ##   High Shares            472             273
    ##   Low Shares             456             271

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
    ##   High Shares          98         310                  337
    ##   Low Shares           90         361                  276

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
    ##   High Shares       571       117                 57
    ##   Low Shares        569       113                 45

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

    ## # A tibble: 8 × 8
    ##   num_keywords   Min firstQuartile   Avg   Med thirdQuartile    max stdDev
    ##          <dbl> <dbl>         <dbl> <dbl> <dbl>         <dbl>  <dbl>  <dbl>
    ## 1            3   720          1800 6878.  2350          8175  21300  8856.
    ## 2            4   595          1250 2233.  1450          2225  10800  2358.
    ## 3            5    28           896 2829.  1300          2650  22300  4108.
    ## 4            6    78           944 3235.  1400          2900  40400  5168.
    ## 5            7   128          1100 3431.  1600          2900  54200  6153.
    ## 6            8   109          1100 4136.  1700          3450 196700 13370.
    ## 7            9    95          1200 3750.  1800          3625  39900  5396.
    ## 8           10   258          1100 3670.  1700          3575 139600  8139.

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
    ## -12076  -1661      0   1515  40027 
    ## 
    ## Coefficients: (304 not defined because of singularities)
    ##                                                             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                                3.645e+03  1.772e+02  20.570  < 2e-16 ***
    ## timedelta                                                 -7.460e+09  2.775e+10  -0.269 0.788268    
    ## n_tokens_title                                             3.724e+09  1.385e+10   0.269 0.788266    
    ## n_tokens_content                                          -8.481e+04  5.631e+04  -1.506 0.133154    
    ## n_unique_tokens                                           -2.429e+04  5.766e+04  -0.421 0.673907    
    ## n_non_stop_words                                          -5.326e+10  1.981e+11  -0.269 0.788266    
    ## n_non_stop_unique_tokens                                  -3.893e+04  5.204e+04  -0.748 0.455033    
    ## num_hrefs                                                 -3.406e+04  3.345e+04  -1.018 0.309460    
    ## num_self_hrefs                                             2.738e+04  2.713e+04   1.009 0.313667    
    ## num_imgs                                                  -3.564e+09  1.326e+10  -0.269 0.788268    
    ## num_videos                                                 3.040e+09  1.131e+10   0.269 0.788264    
    ## average_token_length                                       3.103e+04  3.017e+04   1.029 0.304506    
    ## num_keywords                                               1.919e+09  7.138e+09   0.269 0.788265    
    ## data_channel_is_lifestyle                                         NA         NA      NA       NA    
    ## kw_min_min                                                -1.992e+11  7.410e+11  -0.269 0.788266    
    ## kw_max_min                                                -2.972e+10  1.106e+11  -0.269 0.788267    
    ## kw_avg_min                                                 3.625e+10  1.349e+11   0.269 0.788266    
    ## kw_min_max                                                 3.952e+09  1.470e+10   0.269 0.788262    
    ## kw_max_max                                                -1.696e+11  6.310e+11  -0.269 0.788266    
    ## kw_avg_max                                                -2.702e+09  1.005e+10  -0.269 0.788265    
    ## kw_min_avg                                                 3.976e+08  1.479e+09   0.269 0.788283    
    ## kw_max_avg                                                 9.735e+09  3.622e+10   0.269 0.788276    
    ## kw_avg_avg                                                -4.066e+09  1.513e+10  -0.269 0.788272    
    ## self_reference_min_shares                                 -1.166e+05  3.816e+05  -0.306 0.760190    
    ## self_reference_max_shares                                 -2.981e+05  7.873e+05  -0.379 0.705229    
    ## self_reference_avg_sharess                                 3.822e+05  9.948e+05   0.384 0.701073    
    ## weekday_is_monday                                          1.440e+04  2.940e+04   0.490 0.624740    
    ## weekday_is_tuesday                                         5.649e+04  2.891e+04   1.954 0.051720 .  
    ## weekday_is_wednesday                                       2.800e+04  3.141e+04   0.891 0.373494    
    ## weekday_is_thursday                                        5.695e+04  2.971e+04   1.917 0.056261 .  
    ## weekday_is_friday                                          2.894e+04  2.698e+04   1.073 0.284351    
    ## weekday_is_saturday                                        7.187e+04  2.943e+04   2.442 0.015194 *  
    ## weekday_is_sunday                                                 NA         NA      NA       NA    
    ## is_weekend                                                        NA         NA      NA       NA    
    ## LDA_00                                                    -2.324e+04  1.896e+04  -1.226 0.221132    
    ## LDA_01                                                     7.582e+03  1.955e+04   0.388 0.698356    
    ## LDA_02                                                     3.738e+03  2.077e+04   0.180 0.857299    
    ## LDA_03                                                     5.765e+04  2.505e+04   2.302 0.022067 *  
    ## LDA_04                                                            NA         NA      NA       NA    
    ## global_subjectivity                                       -3.322e+04  2.586e+04  -1.285 0.199963    
    ## global_sentiment_polarity                                  3.034e+04  4.851e+04   0.626 0.532129    
    ## global_rate_positive_words                                -2.797e+04  4.945e+04  -0.566 0.572074    
    ## global_rate_negative_words                                 4.329e+04  4.145e+04   1.044 0.297169    
    ## rate_positive_words                                        4.699e+04  5.954e+04   0.789 0.430605    
    ## rate_negative_words                                               NA         NA      NA       NA    
    ## avg_positive_polarity                                     -2.490e+04  3.587e+04  -0.694 0.488140    
    ## min_positive_polarity                                      2.720e+04  2.135e+04   1.274 0.203685    
    ## max_positive_polarity                                     -7.887e+03  2.727e+04  -0.289 0.772605    
    ## avg_negative_polarity                                     -2.017e+04  4.356e+04  -0.463 0.643746    
    ## min_negative_polarity                                     -6.152e+02  3.731e+04  -0.016 0.986854    
    ## max_negative_polarity                                      2.536e+03  3.593e+04   0.071 0.943773    
    ## title_subjectivity                                        -1.292e+04  3.318e+04  -0.389 0.697242    
    ## title_sentiment_polarity                                   1.650e+09  6.137e+09   0.269 0.788271    
    ## abs_title_subjectivity                                    -3.349e+04  2.269e+04  -1.476 0.140986    
    ## abs_title_sentiment_polarity                              -2.284e+04  4.386e+04  -0.521 0.602934    
    ## `timedelta:n_tokens_title`                                -5.165e+02  4.491e+03  -0.115 0.908509    
    ## `timedelta:n_tokens_content`                               6.824e+03  4.289e+03   1.591 0.112686    
    ## `timedelta:n_unique_tokens`                                1.643e+04  2.066e+04   0.795 0.427113    
    ## `timedelta:n_non_stop_words`                               7.514e+09  2.795e+10   0.269 0.788267    
    ## `timedelta:n_non_stop_unique_tokens`                      -2.238e+03  2.194e+04  -0.102 0.918807    
    ## `timedelta:num_hrefs`                                     -4.781e+03  2.747e+03  -1.740 0.082842 .  
    ## `timedelta:num_self_hrefs`                                -2.748e+03  3.752e+03  -0.732 0.464514    
    ## `timedelta:num_imgs`                                       2.968e+03  2.899e+03   1.024 0.306776    
    ## `timedelta:num_videos`                                     7.430e+03  3.811e+03   1.950 0.052197 .  
    ## `timedelta:average_token_length`                          -1.658e+04  1.869e+04  -0.887 0.375659    
    ## `timedelta:num_keywords`                                  -1.726e+02  5.048e+03  -0.034 0.972749    
    ## `timedelta:data_channel_is_lifestyle`                             NA         NA      NA       NA    
    ## `timedelta:kw_min_min`                                    -2.582e+04  2.962e+04  -0.872 0.384129    
    ## `timedelta:kw_max_min`                                    -4.800e+03  1.478e+04  -0.325 0.745645    
    ## `timedelta:kw_avg_min`                                     9.024e+03  1.353e+04   0.667 0.505288    
    ## `timedelta:kw_min_max`                                    -1.794e+03  2.552e+03  -0.703 0.482604    
    ## `timedelta:kw_max_max`                                    -1.650e+04  2.031e+04  -0.812 0.417338    
    ## `timedelta:kw_avg_max`                                    -1.614e+02  2.852e+03  -0.057 0.954918    
    ## `timedelta:kw_min_avg`                                    -1.562e+03  2.550e+03  -0.613 0.540610    
    ## `timedelta:kw_max_avg`                                    -4.422e+03  9.262e+03  -0.477 0.633432    
    ## `timedelta:kw_avg_avg`                                     2.447e+02  8.173e+03   0.030 0.976134    
    ## `timedelta:self_reference_min_shares`                      4.191e+04  3.235e+04   1.295 0.196284    
    ## `timedelta:self_reference_max_shares`                      9.443e+04  9.055e+04   1.043 0.297843    
    ## `timedelta:self_reference_avg_sharess`                    -1.254e+05  1.100e+05  -1.141 0.254992    
    ## `timedelta:weekday_is_monday`                              3.089e+03  2.779e+03   1.112 0.267244    
    ## `timedelta:weekday_is_tuesday`                             4.806e+03  2.960e+03   1.624 0.105538    
    ## `timedelta:weekday_is_wednesday`                           3.119e+03  3.072e+03   1.015 0.310954    
    ## `timedelta:weekday_is_thursday`                            3.328e+03  2.895e+03   1.150 0.251229    
    ## `timedelta:weekday_is_friday`                              1.963e+03  3.235e+03   0.607 0.544417    
    ## `timedelta:weekday_is_saturday`                            6.017e+02  1.933e+03   0.311 0.755861    
    ## `timedelta:weekday_is_sunday`                                     NA         NA      NA       NA    
    ## `timedelta:is_weekend`                                            NA         NA      NA       NA    
    ## `timedelta:LDA_00`                                        -1.429e+02  1.931e+03  -0.074 0.941060    
    ## `timedelta:LDA_01`                                        -1.736e+03  2.037e+03  -0.853 0.394600    
    ## `timedelta:LDA_02`                                         2.484e+03  1.995e+03   1.245 0.214234    
    ## `timedelta:LDA_03`                                         3.535e+03  2.002e+03   1.765 0.078563 .  
    ## `timedelta:LDA_04`                                                NA         NA      NA       NA    
    ## `timedelta:global_subjectivity`                            9.239e+03  8.004e+03   1.154 0.249347    
    ## `timedelta:global_sentiment_polarity`                      1.027e+04  7.708e+03   1.332 0.183897    
    ## `timedelta:global_rate_positive_words`                    -3.746e+03  7.722e+03  -0.485 0.627960    
    ## `timedelta:global_rate_negative_words`                    -3.261e+03  8.297e+03  -0.393 0.694595    
    ## `timedelta:rate_positive_words`                           -1.735e+04  2.503e+04  -0.693 0.488862    
    ## `timedelta:rate_negative_words`                                   NA         NA      NA       NA    
    ## `timedelta:avg_positive_polarity`                         -1.665e+04  9.817e+03  -1.696 0.090943 .  
    ## `timedelta:min_positive_polarity`                          3.443e+03  4.039e+03   0.852 0.394718    
    ## `timedelta:max_positive_polarity`                          5.342e+03  6.125e+03   0.872 0.383898    
    ## `timedelta:avg_negative_polarity`                         -1.044e+04  6.742e+03  -1.549 0.122426    
    ## `timedelta:min_negative_polarity`                         -7.638e+01  4.717e+03  -0.016 0.987092    
    ## `timedelta:max_negative_polarity`                          8.265e+03  3.414e+03   2.421 0.016096 *  
    ## `timedelta:title_subjectivity`                             1.085e+03  2.996e+03   0.362 0.717579    
    ## `timedelta:title_sentiment_polarity`                       3.523e+02  2.815e+03   0.125 0.900478    
    ## `timedelta:abs_title_subjectivity`                        -2.383e+02  3.040e+03  -0.078 0.937562    
    ## `timedelta:abs_title_sentiment_polarity`                  -3.132e+02  3.771e+03  -0.083 0.933859    
    ## `n_tokens_title:n_tokens_content`                          2.688e+03  6.359e+03   0.423 0.672864    
    ## `n_tokens_title:n_unique_tokens`                          -1.527e+04  1.368e+04  -1.116 0.265303    
    ## `n_tokens_title:n_non_stop_words`                         -4.157e+09  1.546e+10  -0.269 0.788265    
    ## `n_tokens_title:n_non_stop_unique_tokens`                  1.295e+04  1.328e+04   0.975 0.330465    
    ## `n_tokens_title:num_hrefs`                                -2.272e+00  4.401e+03  -0.001 0.999588    
    ## `n_tokens_title:num_self_hrefs`                           -9.862e+02  3.530e+03  -0.279 0.780191    
    ## `n_tokens_title:num_imgs`                                 -2.382e+03  5.701e+03  -0.418 0.676452    
    ## `n_tokens_title:num_videos`                                1.068e+04  7.084e+03   1.507 0.132826    
    ## `n_tokens_title:average_token_length`                      4.782e+03  1.024e+04   0.467 0.640895    
    ## `n_tokens_title:num_keywords`                             -7.109e+03  3.777e+03  -1.882 0.060800 .  
    ## `n_tokens_title:data_channel_is_lifestyle`                        NA         NA      NA       NA    
    ## `n_tokens_title:kw_min_min`                                3.558e+03  4.907e+03   0.725 0.468977    
    ## `n_tokens_title:kw_max_min`                                6.246e+04  3.100e+04   2.015 0.044830 *  
    ## `n_tokens_title:kw_avg_min`                               -5.101e+04  2.581e+04  -1.976 0.049062 *  
    ## `n_tokens_title:kw_min_max`                               -1.358e+03  5.442e+03  -0.250 0.803097    
    ## `n_tokens_title:kw_max_max`                                3.454e+03  7.075e+03   0.488 0.625769    
    ## `n_tokens_title:kw_avg_max`                               -1.293e+04  7.526e+03  -1.717 0.086982 .  
    ## `n_tokens_title:kw_min_avg`                               -3.683e+03  4.106e+03  -0.897 0.370411    
    ## `n_tokens_title:kw_max_avg`                               -4.960e+03  1.203e+04  -0.412 0.680354    
    ## `n_tokens_title:kw_avg_avg`                                1.750e+04  1.073e+04   1.630 0.104084    
    ## `n_tokens_title:self_reference_min_shares`                -1.764e+03  4.479e+04  -0.039 0.968614    
    ## `n_tokens_title:self_reference_max_shares`                 1.751e+04  9.230e+04   0.190 0.849633    
    ## `n_tokens_title:self_reference_avg_sharess`               -7.835e+03  1.160e+05  -0.068 0.946197    
    ## `n_tokens_title:weekday_is_monday`                        -8.974e+03  3.549e+03  -2.529 0.011975 *  
    ## `n_tokens_title:weekday_is_tuesday`                       -1.990e+03  3.371e+03  -0.590 0.555484    
    ## `n_tokens_title:weekday_is_wednesday`                     -7.133e+03  3.736e+03  -1.909 0.057237 .  
    ## `n_tokens_title:weekday_is_thursday`                      -9.013e+03  3.447e+03  -2.615 0.009392 ** 
    ## `n_tokens_title:weekday_is_friday`                        -6.516e+03  3.467e+03  -1.879 0.061203 .  
    ## `n_tokens_title:weekday_is_saturday`                      -4.855e+03  3.541e+03  -1.371 0.171388    
    ## `n_tokens_title:weekday_is_sunday`                                NA         NA      NA       NA    
    ## `n_tokens_title:is_weekend`                                       NA         NA      NA       NA    
    ## `n_tokens_title:LDA_00`                                   -4.614e+03  2.621e+03  -1.760 0.079384 .  
    ## `n_tokens_title:LDA_01`                                    5.727e+02  2.889e+03   0.198 0.842986    
    ## `n_tokens_title:LDA_02`                                    2.248e+03  2.433e+03   0.924 0.356186    
    ## `n_tokens_title:LDA_03`                                   -5.382e+03  3.615e+03  -1.489 0.137667    
    ## `n_tokens_title:LDA_04`                                           NA         NA      NA       NA    
    ## `n_tokens_title:global_subjectivity`                       1.215e+04  5.476e+03   2.218 0.027312 *  
    ## `n_tokens_title:global_sentiment_polarity`                -1.055e+04  7.349e+03  -1.435 0.152388    
    ## `n_tokens_title:global_rate_positive_words`                5.857e+03  6.551e+03   0.894 0.372097    
    ## `n_tokens_title:global_rate_negative_words`               -7.827e+03  9.204e+03  -0.850 0.395766    
    ## `n_tokens_title:rate_positive_words`                      -5.074e+03  1.594e+04  -0.318 0.750433    
    ## `n_tokens_title:rate_negative_words`                              NA         NA      NA       NA    
    ## `n_tokens_title:avg_positive_polarity`                    -1.812e+03  6.251e+03  -0.290 0.772164    
    ## `n_tokens_title:min_positive_polarity`                     3.881e+03  3.485e+03   1.114 0.266298    
    ## `n_tokens_title:max_positive_polarity`                     9.949e+03  4.935e+03   2.016 0.044725 *  
    ## `n_tokens_title:avg_negative_polarity`                     6.061e+01  6.123e+03   0.010 0.992110    
    ## `n_tokens_title:min_negative_polarity`                     4.632e+03  4.784e+03   0.968 0.333795    
    ## `n_tokens_title:max_negative_polarity`                    -1.753e+03  4.743e+03  -0.370 0.711946    
    ## `n_tokens_title:title_subjectivity`                        7.878e+02  4.097e+03   0.192 0.847653    
    ## `n_tokens_title:title_sentiment_polarity`                 -1.745e+03  5.095e+03  -0.342 0.732259    
    ## `n_tokens_title:abs_title_subjectivity`                    2.169e+02  3.253e+03   0.067 0.946866    
    ## `n_tokens_title:abs_title_sentiment_polarity`              1.825e+02  5.460e+03   0.033 0.973351    
    ## `n_tokens_content:n_unique_tokens`                        -3.590e+03  9.630e+03  -0.373 0.709577    
    ## `n_tokens_content:n_non_stop_words`                               NA         NA      NA       NA    
    ## `n_tokens_content:n_non_stop_unique_tokens`                2.021e+04  1.480e+04   1.365 0.173178    
    ## `n_tokens_content:num_hrefs`                               2.932e+03  4.304e+03   0.681 0.496257    
    ## `n_tokens_content:num_self_hrefs`                          3.704e+03  3.618e+03   1.024 0.306801    
    ## `n_tokens_content:num_imgs`                                3.165e+03  8.540e+03   0.371 0.711228    
    ## `n_tokens_content:num_videos`                             -6.692e+03  3.590e+03  -1.864 0.063342 .  
    ## `n_tokens_content:average_token_length`                   -2.427e+04  2.439e+04  -0.995 0.320481    
    ## `n_tokens_content:num_keywords`                            8.731e+03  8.342e+03   1.047 0.296160    
    ## `n_tokens_content:data_channel_is_lifestyle`                      NA         NA      NA       NA    
    ## `n_tokens_content:kw_min_min`                              1.991e+02  4.533e+03   0.044 0.964988    
    ## `n_tokens_content:kw_max_min`                              1.239e+04  1.945e+04   0.637 0.524635    
    ## `n_tokens_content:kw_avg_min`                             -1.966e+04  1.701e+04  -1.156 0.248586    
    ## `n_tokens_content:kw_min_max`                              1.274e+03  3.916e+03   0.325 0.745198    
    ## `n_tokens_content:kw_max_max`                             -8.051e+03  1.376e+04  -0.585 0.559020    
    ## `n_tokens_content:kw_avg_max`                              1.981e+03  8.650e+03   0.229 0.818979    
    ## `n_tokens_content:kw_min_avg`                             -5.159e+03  4.112e+03  -1.255 0.210651    
    ## `n_tokens_content:kw_max_avg`                             -1.164e+04  1.087e+04  -1.070 0.285385    
    ## `n_tokens_content:kw_avg_avg`                              1.910e+04  1.624e+04   1.176 0.240492    
    ## `n_tokens_content:self_reference_min_shares`              -9.955e+03  2.462e+04  -0.404 0.686257    
    ## `n_tokens_content:self_reference_max_shares`              -9.555e+03  5.705e+04  -0.167 0.867110    
    ## `n_tokens_content:self_reference_avg_sharess`              2.260e+04  6.785e+04   0.333 0.739335    
    ## `n_tokens_content:weekday_is_monday`                      -6.470e+03  2.593e+03  -2.495 0.013139 *  
    ## `n_tokens_content:weekday_is_tuesday`                     -3.865e+03  2.686e+03  -1.439 0.151206    
    ## `n_tokens_content:weekday_is_wednesday`                   -3.092e+03  3.613e+03  -0.856 0.392838    
    ## `n_tokens_content:weekday_is_thursday`                    -4.519e+03  2.938e+03  -1.538 0.125173    
    ## `n_tokens_content:weekday_is_friday`                      -3.724e+03  3.117e+03  -1.194 0.233281    
    ## `n_tokens_content:weekday_is_saturday`                    -3.916e+03  3.658e+03  -1.071 0.285192    
    ## `n_tokens_content:weekday_is_sunday`                              NA         NA      NA       NA    
    ## `n_tokens_content:is_weekend`                                     NA         NA      NA       NA    
    ## `n_tokens_content:LDA_00`                                  2.684e+03  2.563e+03   1.047 0.295893    
    ## `n_tokens_content:LDA_01`                                 -3.460e+03  2.483e+03  -1.394 0.164499    
    ## `n_tokens_content:LDA_02`                                  5.582e+03  2.279e+03   2.449 0.014921 *  
    ## `n_tokens_content:LDA_03`                                  3.504e+03  4.194e+03   0.836 0.404101    
    ## `n_tokens_content:LDA_04`                                         NA         NA      NA       NA    
    ## `n_tokens_content:global_subjectivity`                     8.410e+03  1.464e+04   0.575 0.566050    
    ## `n_tokens_content:global_sentiment_polarity`              -1.676e+04  1.370e+04  -1.224 0.222036    
    ## `n_tokens_content:global_rate_positive_words`             -5.161e+03  1.712e+04  -0.302 0.763206    
    ## `n_tokens_content:global_rate_negative_words`              9.349e+03  1.411e+04   0.662 0.508244    
    ## `n_tokens_content:rate_positive_words`                     5.066e+04  4.811e+04   1.053 0.293228    
    ##  [ reached getOption("max.print") -- omitted 1286 rows ]
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 6798 on 290 degrees of freedom
    ## Multiple R-squared:  0.8673, Adjusted R-squared:  0.327 
    ## F-statistic: 1.605 on 1181 and 290 DF,  p-value: 6.494e-07

``` r
#viewing the resulting model

pred <- predict(simon_share_lmfit, newdata = channelTest)
#Creating predictions based on the channelTest data
l1 <- postResample(pred, obs = channelTest$shares)
#creating first linear model postResample results object
l1 #viewing how well the first linear model did 
```

    ##         RMSE     Rsquared          MAE 
    ## 1.156185e+09 3.200816e-03 8.854044e+07

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
    ## -27948  -2658  -1318    427 181214 
    ## 
    ## Coefficients: (5 not defined because of singularities)
    ##                              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                   3644.63     213.59  17.064  < 2e-16 ***
    ## timedelta                      928.40     389.14   2.386   0.0172 *  
    ## n_tokens_title                 231.54     228.88   1.012   0.3119    
    ## n_tokens_content               581.36     401.18   1.449   0.1475    
    ## n_unique_tokens              -1302.73     862.85  -1.510   0.1313    
    ## n_non_stop_words              -999.49     723.00  -1.382   0.1671    
    ## n_non_stop_unique_tokens      1717.51     786.22   2.185   0.0291 *  
    ## num_hrefs                      307.98     279.50   1.102   0.2707    
    ## num_self_hrefs                -157.03     238.93  -0.657   0.5111    
    ## num_imgs                        87.21     352.16   0.248   0.8045    
    ## num_videos                    1066.51     225.33   4.733 2.43e-06 ***
    ## average_token_length           527.58     513.59   1.027   0.3045    
    ## num_keywords                   138.49     263.93   0.525   0.5998    
    ## data_channel_is_lifestyle          NA         NA      NA       NA    
    ## kw_min_min                    -253.33     417.24  -0.607   0.5438    
    ## kw_max_min                    -627.47     971.38  -0.646   0.5184    
    ## kw_avg_min                     439.58     983.30   0.447   0.6549    
    ## kw_min_max                     186.16     284.81   0.654   0.5135    
    ## kw_max_max                     126.26     495.92   0.255   0.7991    
    ## kw_avg_max                      49.07     527.84   0.093   0.9259    
    ## kw_min_avg                    -533.95     341.89  -1.562   0.1186    
    ## kw_max_avg                   -1105.21     561.52  -1.968   0.0492 *  
    ## kw_avg_avg                    1657.92     688.60   2.408   0.0162 *  
    ## self_reference_min_shares     1103.28     660.76   1.670   0.0952 .  
    ## self_reference_max_shares      602.89    1061.29   0.568   0.5701    
    ## self_reference_avg_sharess   -1321.97    1439.77  -0.918   0.3587    
    ## weekday_is_monday              221.29     337.84   0.655   0.5126    
    ## weekday_is_tuesday            -302.90     339.02  -0.893   0.3718    
    ## weekday_is_wednesday          -260.05     350.74  -0.741   0.4586    
    ## weekday_is_thursday            -75.13     344.41  -0.218   0.8274    
    ## weekday_is_friday             -310.45     330.95  -0.938   0.3484    
    ## weekday_is_saturday             37.32     285.59   0.131   0.8961    
    ## weekday_is_sunday                  NA         NA      NA       NA    
    ## is_weekend                         NA         NA      NA       NA    
    ## LDA_00                         261.44     232.14   1.126   0.2602    
    ## LDA_01                        -148.18     224.96  -0.659   0.5102    
    ## LDA_02                        -171.52     239.53  -0.716   0.4741    
    ## LDA_03                        -358.04     293.56  -1.220   0.2228    
    ## LDA_04                             NA         NA      NA       NA    
    ## global_subjectivity            377.69     337.81   1.118   0.2637    
    ## global_sentiment_polarity       20.82     622.68   0.033   0.9733    
    ## global_rate_positive_words     284.83     463.38   0.615   0.5389    
    ## global_rate_negative_words    -312.06     643.69  -0.485   0.6279    
    ## rate_positive_words           -877.38     840.81  -1.043   0.2969    
    ## rate_negative_words                NA         NA      NA       NA    
    ## avg_positive_polarity          298.03     491.67   0.606   0.5445    
    ## min_positive_polarity         -171.24     285.82  -0.599   0.5492    
    ## max_positive_polarity          -70.11     355.62  -0.197   0.8437    
    ## avg_negative_polarity         -315.67     543.57  -0.581   0.5615    
    ## min_negative_polarity          514.38     449.58   1.144   0.2528    
    ## max_negative_polarity           16.97     379.53   0.045   0.9643    
    ## title_subjectivity             461.40     327.11   1.411   0.1586    
    ## title_sentiment_polarity      -121.86     284.78  -0.428   0.6688    
    ## abs_title_subjectivity         479.67     256.82   1.868   0.0620 .  
    ## abs_title_sentiment_polarity  -188.88     360.28  -0.524   0.6002    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 8195 on 1422 degrees of freedom
    ## Multiple R-squared:  0.05459,    Adjusted R-squared:  0.02201 
    ## F-statistic: 1.676 on 49 and 1422 DF,  p-value: 0.002645

``` r
#viewing the resulting model

pred <- predict(simon_share_lmfit, newdata = channelTest)
#Creating predictions based on the channelTest data
l2 <- postResample(pred, obs = channelTest$shares)
#creating second linear model postResample results object
l2 #viewing how well the second linear model did 
```

    ##         RMSE     Rsquared          MAE 
    ## 1.156185e+09 3.200816e-03 8.854044e+07

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
    ## 1472 samples
    ##   54 predictor
    ## 
    ## Pre-processing: centered (54), scaled (54) 
    ## Resampling: Cross-Validated (5 fold, repeated 3 times) 
    ## Summary of sample sizes: 1177, 1178, 1177, 1178, 1178, 1177, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE      Rsquared     MAE     
    ##   1                   25      7766.313  0.002376522  3277.276
    ##   1                   50      7824.535  0.002354536  3299.450
    ##   1                  100      7898.540  0.002134645  3348.569
    ##   1                  150      7951.666  0.002402631  3391.540
    ##   1                  200      7981.544  0.001738577  3424.843
    ##   2                   25      7785.920  0.002459903  3275.900
    ##   2                   50      7885.925  0.002683881  3341.709
    ##   2                  100      7969.602  0.004551328  3420.059
    ##   2                  150      8037.502  0.005054204  3491.484
    ##   2                  200      8107.067  0.004809578  3536.655
    ##   3                   25      7827.837  0.002800863  3299.958
    ##   3                   50      7888.354  0.005349167  3347.298
    ##   3                  100      8084.176  0.003567994  3488.927
    ##   3                  150      8148.721  0.003593131  3572.588
    ##   3                  200      8196.094  0.005244976  3644.313
    ##   4                   25      7827.903  0.002508070  3293.118
    ##   4                   50      7944.465  0.003048037  3370.360
    ##   4                  100      8058.739  0.005913457  3469.663
    ##   4                  150      8184.879  0.005652855  3578.494
    ##   4                  200      8245.212  0.005880425  3666.351
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 25, interaction.depth = 1, shrinkage = 0.1 and n.minobsinnode = 10.

``` r
#viewing the resulting model

pred <- predict(boostTreeFit, newdata = channelTest)
#Creating predictions based on the channelTest data
bt <- postResample(pred, obs = channelTest$shares)
#creating boosted tree postResample results object
bt #viewing how well the boosted tree did 
```

    ##         RMSE     Rsquared          MAE 
    ## 1.017066e+04 3.016651e-03 3.322543e+03

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
    ## 1472 samples
    ##   54 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold, repeated 3 times) 
    ## Summary of sample sizes: 1177, 1178, 1177, 1179, 1177, 1178, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  RMSE      Rsquared     MAE     
    ##    1    7677.197  0.004560645  3307.569
    ##    2    7722.571  0.006104627  3397.121
    ##    3    7790.650  0.004171842  3446.579
    ##    4    7827.086  0.005042071  3477.581
    ##    5    7858.409  0.004679543  3496.628
    ##    6    7907.357  0.004350704  3514.520
    ##    7    7965.106  0.003119229  3540.059
    ##    8    8019.285  0.003414720  3556.642
    ##    9    8031.234  0.003972484  3544.367
    ##   10    8098.413  0.003598575  3578.969
    ##   11    8147.836  0.002909829  3584.475
    ##   12    8167.920  0.003111121  3592.052
    ##   13    8224.042  0.003061524  3606.455
    ##   14    8257.197  0.002929499  3616.836
    ##   15    8301.542  0.002783551  3618.513
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
    ## 1.007957e+04 1.655040e-02 3.361588e+03

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
