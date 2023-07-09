# ST558-Project2

### Purpose of this Repo

This repo was created to showcase all of the reports made from a single .Rmd file for the different 
data channels within the online news popularity dataset provided by the UC Irvine Machine Learning 
Repository. The single .Rmd file was coded and automated to take in the dataset and subset it based 
on a channel parameter so that summarizations and models could be used to explore the relationship 
between the number of shares an article received and all other predictive variables in the dataset.
A link to each report can be found below.

### List of R Packages Used  

  * `tidyverse`
  * `ggplot2`
  * `stringr`
  * `caret`
  * `gbm`  

### Links to Reports

[Lifestyle report is available here](lifestyle.md)
  
[Entertainment report is available here](entertainment.md)
  
[Business report is available here](bus.md)
  
[Social Media report is available here](socmed.md)
  
[Tech report is available here](tech.md)
  
[World report is available here](world.md)

### Code Used to Create the Analyses

`
library(tidyverse) #calling appropriate library

#The following approach allows for future data channels to be added to this
#dataset and still work

data <- read_csv("data/OnlineNewsPopularity.csv", show_col_types=FALSE)
#reading in data

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

for(i in 1:length(channel_names)){
  
  params<<-list(data_channel=channel_names[i])
  
  rmarkdown::render(input = "index.Rmd", output_file = channel_names[i], 
                    params=params, 
                    output_format = "github_document", 
                    output_options = list(html_preview=FALSE))
}
`
