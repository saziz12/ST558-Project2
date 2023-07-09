
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
