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

`library(tidyverse)`  

`data <- read_csv("data/OnlineNewsPopularity.csv", show_col_types=FALSE)`  

`channel_vars <- data %>% select(starts_with("data_channel_is_")) %>%   
  pivot_longer(cols=everything(), names_to = "Channels")`  

`strings <- unlist(strsplit(unique(channel_vars$Channels),split = '_'))`  

`channel_names <- c()`  

`for(i in 1:length(strings))`{  
  `if(!(strings[i] %in% c("data", "channel", "is")))`{  
    `channel_names[i] <- strings[i]`  
  }  
}  
`channel_names <- channel_names[!is.na(channel_names)]`  

`for(i in 1:length(channel_names))`{  
  `params<<-list(data_channel=channel_names[i])`  
  `rmarkdown::render(input = "index.Rmd", output_file = channel_names[i],  
                    params=params,  
                    output_format = "github_document",  
                    output_options = list(html_preview=FALSE))`  
}  
  
