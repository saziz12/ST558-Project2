for(i in 1:length(channel_names)){
  rmarkdown::render(input = "code/index.Rmd", output_file = channel_names[i], 
                    params=list(data_channel=channel_names[i]), 
                    output_format = "github_document", 
                    output_options = list(html_preview=FALSE))
}
