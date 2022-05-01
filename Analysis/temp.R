library('dplyr')
library('ggplot2')
library('ggthemes')

data <- read.csv("temp.csv", sep="\t", as.is=FALSE) %>%
  mutate(year=rownames(data), breaches=as.numeric(as.character(breaches)))

p <- ggplot(data, aes(x=year, y=breaches)) +
  geom_bar(stat="identity") +
  theme_hc() +
  xlab('') +
  ylab('Breaches')

print(p)
