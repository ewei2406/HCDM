require("dplyr")
require("tidyr")
require("purrr")
require("ggplot2")
require("ggthemes")

inputFile <- "SelectiveAttack.csv"

rawData <- read.csv(file="../Results/" %>% paste(inputFile, sep=""))

fixParentheses <- function(d) {
  stripParentheses <- function(s) {
    return(strsplit(gsub(" ", "", gsub("% similar", "", s)), '[()]'))
  }
  
  fixAddRem <- function(s) {
    return(as.double(stripParentheses(s) %>% map(., 1)))
  }
  
  getSim <- function(s) {
    return(as.double(stripParentheses(s) %>% map(., ~ pluck(.x, length(.x)))) / 100)
  }
  
  fixAdd <- d %>% mutate(
    add_g0_similar = getSim(add_g0),
    add_g0 = fixAddRem(add_g0),
    add_gX_similar = getSim(add_gX),
    add_gX = fixAddRem(add_gX),
    add_g0gX_similar = getSim(add_g0gX),
    add_g0gX = fixAddRem(add_g0gX),
    
    remove_g0_similar = getSim(remove_g0),
    remove_g0 = fixAddRem(remove_g0),
    remove_gX_similar = getSim(remove_gX),
    remove_gX = fixAddRem(remove_gX),
    remove_g0gX_similar = getSim(remove_g0gX),
    remove_g0gX = fixAddRem(remove_g0gX),
  )
  
  return(fixAdd)
}

cleanedData <- rawData %>% fixParentheses()

byDataset <- cleanedData %>% 
  group_by(dataset, ptb_rate) %>% 
  summarize(sd=sd(d_g0), d_g0=mean(d_g0), d_gX=mean(d_gX)) %>%
  pivot_longer(c(d_g0, d_gX), names_to="location", values_to="delta")

gg <- ggplot(byDataset, aes(x=ptb_rate, y=delta, group=interaction(dataset, location), shape=location, color=dataset)) +
  facet_grid(. ~ location) +
  geom_ribbon(aes(ymin=delta-sd/2, ymax=delta+sd/2), alpha=0.1, color=NA, fill="grey") +
  guides(fill="none") +
  geom_line() +
  labs(title='∆ acc(G) by ptb_rate: ' %>% paste(inputFile, sep="")) + 
  geom_point(size=2, fill="white") +
  xlab("ptb_rate") +
  ylab("∆") +
  theme_minimal() +
  scale_shape_manual(values = c(21, 16)) +
  theme(plot.margin = margin(1,1,1.5,1.2, "cm"))
  
show(gg)
ggsave("./images/" %>% paste("DeltaG", inputFile, ".png", sep=""), units="in", dpi=300, width=7, height=5)
