require("dplyr")
require("tidyr")
require("purrr")
require("ggplot2")
require("ggthemes")

inputFile <- "SampleResults.csv"

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

cleanedData <- rawData %>% fixParentheses() %>% select(-date)

normalize <- function(d, b) {
  return(d/b)
}

byDataset <- cleanedData %>% 
  group_by(dataset, ptb_rate) %>%
  summarize(across(c(starts_with("add_"), starts_with("remove_")), mean)) %>%
  mutate(total_add=(add_g0 + add_gX + add_g0gX), total_remove=(remove_g0 + remove_gX + remove_g0gX)) %>%
  mutate(across(c("add_g0", "add_gX", "add_g0gX"), ~ ./total_add)) %>%
  mutate(across(c("remove_g0", "remove_gX", "remove_g0gX"), ~ ./total_remove)) %>%
  pivot_longer(c(starts_with("add_"), starts_with("remove_")), values_to="ratio", names_to="location") %>%
  separate(location, c('operation', 'location', 'type'), sep="_") %>% 
  mutate(type=replace_na(type, "r")) %>%
  pivot_wider(names_from=type, values_from=ratio) %>%
  mutate(r=replace_na(r, 0))

# Location

gg <- ggplot(byDataset, aes(x=r, y=dataset, fill=location)) +
  facet_grid(ptb_rate ~ operation) +
  geom_bar(position="fill", stat="identity") +
  theme_minimal() +
  ggtitle(paste("Change Locations: ", inputFile, sep="")) +
  theme(plot.margin = margin(1,1,1.5,1.2, "cm"))

show(gg)

ggsave("./images/" %>% paste("ChangeLocations", inputFile, ".png", sep=""), units="in", dpi=300, width=7, height=5)

# Similarity

gg <- ggplot(byDataset, aes(x=r, y=dataset, fill=location)) +
  facet_grid(ptb_rate ~ operation) +
  geom_bar(aes(x=similar, y=dataset), position="dodge", stat="identity") +
  theme_minimal() +
  ggtitle(paste("Similarity of Changes: ", inputFile, sep="")) +
  theme(plot.margin = margin(1,1,1.5,1.2, "cm"))

show(gg)

ggsave("./images/" %>% paste("SimilarityLocation", inputFile, ".png", sep=""), units="in", dpi=300, width=7, height=5)
