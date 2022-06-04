require("dplyr")
require("tidyr")
require("purrr")
require("ggplot2")
require("ggthemes")

rawCora <- read.csv(file="FeatureSelection/featureResults-cora.csv") %>% distinct()
rawBlog <- read.csv(file="FeatureSelection/featureResults-BlogCatalog.csv") %>% distinct()

rawMetrics <- read.csv(file="FeatureSelection/featureMetrics.csv") %>% distinct()

get_sign <- function(delta) {
  if (abs(delta) < 0.05) return(0);
  return(sign(delta));
}

is_effective <- function(d_g0, d_gX) {
  if (d_g0 < -0.05) {
    if (abs(d_g0) / abs(d_gX) > 1.5) {
      return(1)
    }
  }
  return(0)
}

joined <- bind_rows(rawCora, rawBlog) %>% 
  left_join(rawMetrics, by=c('dataset', 'feat_idx')) %>%
  rowwise %>%
  mutate(
    delta_g0 = (lock_g0 - base_g0) / base_g0,
    delta_gX = (lock_gX - base_gX) / base_gX,
    effective = is_effective((lock_g0 - base_g0) / base_g0, (lock_gX - base_gX) / base_gX)
  ) %>% arrange(effective) #%>% filter(effective == 1)
  #pivot_longer(c("delta_g0", "delta_gX"), names_to="location", values_to="delta") %>% 
  #pivot_longer(c("information_gain","mutual_information","chi_sq"), names_to="type", values_to="value")
  
p <- ggplot(joined, aes(x=entropy, y=chi_sq, color=factor(effective))) +
  facet_grid(. ~ dataset) +
  ggtitle("Effective features by entropy and chi_sq") +
  geom_point(size=0.75) + 
  scale_color_manual(values=c("#dddddd", "black")) +
  theme_linedraw()


show(p)