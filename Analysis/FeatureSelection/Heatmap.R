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
      return('effective')
    }
  }
  return('ineffective')
}

tile_density_entropy <- 7
tile_density_chisq <- 15

joined <- bind_rows(rawCora, rawBlog) %>% 
  left_join(rawMetrics, by=c('dataset', 'feat_idx')) %>%
  rowwise %>%
  mutate(
    delta_g0 = (lock_g0 - base_g0) / base_g0,
    delta_gX = (lock_gX - base_gX) / base_gX,
    effective = is_effective((lock_g0 - base_g0) / base_g0, (lock_gX - base_gX) / base_gX)
  ) %>% arrange(effective) %>%
  mutate (
    chi_sq = floor(chi_sq * tile_density_chisq) / tile_density_chisq,
    entropy = floor(entropy * tile_density_entropy) / tile_density_entropy,
  ) %>% 
  group_by(chi_sq, entropy, effective, dataset) %>% 
  summarize(count = n()) %>%
  pivot_wider(names_from=effective, values_from=count) %>%
  replace(is.na(.), 0) %>%
  mutate(density = effective / (effective + ineffective))
#%>% filter(effective == 1)
#pivot_longer(c("delta_g0", "delta_gX"), names_to="location", values_to="delta") %>% 
#pivot_longer(c("information_gain","mutual_information","chi_sq"), names_to="type", values_to="value")

p <- ggplot(joined, aes(x=entropy, y=chi_sq, fill=density)) +
  ggtitle("relative frequency of 'effective' features by entropy and chi_sq") +
  facet_grid(. ~ dataset) +
  geom_tile() + 
  scale_fill_gradient(low = "yellow", high = "red") +
  theme_linedraw()


show(p)