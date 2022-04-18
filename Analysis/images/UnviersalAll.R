library('tidyr')
library('ggplot2')
library('ggthemes')

raw <- read.csv("~/HCDM/Results/UniversalResults.csv") 

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

data <- raw %>%
  group_by(dataset, ptb_rate, feature) %>%
  filter(feature != -1) %>%
  summarise_all(mean) %>%
  rowwise() %>%
  mutate(effective=(is_effective(d_g0, d_gX))) %>%
  pivot_longer(c("d_g0", "d_gX"), names_to="location", values_to="delta") %>%
  rowwise() %>%
  mutate(sign=get_sign(delta))
  
p <- ggplot(data, aes(x=entropy, y=corr, fill=delta, shape=factor(sign), label=feature, color=factor(effective))) +
  facet_grid(ptb_rate ~ location) +
  geom_point(size=1.5) +
  scale_fill_gradient2(
    low = "red",
    mid = "yellow",
    high = "green",
    midpoint = 0,
    limits=c(-0.25, 0.25)
  ) + 
  scale_colour_manual(values = c("black", "red")) +
  scale_shape_manual(values = c(25, 21, 24)) +
  theme_light() +
  expand_limits(x=c(0, max(data$entropy) * 1.2), y=c(0,  max(data$corr) * 1.2)) +
  ggtitle(paste("Universal Task Evaluation"))
  
print(p)
#ggsave(paste("./images/Universal_", target_dataset, ".png", sep=""), p, device="png", width=7, height=5, units="in", dpi=300)
