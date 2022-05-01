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

for (target_ptb in c(0.25, 0.5)) {
  data <- raw %>%
    group_by(dataset, ptb_rate, feature) %>%
    filter(feature != -1) %>%
    filter(ptb_rate == target_ptb) %>%
    summarise_all(mean) %>%
    rowwise() %>%
    mutate(effective=(is_effective(d_g0, d_gX))) %>%
    filter(effective==1) %>%
    pivot_longer(c("d_g0", "d_gX"), names_to="location", values_to="delta") %>%
    rowwise() %>%
    filter(location=="d_g0") %>%
    mutate(sign=get_sign(delta))
  
  p <- ggplot(data, aes(x=entropy, y=corr, label=feature, color=dataset)) +
    geom_point(size=1.5) +
    theme_light() +
    expand_limits(x=c(0, max(data$entropy) * 1.2), y=c(0,  max(data$corr) * 1.2)) +
    ggtitle(paste("All effective features", target_ptb))
  
  print(p)
  ggsave(paste("./images/Universal/All_", target_ptb, ".png", sep=""), p, device="png", width=7, height=5, units="in", dpi=300)
}
