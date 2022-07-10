require("dplyr")
require("tidyr")
require("purrr")

data <- read.csv("../Results/SampleResults.csv") %>% mutate(guided = "E_guided")
data2 <- read.csv("../Results/NoSampleResults.csv") %>% mutate(guided = "E_none")
data <- data %>% bind_rows(data2)


#  pivot_wider(names_from = guided, values_from = E)

data <- data %>%
  mutate(E = - (d_g0 / base_g0) - abs(d_gX / base_gX)) %>%
  select(dataset, ptb_rate, E, guided) %>%
  group_by(dataset, ptb_rate, guided) %>%
  summarize_all(mean) %>%
  pivot_wider(names_from = guided, values_from = E)

write.csv(data, "./out.csv")