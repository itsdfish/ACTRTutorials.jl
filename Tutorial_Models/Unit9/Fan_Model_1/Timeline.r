library(ggthemes) # Load
library(tidyverse)
library(ggplot2)
gantt <- read.csv("TimeData.csv", h=T)
acts <- c("Procedural","Declarative","Imaginal","Visual","Motor")
els <-  c("Procedural","Declarative","Imaginal","Visual","Motor")
g.gantt <- gather(gantt, "state", "date", 4:5) %>% mutate(Activity=factor(Module, acts[length(acts):1]), Project.element=factor(Color, els))
ggplot(g.gantt, aes(date, Activity, color = Project.element, group=Item)) +
  geom_line(size = 15) +
  labs(x="Time (seconds)", y=NULL) + 
  theme(text = element_text(size=12),axis.text=element_text(size=10),legend.position = "none", panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black")) +
 scale_colour_economist()
#Sized to look good at 300 by 200 pixels
ggsave("TimeCourse.png", width = 6, height = 3, units = "in")

# cr 50
# find person 0 (200 imaginal p)
# cr 50
# attend location 85
# cr 50
# retreive meaning 0
# finish imaginal 15
# cr 50
# encode person imaginal 0
# cr 50
# attend location 85
# cr 50
# retrieve meaning 0
# cr 50
# encode location imaginal 0
# cr 50
# retrieve fact -
# cr 50
# respond ~ 210

