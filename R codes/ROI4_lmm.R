library(lme4)
library(lmerTest)
library(multcompView)
library(car)
library(ggplot2)
library(DescTools)
library(emmeans)
library(multcomp)
library(report)

data_roi4= read.csv("E:\\SocialDetection7T\\Nahid_Project\\IS_RSA\\ROI_LMM\\ROI_4_LMM.csv")






#Build model

roi4_full<- lmer(Similarity~ Group+Treatment + Group*Treatment + (1|Sub_ID), data = data_roi4)
roi4_0<-lmer(Similarity~ Treatment + (1|Sub_ID), data = data_roi4)
roi4_1<- lmer(Similarity~ Group + (1|Sub_ID), data = data_roi4)

#model comparison
model_com= anova(roi4_0, roi4_1, roi4_full)
write.csv(model_com, "E:\\SocialDetection7T\\Nahid_Project\\IS_RSA\\ROI_LMM\\model_comp_ROI4.csv")


#check the summary
summary(roi4_full)

#make table of the summary
roi4_report=report_table(roi4_full)
#write csv
write.csv(roi4_report, "E:\\SocialDetection7T\\Nahid_Project\\IS_RSA\\ROI_LMM\\lmm_result_ROI4.csv")

#analysis of deviance
aod= Anova(roi4_full, type = 'III')
write.csv(aod, "E:\\SocialDetection7T\\Nahid_Project\\IS_RSA\\ROI_LMM\\aod_ROI4.csv")

text_result= report_text(roi4_full)
text_result
emm_roi4= emmeans(roi4_full, pairwise~ Group*Treatment, adjust= 'bonferroni', lmer.df= 'asymp')
# for bonf correction 0.05/ number of contrasts * 3 because we have 3 represenational hypothesis
emm_roi4
letters= c('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i')
cld_emo=cld(emm_roi4, alpha= 0.05, Letters=letters)
write.csv(cld_emo, "E:\\SocialDetection7T\\Nahid_Project\\IS_RSA\\ROI_LMM\\cld_ROI4.csv" )


#add a horizontal line at 0 
abline(0,0)
qqnorm(res)

ggplot(data = data_bio_motion, aes(x = ROI, y = Similarity, colour= Treatment, fill= Treatment))+
  geom_boxplot(alpha= 0.5) +labs(x= "ROIs", y= "Similarity(z-transformed)") +
  theme_classic()