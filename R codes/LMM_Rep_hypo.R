library(lme4)
library(lmerTest)
library(multcompView)
library(car)
library(ggplot2)
library(DescTools)
library(emmeans)
library(multcomp)
library(report)

data_bio_motion= read.csv("E:\\SocialDetection7T\\Nahid_Project\\IS_RSA\\Rep_hypothesis\\Bio_motion\\bio_motion_long_lpz_plc.csv")
data_action_cat= read.csv("E:\\SocialDetection7T\\Nahid_Project\\IS_RSA\\Rep_hypothesis\\Action_cat\\action_cat_long_lpz_plc.csv")
data_emotion= read.csv("E:\\SocialDetection7T\\Nahid_Project\\IS_RSA\\Rep_hypothesis\\Emotion\\emotion_long_lpz_plc.csv")



data_action_cat$ROI= as.factor(data_action_cat$ROI) 
data_action_cat$Similarity= FisherZ(data_action_cat$Similarity)

#Build model

lmm_act<- lmer(Similarity~ ROI+Treatment + ROI*Treatment + (1|Sub), data = data_action_cat)
#check the summary
summary(lmm_act)

#make table of the summary
lmm_emo_report=report_table(lmm_emo)
#write csv
write.csv(lmm_emo_report, "E:\\SocialDetection7T\\Nahid_Project\\IS_RSA\\Rep_hypothesis\\Emotion\\lmm_result_emotion.csv")



#analysis of deviance
aod= Anova(lmm_act)
write.csv(aod, "E:\\SocialDetection7T\\Nahid_Project\\IS_RSA\\Rep_hypothesis\\Action_cat\\aod_action_cat.csv")

text_result= report_text(lmm_emo)
text_result
emm_emo= emmeans(lmm_act, pairwise~ ROI*Treatment, adjust= 'none')
# for bonf correction 0.05/ number of contrasts * 3 because we have 3 represenational hypothesis
bonf_thr= 0.05/nrow(emm_emo[["contrasts"]]@linfct)*3
letters= c('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i')
cld_emo=cld(emm_emo, alpha= bonf_thr, Letters=letters)
write.csv(cld_emo, "E:\\SocialDetection7T\\Nahid_Project\\IS_RSA\\Rep_hypothesis\\Emotion\\cld_emotion_lmm.csv" )


#add a horizontal line at 0 
abline(0,0)
qqnorm(res)

ggplot(data = data_bio_motion, aes(x = ROI, y = Similarity, colour= Treatment, fill= Treatment))+
  geom_boxplot(alpha= 0.5) +labs(x= "ROIs", y= "Similarity(z-transformed)") +
  theme_classic()
  
