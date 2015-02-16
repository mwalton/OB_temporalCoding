# TODO: Preliminary PLS modelling on nansosense data 2015/02/12
# 
# Author: M Eiden, PhD
###############################################################################
library(xlsx)
library(pls)

# set home dir
homeDir <- "/Users/michaelwalton/Google Drive/Evolved Machines/chemometrics"

# set background set for training
trainbackgr <- 'med'

# set background set for testing
testbackgr <- 'high'

# parse XLSX or use serialized data
parsing <- FALSE

if(parsing==TRUE){
	
	setwd(homeDir)	
	
	trainbackgr <- 'med'
	
	colors <- c('blue','green','red','yellow')
	
	for(c in seq(colors)){
		col <- colors[c]
		cat(col)
		cat(' : ')
		
		if(trainbackgr=='high'){
			if(col=='blue'){
				setwd("High BG Feb 9 Blue")
			} else if (col=='green'){
				setwd("High BG Feb 9 Green")
			} else if (col=='red'){
				setwd("High BG Feb 9 Red" )
			} else if (col=='yellow'){
				setwd("High BG Feb 9 Yellow")
			}		  
			
		} else if (trainbackgr=='med'){
			if(col=='blue'){
				setwd("Med BG2 Feb 9 Blue")
			} else if (col=='green'){
				setwd("Med BG2 Feb 9 Green")
			} else if (col=='red'){
				setwd("Med BG2 Feb 9 Red")
			} else if (col=='yellow'){
				setwd("Med BG2 Feb 9 Yellow")
			}	
		}
		
		cat(getwd())
		cat('\n')
		conc 	<- read.xlsx("concentration.xlsx", sheetIndex=1, header=F)
		act 	<- read.xlsx("sensorActivation.xlsx", sheetIndex=1, header=F)
		setwd('../')
		
		colnames(conc) <- c('BKG','RED','GRE','BLU','YEL')
		
		if(c==1){
			conc.final 	<- conc
			act.final 	<- act
		} else {
			conc.final 	<- rbind(conc.final, conc)
			act.final 	<- rbind(act.final, act)
		}
		
		rm(act)
		rm(conc)	
		
	}
	
} else {
	setwd(homeDir)
	setwd("20150212--PLS-modelling")
	load('data.RData')
}

# reformat
odorants 	<- as.matrix(conc.final[,-1])
act 		<- as.matrix(act.final)

# build 5 different predictive models and validate them with CV
sensor.pcr 			<- pcr(odorants ~ act, 10, validation = "CV")
sensor.pls 			<- plsr(odorants ~ act, 10, validation = "CV")
sensor.cppls 		<- cppls(odorants ~ act, 10, validation = "CV")
sensor.oscorespls 	<- mvr(odorants ~ act, 10, validation = "CV", method = "oscorespls")
sensor.simpls 		<- mvr(odorants ~ act, 10, validation = "CV",method = "simpls")

# print summary for an example model
summary(sensor.pcr)

# plot prediction on hold out validationb data
plot(sensor.pcr)

# load test data
if(testbackgr=='high'){
  setwd(homeDir)
	setwd("High BG Feb 9 Test")
	test.conc 	<- read.xlsx("concentration.xlsx", sheetIndex=1, header=F)
	test.act 	<- read.xlsx("sensorActivation.xlsx", sheetIndex=1, header=F)
	test.odorants 	<- as.matrix(test.conc[,-1])
	test.act 		<- as.matrix(test.act)
} else if (testbackgr=='med'){
  setwd(homeDir)  
	setwd("Med BG2 Feb 9 Test")
	test.conc 	<- read.xlsx("concentration.xlsx", sheetIndex=1, header=F)
	test.act 	<- read.xlsx("sensorActivation.xlsx", sheetIndex=1, header=F)
	test.odorants 	<- as.matrix(test.conc[,-1])
	test.act 		<- as.matrix(test.act)
}

#run prediction for PCR model
pred.sensor.pcr 	<- predict(sensor.pcr, newdata=test.act)
# ....

# slice out prediction matrix for 10 dcomponent model
pred.sensor.pcr.10 	<- pred.sensor.pcr[,,10]
#....

# plot test truth vs test prediction for 1 odorant
#plot(test.odorants[,1], pred.sensor.pcr.10[,1])
